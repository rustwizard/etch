//! Streaming (mmap-free) reading of safetensors weights.
//!
//! A long-lived full-file mmap keeps every touched page resident in the
//! process RSS for the mapping's lifetime, and model construction touches
//! practically all of them — so an mmaped base defeats the peak-RSS goal of
//! the LoRA path. Positional reads go through the page cache instead: the
//! bytes are copied into a small transient buffer, moved to the device, and
//! freed, so the process RSS grows by one tensor at a time, not by the whole
//! file.
//!
//! [`StreamingBackend`] is the plain variant used for the SDXL UNet without
//! LoRA; [`crate::lora::LazyLoraBackend`] layers the lazy LoRA merge on top
//! of the same loader so validation cannot diverge.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Index over one or more safetensors files that reads tensors on demand
/// with positional reads instead of memory-mapping the files.
pub struct StreamingTensors {
    files: Vec<std::fs::File>,
    index: std::collections::HashMap<String, TensorLocation>,
}

struct TensorLocation {
    file: usize,
    /// Absolute byte offset of the tensor data in the file.
    offset: u64,
    len: u64,
    dtype: DType,
    shape: Vec<usize>,
}

/// Generous upper bound for the JSON header. Real safetensors headers are a
/// few MB at most; anything larger is either corrupt or hostile.
const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

impl StreamingTensors {
    pub fn open(paths: &[impl AsRef<std::path::Path>]) -> Result<Self> {
        use std::os::unix::fs::FileExt;
        let mut files = Vec::with_capacity(paths.len());
        let mut index = std::collections::HashMap::new();
        for (file_idx, path) in paths.iter().enumerate() {
            let path = path.as_ref();
            let file = std::fs::File::open(path)
                .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
            let file_len = file
                .metadata()
                .map_err(|e| anyhow::anyhow!("cannot stat {}: {e}", path.display()))?
                .len();
            if file_len < 8 {
                anyhow::bail!("{}: too small to be a safetensors file", path.display());
            }
            let header_len = {
                let mut buf = [0u8; 8];
                file.read_exact_at(&mut buf, 0).map_err(|e| {
                    anyhow::anyhow!("cannot read header of {}: {e}", path.display())
                })?;
                u64::from_le_bytes(buf)
            };
            if header_len > MAX_HEADER_LEN {
                anyhow::bail!(
                    "{}: safetensors header too large ({header_len} bytes)",
                    path.display()
                );
            }
            let data_base = 8u64.checked_add(header_len).ok_or_else(|| {
                anyhow::anyhow!("{}: safetensors header length overflows", path.display())
            })?;
            if data_base > file_len {
                anyhow::bail!(
                    "{}: truncated file (header of {header_len} bytes, file is {file_len} bytes)",
                    path.display()
                );
            }
            let mut header = vec![
                0u8;
                usize::try_from(header_len).map_err(|_| {
                    anyhow::anyhow!("{}: safetensors header too large", path.display())
                })?
            ];
            file.read_exact_at(&mut header, 8)
                .map_err(|e| anyhow::anyhow!("cannot read header of {}: {e}", path.display()))?;
            let header: serde_json::Value = serde_json::from_slice(&header).map_err(|e| {
                anyhow::anyhow!("bad safetensors header in {}: {e}", path.display())
            })?;
            let Some(entries) = header.as_object() else {
                anyhow::bail!(
                    "bad safetensors header in {}: not an object",
                    path.display()
                );
            };
            for (name, meta) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = meta.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                    anyhow::anyhow!("{}: tensor '{name}' has no dtype", path.display())
                })?;
                let dtype = dtype_from_str(dtype).ok_or_else(|| {
                    anyhow::anyhow!(
                        "{}: tensor '{name}' has unsupported dtype '{dtype}'",
                        path.display()
                    )
                })?;
                let shape: Vec<usize> = meta
                    .get("shape")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .ok_or_else(|| {
                        anyhow::anyhow!("{}: tensor '{name}' has no shape", path.display())
                    })?;
                let offsets: [u64; 2] = meta
                    .get("data_offsets")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .ok_or_else(|| {
                        anyhow::anyhow!("{}: tensor '{name}' has no data_offsets", path.display())
                    })?;
                // All offset/size arithmetic is checked: a corrupt file must
                // produce a readable error, never a panic or a giant
                // allocation.
                let [start, end] = offsets;
                if start > end {
                    anyhow::bail!(
                        "{}: tensor '{name}' has reversed data offsets [{start}, {end}]",
                        path.display()
                    );
                }
                let abs_end = data_base.checked_add(end).ok_or_else(|| {
                    anyhow::anyhow!("{}: tensor '{name}' data end overflows", path.display())
                })?;
                if abs_end > file_len {
                    anyhow::bail!(
                        "{}: tensor '{name}' data [{start}, {end}] extends past end of file ({file_len} bytes)",
                        path.display()
                    );
                }
                let numel: u64 = shape
                    .iter()
                    .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
                    .ok_or_else(|| {
                        anyhow::anyhow!("{}: tensor '{name}' shape overflows", path.display())
                    })?;
                let expected_len =
                    numel
                        .checked_mul(dtype.size_in_bytes() as u64)
                        .ok_or_else(|| {
                            anyhow::anyhow!("{}: tensor '{name}' size overflows", path.display())
                        })?;
                let len = end - start;
                if len != expected_len {
                    anyhow::bail!(
                        "{}: tensor '{name}' has {len} data bytes, expected {expected_len} for shape {shape:?} and dtype {dtype:?}",
                        path.display()
                    );
                }
                index.insert(
                    name.clone(),
                    TensorLocation {
                        file: file_idx,
                        offset: data_base + start,
                        len,
                        dtype,
                        shape,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    pub fn read(
        &self,
        name: &str,
        dev: &candle_core::Device,
    ) -> candle_core::Result<Option<Tensor>> {
        use std::os::unix::fs::FileExt;
        let Some(loc) = self.index.get(name) else {
            return Ok(None);
        };
        let mut buf = vec![0u8; loc.len as usize];
        self.files[loc.file]
            .read_exact_at(&mut buf, loc.offset)
            .map_err(|e| {
                candle_core::Error::wrap(std::io::Error::new(
                    e.kind(),
                    format!("cannot read tensor '{name}': {e}"),
                ))
            })?;
        Ok(Some(Tensor::from_raw_buffer(
            &buf, loc.dtype, &loc.shape, dev,
        )?))
    }
}

fn dtype_from_str(s: &str) -> Option<DType> {
    Some(match s {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "U8" => DType::U8,
        "U32" => DType::U32,
        _ => return None,
    })
}

/// VarBuilder backend that streams base weights from safetensors with
/// positional reads (no mmap, no full CPU copy), one tensor at a time,
/// straight onto the target device. Used for the SDXL UNet when no LoRA is
/// requested.
pub struct StreamingBackend {
    tensors: StreamingTensors,
}

impl StreamingBackend {
    pub fn open(paths: &[impl AsRef<std::path::Path>]) -> Result<Self> {
        Ok(Self {
            tensors: StreamingTensors::open(paths)?,
        })
    }
}

impl candle_nn::var_builder::SimpleBackend for StreamingBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let tensor = self
            .tensors
            .read(name, dev)?
            .ok_or_else(|| candle_core::Error::msg(format!("tensor '{name}' not found")))?
            .to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            })
        } else {
            Ok(tensor)
        }
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.tensors
            .read(name, dev)?
            .ok_or_else(|| candle_core::Error::msg(format!("tensor '{name}' not found")))
            .and_then(|t| t.to_dtype(dtype))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t2(data: &[f32], d1: usize, d2: usize) -> Tensor {
        Tensor::from_slice(data, (d1, d2), &Device::Cpu).expect("test tensor")
    }

    /// Hand-craft a raw safetensors file: 8-byte LE header length + JSON
    /// header + tensor data.
    fn raw_st_file(header: &serde_json::Value, data: &[u8]) -> Vec<u8> {
        let header = serde_json::to_vec(header).expect("serialize header");
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(data);
        file
    }

    fn open_raw(dir: &std::path::Path, bytes: &[u8]) -> Result<StreamingTensors> {
        let path = dir.join("corrupt.safetensors");
        std::fs::write(&path, bytes).expect("write corrupt file");
        StreamingTensors::open(&[path])
    }

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "etch-st-{tag}-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// StreamingTensors has no Debug impl, so expect_err does not apply.
    fn expect_open_err(dir: &std::path::Path, bytes: &[u8], what: &str) -> anyhow::Error {
        match open_raw(dir, bytes) {
            Ok(_) => panic!("{what}: corrupt file must be rejected"),
            Err(e) => e,
        }
    }

    #[test]
    fn streaming_rejects_oversized_header() {
        let dir = unique_dir("oversized");
        // header_len = u64::MAX would try to allocate ~18 exabytes.
        let err = expect_open_err(&dir, &u64::MAX.to_le_bytes(), "oversized header");
        assert!(format!("{err:#}").contains("header too large"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_rejects_reversed_offsets() {
        let dir = unique_dir("reversed");
        let header = serde_json::json!({
            "t": {"dtype": "F32", "shape": [1], "data_offsets": [8, 0]},
        });
        let err = expect_open_err(&dir, &raw_st_file(&header, &[0u8; 8]), "reversed offsets");
        assert!(
            format!("{err:#}").contains("reversed data offsets"),
            "{err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_rejects_truncated_tensor() {
        let dir = unique_dir("truncated");
        // Declares 16 bytes of data but the file ends after 4.
        let header = serde_json::json!({
            "t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        });
        let err = expect_open_err(&dir, &raw_st_file(&header, &[0u8; 4]), "truncated tensor");
        assert!(
            format!("{err:#}").contains("extends past end of file"),
            "{err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_rejects_size_mismatch() {
        let dir = unique_dir("mismatch");
        // 4 F32 values need 16 bytes; the entry claims 8.
        let header = serde_json::json!({
            "t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]},
        });
        let err = expect_open_err(&dir, &raw_st_file(&header, &[0u8; 8]), "size mismatch");
        assert!(
            format!("{err:#}").contains("data bytes, expected"),
            "{err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_reads_valid_file() {
        let dir = unique_dir("valid");
        let header = serde_json::json!({
            "t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        });
        let st = open_raw(
            &dir,
            &raw_st_file(
                &header,
                &1.0f32
                    .to_le_bytes()
                    .into_iter()
                    .chain(2.0f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        )
        .expect("valid file opens");
        let t = st.read("t", &Device::Cpu).expect("read").expect("present");
        assert_eq!(t.to_vec1::<f32>().expect("vec"), vec![1.0, 2.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Build a small safetensors file with one F32 and one BF16 tensor.
    fn sample_file(dir: &std::path::Path) -> std::path::PathBuf {
        let mut map = std::collections::HashMap::new();
        map.insert("w_f32".to_string(), t2(&[1., 2., 3., 4.], 2, 2));
        map.insert(
            "w_bf16".to_string(),
            t2(&[0.5, 1.5], 1, 2).to_dtype(DType::BF16).expect("bf16"),
        );
        let path = dir.join("sample.safetensors");
        candle_core::safetensors::save(&map, &path).expect("save sample");
        path
    }

    #[test]
    fn streaming_backend_matches_mmap() {
        let dir = unique_dir("mmap-cmp");
        let path = sample_file(&dir);
        let backend = StreamingBackend::open(&[path.clone()]).expect("open streaming backend");
        // SAFETY: the test file is not modified while mapped.
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }
            .expect("mmap sample");

        use candle_nn::var_builder::SimpleBackend as _;
        for (name, shape) in [("w_f32", (2, 2)), ("w_bf16", (1, 2))] {
            let streamed = backend
                .get(
                    candle_core::Shape::from(shape),
                    name,
                    candle_nn::Init::Const(0.),
                    DType::F32,
                    &Device::Cpu,
                )
                .expect("streamed get");
            let mapped = mmap.load(name, &Device::Cpu).expect("mmap load");
            assert_eq!(
                streamed.to_vec2::<f32>().expect("vec"),
                mapped
                    .to_dtype(DType::F32)
                    .expect("cast")
                    .to_vec2::<f32>()
                    .expect("vec"),
                "{name}: streaming and mmap must agree"
            );
        }

        assert!(backend.contains_tensor("w_f32"));
        assert!(!backend.contains_tensor("absent"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_backend_missing_tensor_errors() {
        let dir = unique_dir("missing");
        let path = sample_file(&dir);
        let backend = StreamingBackend::open(&[path]).expect("open");
        use candle_nn::var_builder::SimpleBackend as _;
        let err = backend
            .get(
                candle_core::Shape::from((1, 1)),
                "nope",
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect_err("missing tensor must error");
        assert!(format!("{err}").contains("'nope' not found"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_backend_shape_error_names_tensor() {
        let dir = unique_dir("shape");
        let path = sample_file(&dir);
        let backend = StreamingBackend::open(&[path]).expect("open");
        use candle_nn::var_builder::SimpleBackend as _;
        let err = backend
            .get(
                candle_core::Shape::from((4, 4)),
                "w_f32",
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect_err("shape mismatch must error");
        assert!(format!("{err}").contains("w_f32"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_backend_bf16_conversion() {
        let dir = unique_dir("bf16");
        let path = sample_file(&dir);
        let backend = StreamingBackend::open(&[path]).expect("open");
        use candle_nn::var_builder::SimpleBackend as _;
        let t = backend
            .get(
                candle_core::Shape::from((1, 2)),
                "w_bf16",
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect("bf16 get as f32");
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.to_vec2::<f32>().expect("vec"), vec![vec![0.5, 1.5]]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
