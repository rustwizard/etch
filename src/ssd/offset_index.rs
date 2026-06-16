use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct SafetensorsHeaderValue {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub struct OffsetIndex {
    file_path: std::path::PathBuf,
    data_start: u64,
    tensors: HashMap<String, TensorMeta>,
}

impl OffsetIndex {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;

        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf);

        let mut header_json = vec![0u8; header_len as usize];
        file.read_exact(&mut header_json)?;
        let header: HashMap<String, SafetensorsHeaderValue> = serde_json::from_slice(&header_json)
            .with_context(|| format!("parsing safetensors header in {}", path.display()))?;

        let data_start = 8 + header_len;
        let mut tensors = HashMap::new();
        for (name, val) in header {
            tensors.insert(
                name,
                TensorMeta {
                    dtype: parse_dtype(&val.dtype)?,
                    shape: val.shape,
                    offset_bytes: data_start + val.data_offsets[0],
                    size_bytes: val.data_offsets[1] - val.data_offsets[0],
                },
            );
        }

        Ok(Self {
            file_path: path.to_path_buf(),
            data_start,
            tensors,
        })
    }

    pub fn tensor_meta(&self, name: &str) -> Result<&TensorMeta> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in {}", self.file_path.display()))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn read_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let meta = self.tensor_meta(name)?;
        let file = File::open(&self.file_path)
            .with_context(|| format!("opening {} for read", self.file_path.display()))?;
        let mut buf = vec![0u8; meta.size_bytes as usize];
        file.read_exact_at(&mut buf, meta.offset_bytes)
            .with_context(|| format!("reading tensor '{name}' at offset {}", meta.offset_bytes))?;
        Tensor::from_raw_buffer(&buf, meta.dtype, &meta.shape, device)
            .with_context(|| format!("creating tensor '{name}' from raw buffer"))
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "F64" => Ok(DType::F64),
        "U8" => Ok(DType::U8),
        "U32" => Ok(DType::U32),
        "I64" => Ok(DType::I64),
        other => anyhow::bail!("unsupported safetensors dtype: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_and_read_tensor() -> Result<()> {
        let path = std::env::temp_dir().join("etch_test_offset_index.safetensors");

        let header = r#"{"test":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, 24)
        };

        let mut file = std::fs::File::create(&path)?;
        use std::io::Write as _;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;
        file.write_all(data_bytes)?;
        file.flush()?;

        let index = OffsetIndex::from_file(&path)?;
        assert_eq!(index.tensor_count(), 1);

        let meta = index.tensor_meta("test")?;
        assert_eq!(meta.dtype, DType::F32);
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.size_bytes, 24);

        let tensor = index.read_tensor("test", &Device::Cpu)?;
        assert_eq!(tensor.dims(), &[2, 3]);
        let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
