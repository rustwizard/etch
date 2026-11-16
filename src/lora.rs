use anyhow::Result;
use candle_core::{DType, Tensor};
use std::collections::HashMap;

/// Merge one LoRA delta into a base weight: `base + scale * (up @ down)`,
/// computed in F32 and cast back to the base dtype. Shared by the eager
/// text-encoder path and the lazy UNet backend so both cannot diverge.
pub fn merge_lora_weight(
    w: &Tensor,
    lora_up: &Tensor,
    lora_down: &Tensor,
    scale: f64,
) -> candle_core::Result<Tensor> {
    let delta = (lora_up
        .to_dtype(DType::F32)?
        .matmul(&lora_down.to_dtype(DType::F32)?)?
        * scale)?;
    let orig = w.dtype();
    (w.to_dtype(DType::F32)? + delta)?.to_dtype(orig)
}

/// Effective merge scale for one LoRA layer: `lora_scale * alpha / rank` when
/// the file carries an alpha tensor, plain `lora_scale` otherwise.
fn merge_scale(
    lora: &HashMap<String, Tensor>,
    lora_base: &str,
    rank: usize,
    lora_scale: f64,
) -> Result<f64> {
    if let Some(alpha) = lora.get(&format!("{lora_base}.alpha")) {
        Ok(lora_scale * alpha.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64 / rank as f64)
    } else {
        Ok(lora_scale)
    }
}

/// A single planned UNet merge: which LoRA tensors patch `weight_key` and at
/// what scale. Resolved up front so the lazy backend never touches key
/// mapping and the zero-match case fails before the UNet is constructed.
#[derive(Debug)]
pub struct MergeSpec {
    pub up: String,
    pub down: String,
    pub scale: f64,
}

/// Pre-resolved LoRA-to-UNet match plan (see `build_unet_lora_plan`).
#[derive(Debug, Default)]
pub struct LoraPlan {
    merges: HashMap<String, MergeSpec>,
}

impl LoraPlan {
    pub fn applied(&self) -> usize {
        self.merges.len()
    }

    pub(crate) fn merges(&self) -> &HashMap<String, MergeSpec> {
        &self.merges
    }
}

/// Map a UNet LoRA file onto UNet weight names without loading any UNet
/// weights. Mirrors the original eager `apply_lora` semantics exactly:
/// diffusers-format keys first, the ldm/ComfyUI mapping only when the first
/// pass matched nothing, incomplete down/up pairs are not matches, and zero
/// matches are a hard error.
pub fn build_unet_lora_plan(
    unet_names: &[String],
    lora: &HashMap<String, Tensor>,
    lora_scale: f64,
) -> Result<LoraPlan> {
    let mut merges = HashMap::new();

    // ── Pass 1: diffusers format ────────────────────────────────────────────
    // UNet key "a.b.c.weight" → LoRA base "lora_unet_a_b_c"
    for weight_key in unet_names.iter().filter(|k| k.ends_with(".weight")) {
        let base = weight_key.strip_suffix(".weight").expect("filtered");
        let lora_base = format!("lora_unet_{}", base.replace('.', "_"));
        if let Some(spec) = plan_lora_layer(lora, &lora_base, lora_scale)? {
            merges.insert(weight_key.clone(), spec);
        }
    }

    // ── Pass 2: ldm / ComfyUI format ────────────────────────────────────────
    // LoRA base "lora_unet_input_blocks_N_M_..." → UNet key via block mapping.
    // Only runs when Pass 1 matched nothing (avoids double-applying).
    if merges.is_empty() {
        let unet_name_set: std::collections::HashSet<&String> = unet_names.iter().collect();
        let lora_bases: Vec<String> = lora
            .keys()
            .filter_map(|k| k.strip_suffix(".lora_down.weight"))
            .filter(|k| k.starts_with("lora_unet_"))
            .map(str::to_string)
            .collect();
        let mut skipped = 0usize;
        for lora_full_base in lora_bases {
            let ldm_base = &lora_full_base["lora_unet_".len()..];
            let Some(unet_key) = ldm_lora_base_to_unet_key(ldm_base) else {
                skipped += 1;
                continue;
            };
            if !unet_name_set.contains(&unet_key) {
                continue;
            }
            if let Some(spec) = plan_lora_layer(lora, &lora_full_base, lora_scale)? {
                merges.insert(unet_key, spec);
            }
        }
        if skipped > 0 {
            tracing::warn!("UNet LoRA: {skipped} ldm keys skipped (unmapped block indices)");
        }
    }

    if merges.is_empty() {
        let sample: Vec<_> = lora.keys().take(5).collect();
        anyhow::bail!("LoRA matched 0 UNet layers.\nFirst keys in file: {sample:?}");
    }
    tracing::info!("UNet LoRA will merge into {} layers", merges.len());
    Ok(LoraPlan { merges })
}

fn plan_lora_layer(
    lora: &HashMap<String, Tensor>,
    lora_base: &str,
    lora_scale: f64,
) -> Result<Option<MergeSpec>> {
    let down_key = format!("{lora_base}.lora_down.weight");
    let up_key = format!("{lora_base}.lora_up.weight");
    let (Some(lora_down), Some(_lora_up)) = (lora.get(&down_key), lora.get(&up_key)) else {
        return Ok(None);
    };
    let rank = lora_down.dim(0)?;
    let scale = merge_scale(lora, lora_base, rank, lora_scale)?;
    Ok(Some(MergeSpec {
        up: up_key,
        down: down_key,
        scale,
    }))
}

pub fn apply_te_lora(
    mut tensors: HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    te_prefix: &str,
    lora_scale: f64,
) -> Result<(HashMap<String, Tensor>, usize)> {
    let mut applied = 0usize;
    let lora_bases: Vec<String> = lora
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_down.weight"))
        .filter(|k| k.starts_with(te_prefix))
        .map(str::to_string)
        .collect();
    let mut skipped = 0usize;
    for lora_full_base in lora_bases {
        let inner = &lora_full_base[te_prefix.len()..];
        let Some(weight_key) = te_lora_base_to_weight_key(inner) else {
            skipped += 1;
            continue;
        };
        if !tensors.contains_key(&weight_key) {
            continue;
        }
        if let Some(merged) =
            merge_lora_layer(&tensors, lora, &weight_key, &lora_full_base, lora_scale)?
        {
            tensors.insert(weight_key, merged);
            applied += 1;
        }
    }
    if skipped > 0 {
        tracing::warn!("TE LoRA ({te_prefix}): {skipped} keys had no weight mapping");
    }
    Ok((tensors, applied))
}

fn merge_lora_layer(
    tensors: &HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    weight_key: &str,
    lora_base: &str,
    lora_scale: f64,
) -> Result<Option<Tensor>> {
    let down_key = format!("{lora_base}.lora_down.weight");
    let up_key = format!("{lora_base}.lora_up.weight");
    let (Some(lora_down), Some(lora_up)) = (lora.get(&down_key), lora.get(&up_key)) else {
        return Ok(None);
    };
    let rank = lora_down.dim(0)?;
    let scale = merge_scale(lora, lora_base, rank, lora_scale)?;
    let w = tensors.get(weight_key).expect("caller checked");
    Ok(Some(merge_lora_weight(w, lora_up, lora_down, scale)?))
}

/// Index over one or more safetensors files that reads tensors on demand
/// with positional reads instead of memory-mapping the files.
///
/// A long-lived full-file mmap keeps every touched page resident in the
/// process RSS for the mapping's lifetime, and model construction touches
/// practically all of them — so an mmaped base defeats the peak-RSS goal of
/// the LoRA path. Positional reads go through the page cache instead: the
/// bytes are copied into a small transient buffer, moved to the device, and
/// freed, so the process RSS grows by one tensor at a time, not by the whole
/// file.
struct StreamingTensors {
    files: Vec<std::fs::File>,
    index: HashMap<String, TensorLocation>,
}

struct TensorLocation {
    file: usize,
    /// Absolute byte offset of the tensor data in the file.
    offset: u64,
    len: u64,
    dtype: DType,
    shape: Vec<usize>,
}

impl StreamingTensors {
    fn open(paths: &[impl AsRef<std::path::Path>]) -> Result<Self> {
        use std::os::unix::fs::FileExt;
        let mut files = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (file_idx, path) in paths.iter().enumerate() {
            let path = path.as_ref();
            let file = std::fs::File::open(path)
                .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
            let header_len = {
                let mut buf = [0u8; 8];
                file.read_exact_at(&mut buf, 0).map_err(|e| {
                    anyhow::anyhow!("cannot read header of {}: {e}", path.display())
                })?;
                u64::from_le_bytes(buf)
            };
            let mut header = vec![0u8; header_len as usize];
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
            let data_base = 8 + header_len;
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
                index.insert(
                    name.clone(),
                    TensorLocation {
                        file: file_idx,
                        offset: data_base + offsets[0],
                        len: offsets[1] - offsets[0],
                        dtype,
                        shape,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, index })
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn read(&self, name: &str, dev: &candle_core::Device) -> candle_core::Result<Option<Tensor>> {
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

/// VarBuilder backend that applies planned LoRA merges lazily while the UNet
/// is constructed. Weights without a merge are streamed straight from the
/// base safetensors onto the target device (no mmap, no full CPU copy);
/// merged weights are loaded one layer at a time on the CPU, merged in F32,
/// and moved to the device. Peak memory during `prepare` is the resident
/// model plus one layer's worth of temporaries.
pub struct LazyLoraBackend {
    base: StreamingTensors,
    lora: std::sync::Arc<HashMap<String, Tensor>>,
    plan: LoraPlan,
}

impl LazyLoraBackend {
    /// Open the base safetensors files, build the merge plan from their
    /// tensor names, and return the backend plus the number of layers the
    /// plan applies to (for logging).
    pub fn open(
        base_paths: &[impl AsRef<std::path::Path>],
        lora: &std::sync::Arc<HashMap<String, Tensor>>,
        lora_scale: f64,
    ) -> Result<(Self, usize)> {
        let base = StreamingTensors::open(base_paths)?;
        let plan = build_unet_lora_plan(&base.names(), lora, lora_scale)?;
        let applied = plan.applied();
        Ok((
            Self {
                base,
                lora: std::sync::Arc::clone(lora),
                plan,
            },
            applied,
        ))
    }

    fn load_merged(&self, name: &str, dev: &candle_core::Device) -> candle_core::Result<Tensor> {
        match self.plan.merges().get(name) {
            None => self
                .base
                .read(name, dev)?
                .ok_or_else(|| candle_core::Error::msg(format!("tensor '{name}' not found"))),
            Some(spec) => {
                // One base layer at a time on the CPU; the merged tensor is
                // moved to the device right away.
                let base = self
                    .base
                    .read(name, &candle_core::Device::Cpu)?
                    .ok_or_else(|| candle_core::Error::msg(format!("tensor '{name}' not found")))?;
                let lora_up = self.lora.get(&spec.up).ok_or_else(|| {
                    candle_core::Error::msg(format!("LoRA tensor '{}' missing", spec.up))
                })?;
                let lora_down = self.lora.get(&spec.down).ok_or_else(|| {
                    candle_core::Error::msg(format!("LoRA tensor '{}' missing", spec.down))
                })?;
                merge_lora_weight(&base, lora_up, lora_down, spec.scale)?.to_device(dev)
            }
        }
    }
}

impl candle_nn::var_builder::SimpleBackend for LazyLoraBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_merged(name, dev)?.to_dtype(dtype)?;
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
    ) -> candle_core::Result<Tensor> {
        self.load_merged(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.base.contains(name)
    }
}

/// Converts the inner part of a TE LoRA base (after stripping "lora_te1_" / "lora_te2_")
/// to the weight key used in the safetensors file.
/// e.g. "text_model_encoder_layers_0_self_attn_q_proj" → "text_model.encoder.layers.0.self_attn.q_proj.weight"
fn te_lora_base_to_weight_key(base: &str) -> Option<String> {
    const TE_TOKENS: &[(&str, &str)] = &[
        ("text_model_", "text_model"),
        ("encoder_", "encoder"),
        ("layers_", "layers"),
        ("self_attn_", "self_attn"),
        ("mlp_", "mlp"),
        ("layer_norm1", "layer_norm1"),
        ("layer_norm2", "layer_norm2"),
        ("out_proj", "out_proj"),
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("fc1", "fc1"),
        ("fc2", "fc2"),
    ];
    let result = greedy_tokenize(base, TE_TOKENS);
    if result.is_empty() {
        return None;
    }
    Some(format!("{result}.weight"))
}

/// Converts a LoRA base in ldm/ComfyUI format to a diffusers UNet weight key.
/// e.g. "input_blocks_4_1_transformer_blocks_0_attn1_to_q"
///   → "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight"
fn ldm_lora_base_to_unet_key(base: &str) -> Option<String> {
    fn pop_num(s: &str) -> Option<(usize, &str)> {
        let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        if end == 0 {
            return None;
        }
        Some((s[..end].parse().ok()?, &s[end..]))
    }

    let (block_prefix, rest) = if let Some(s) = base.strip_prefix("input_blocks_") {
        let (n, s) = pop_num(s)?;
        let s = s.strip_prefix('_')?;
        let (m, s) = pop_num(s)?;
        let rest = s.strip_prefix('_').unwrap_or(s);
        (sdxl_input_block(n, m)?, rest)
    } else if let Some(s) = base.strip_prefix("output_blocks_") {
        let (n, s) = pop_num(s)?;
        let s = s.strip_prefix('_')?;
        let (m, s) = pop_num(s)?;
        let rest = s.strip_prefix('_').unwrap_or(s);
        (sdxl_output_block(n, m)?, rest)
    } else {
        let s = base.strip_prefix("middle_block_")?;
        let (n, s) = pop_num(s)?;
        let rest = s.strip_prefix('_').unwrap_or(s);
        (sdxl_middle_block(n)?, rest)
    };

    let unet_path = if rest.is_empty() {
        block_prefix
    } else {
        format!("{block_prefix}.{}", ldm_suffix_to_diffusers(rest))
    };
    Some(format!("{unet_path}.weight"))
}

fn sdxl_input_block(n: usize, m: usize) -> Option<String> {
    match (n, m) {
        (1, 0) => Some("down_blocks.0.resnets.0".into()),
        (2, 0) => Some("down_blocks.0.resnets.1".into()),
        (4, 0) => Some("down_blocks.1.resnets.0".into()),
        (4, 1) => Some("down_blocks.1.attentions.0".into()),
        (5, 0) => Some("down_blocks.1.resnets.1".into()),
        (5, 1) => Some("down_blocks.1.attentions.1".into()),
        (7, 0) => Some("down_blocks.2.resnets.0".into()),
        (7, 1) => Some("down_blocks.2.attentions.0".into()),
        (8, 0) => Some("down_blocks.2.resnets.1".into()),
        (8, 1) => Some("down_blocks.2.attentions.1".into()),
        _ => None,
    }
}

fn sdxl_output_block(n: usize, m: usize) -> Option<String> {
    let (up_block, layer) = match n {
        0..=2 => (0, n),
        3..=5 => (1, n - 3),
        6..=8 => (2, n - 6),
        _ => return None,
    };
    match m {
        0 => Some(format!("up_blocks.{up_block}.resnets.{layer}")),
        1 => Some(format!("up_blocks.{up_block}.attentions.{layer}")),
        _ => None,
    }
}

fn sdxl_middle_block(n: usize) -> Option<String> {
    match n {
        0 => Some("mid_block.resnets.0".into()),
        1 => Some("mid_block.attentions.0".into()),
        2 => Some("mid_block.resnets.1".into()),
        _ => None,
    }
}

/// Greedy underscore-to-dot tokenizer shared by UNet and TE LoRA key converters.
/// Scans `s` left-to-right, replacing known multi-word tokens first (longest-match
/// via table order), then falling back to consuming one `_`-delimited segment.
fn greedy_tokenize(mut s: &str, tokens: &[(&str, &str)]) -> String {
    let mut result = String::new();
    while !s.is_empty() {
        let matched = tokens.iter().find_map(|&(ldm, diff)| {
            s.starts_with(ldm).then(|| {
                s = &s[ldm.len()..];
                diff
            })
        });
        match matched {
            Some(tok) => {
                if !result.is_empty() {
                    result.push('.');
                }
                result.push_str(tok);
            }
            None => {
                let end = s.find('_').unwrap_or(s.len());
                if !result.is_empty() {
                    result.push('.');
                }
                result.push_str(&s[..end]);
                s = if end < s.len() { &s[end + 1..] } else { "" };
            }
        }
    }
    result
}

/// Converts the module-path portion of an ldm LoRA key to diffusers dot-notation.
/// e.g. "transformer_blocks_0_attn1_to_out_0" → "transformer_blocks.0.attn1.to_out.0"
fn ldm_suffix_to_diffusers(s: &str) -> String {
    const TOKENS: &[(&str, &str)] = &[
        ("transformer_blocks_", "transformer_blocks"),
        ("to_out_", "to_out"),
        ("ff_net_", "ff.net"),
        ("attn1_", "attn1"),
        ("attn2_", "attn2"),
        ("proj_out", "proj_out"),
        ("proj_in", "proj_in"),
        ("to_out", "to_out"),
        ("to_q", "to_q"),
        ("to_k", "to_k"),
        ("to_v", "to_v"),
        ("norm1", "norm1"),
        ("norm2", "norm2"),
        ("norm3", "norm3"),
    ];
    greedy_tokenize(s, TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight
    const W_KEY: &str = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight";
    const LORA_BASE: &str = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q";

    fn t2(data: &[f32], d0: usize, d1: usize) -> Tensor {
        Tensor::from_slice(data, (d0, d1), &Device::Cpu).expect("test tensor")
    }

    /// rank=4, in=8 down; out=3, rank=4 up; valid pair for a (3, 8) weight.
    fn diffusers_lora(alpha: Option<f32>) -> HashMap<String, Tensor> {
        let mut lora = HashMap::new();
        lora.insert(
            format!("{LORA_BASE}.lora_down.weight"),
            // One-hot rows: row r has 0.5 in column r, so delta lands 0.5 on
            // the diagonal and expectations stay obvious.
            t2(
                &[
                    0.5, 0., 0., 0., 0., 0., 0., 0., //
                    0., 0.5, 0., 0., 0., 0., 0., 0., //
                    0., 0., 0.5, 0., 0., 0., 0., 0., //
                    0., 0., 0., 0.5, 0., 0., 0., 0.,
                ],
                4,
                8,
            ),
        );
        lora.insert(
            format!("{LORA_BASE}.lora_up.weight"),
            t2(&[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], 3, 4),
        );
        if let Some(a) = alpha {
            lora.insert(
                format!("{LORA_BASE}.alpha"),
                Tensor::new(a, &Device::Cpu).expect("alpha tensor"),
            );
        }
        lora
    }

    fn unet_names() -> Vec<String> {
        vec![W_KEY.to_string(), "other.weight".to_string()]
    }

    #[test]
    fn plan_diffusers_format_with_alpha() {
        let lora = diffusers_lora(Some(2.0));
        let plan = build_unet_lora_plan(&unet_names(), &lora, 0.5).expect("plan builds");
        assert_eq!(plan.applied(), 1);
        let spec = plan.merges().get(W_KEY).expect("W_KEY planned");
        assert_eq!(spec.up, format!("{LORA_BASE}.lora_up.weight"));
        assert_eq!(spec.down, format!("{LORA_BASE}.lora_down.weight"));
        // scale = lora_scale * alpha / rank = 0.5 * 2 / 4
        assert_eq!(spec.scale, 0.25);
    }

    #[test]
    fn plan_without_alpha_uses_lora_scale() {
        let lora = diffusers_lora(None);
        let plan = build_unet_lora_plan(&unet_names(), &lora, 0.75).expect("plan builds");
        assert_eq!(plan.merges().get(W_KEY).expect("planned").scale, 0.75);
    }

    #[test]
    fn plan_zero_matches_errors() {
        let mut lora = HashMap::new();
        lora.insert(
            "lora_unet_something_else.lora_down.weight".to_string(),
            t2(&[1.], 1, 1),
        );
        lora.insert(
            "lora_unet_something_else.lora_up.weight".to_string(),
            t2(&[1.], 1, 1),
        );
        let err =
            build_unet_lora_plan(&unet_names(), &lora, 1.0).expect_err("zero matches must fail");
        assert!(
            format!("{err:#}").contains("matched 0 UNet layers"),
            "{err}"
        );
    }

    #[test]
    fn plan_incomplete_pair_is_not_a_match() {
        let mut lora = diffusers_lora(None);
        lora.remove(&format!("{LORA_BASE}.lora_up.weight"));
        assert!(build_unet_lora_plan(&unet_names(), &lora, 1.0).is_err());
    }

    #[test]
    fn plan_ldm_format_fallback() {
        let mut lora = HashMap::new();
        let base = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q";
        lora.insert(format!("{base}.lora_down.weight"), t2(&[0.5; 32], 4, 8));
        lora.insert(format!("{base}.lora_up.weight"), t2(&[1.; 12], 3, 4));
        let plan = build_unet_lora_plan(&unet_names(), &lora, 1.0).expect("plan builds");
        assert_eq!(plan.applied(), 1);
        assert!(plan.merges().contains_key(W_KEY), "ldm key maps to {W_KEY}");
    }

    #[test]
    fn plan_diffusers_wins_over_mixed_ldm() {
        let mut lora = diffusers_lora(None);
        let ldm_base = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q";
        lora.insert(format!("{ldm_base}.lora_down.weight"), t2(&[0.5; 32], 4, 8));
        lora.insert(format!("{ldm_base}.lora_up.weight"), t2(&[1.; 12], 3, 4));
        let plan = build_unet_lora_plan(&unet_names(), &lora, 1.0).expect("plan builds");
        // Pass 1 matched, so the ldm keys must be ignored (no double-apply).
        assert_eq!(plan.applied(), 1);
        assert!(plan.merges().contains_key(W_KEY));
    }

    #[test]
    fn merge_math_is_exact_for_dyadic_values() {
        // up @ down with the diffusers_lora fixture: each up row picks the
        // matching down row (identity-like up), so delta rows equal down
        // rows 0..2; scale 0.5 on an all-ones base.
        let lora = diffusers_lora(None);
        let up = lora
            .get(&format!("{LORA_BASE}.lora_up.weight"))
            .expect("up");
        let down = lora
            .get(&format!("{LORA_BASE}.lora_down.weight"))
            .expect("down");
        let w = t2(&[1.; 24], 3, 8);
        let merged = merge_lora_weight(&w, up, down, 0.5).expect("merge");
        let got = merged.to_vec2::<f32>().expect("vec2");
        // merged = 1 + 0.5 * delta, delta diagonal = 0.5 (rows 0..2 of up
        // pick down rows 0..2).
        assert_eq!(got[0], vec![1.25, 1., 1., 1., 1., 1., 1., 1.]);
        assert_eq!(got[1], vec![1., 1.25, 1., 1., 1., 1., 1., 1.]);
        assert_eq!(got[2], vec![1., 1., 1.25, 1., 1., 1., 1., 1.]);
        // Base dtype is restored.
        assert_eq!(merged.dtype(), w.dtype());
    }

    #[test]
    fn lazy_backend_matches_eager_merge() {
        let dir = std::env::temp_dir().join(format!(
            "etch-lora-test-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");

        // A tiny safetensors file standing in for the UNet: one LoRA target
        // and one untouched weight.
        let w = t2(&[1.; 24], 3, 8);
        let plain = t2(&[7.; 6], 2, 3);
        let mut base_map = HashMap::new();
        base_map.insert(W_KEY.to_string(), w.clone());
        base_map.insert("plain.weight".to_string(), plain.clone());
        let st_path = dir.join("unet.safetensors");
        candle_core::safetensors::save(&base_map, &st_path).expect("save base");

        let lora = std::sync::Arc::new(diffusers_lora(None));
        let (backend, applied) =
            LazyLoraBackend::open(&[st_path], &lora, 0.5).expect("open backend");
        assert_eq!(applied, 1, "one planned merge");

        use candle_nn::var_builder::SimpleBackend as _;
        let got = backend
            .get(
                candle_core::Shape::from((3, 8)),
                W_KEY,
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect("merged get");
        let want = merge_lora_weight(
            &w,
            &diffusers_lora(None)[&format!("{LORA_BASE}.lora_up.weight")],
            &diffusers_lora(None)[&format!("{LORA_BASE}.lora_down.weight")],
            0.5,
        )
        .expect("eager merge");
        assert_eq!(
            got.to_vec2::<f32>().expect("vec2"),
            want.to_vec2::<f32>().expect("vec2"),
            "lazy merge must equal eager merge"
        );

        // Unmerged weights pass through untouched.
        let got_plain = backend
            .get(
                candle_core::Shape::from((2, 3)),
                "plain.weight",
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect("plain get");
        assert_eq!(
            got_plain.to_vec2::<f32>().expect("vec2"),
            plain.to_vec2::<f32>().expect("vec2")
        );
        assert!(backend.contains_tensor("plain.weight"));
        assert!(!backend.contains_tensor("absent.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lazy_backend_shape_error_names_tensor() {
        let dir = std::env::temp_dir().join(format!(
            "etch-lora-shape-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut base_map = HashMap::new();
        base_map.insert(W_KEY.to_string(), t2(&[1.; 24], 3, 8));
        let st_path = dir.join("unet.safetensors");
        candle_core::safetensors::save(&base_map, &st_path).expect("save base");

        let lora = std::sync::Arc::new(diffusers_lora(None));
        let (backend, _) = LazyLoraBackend::open(&[st_path], &lora, 1.0).expect("open backend");

        use candle_nn::var_builder::SimpleBackend as _;
        let err = backend
            .get(
                candle_core::Shape::from((4, 4)),
                W_KEY,
                candle_nn::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )
            .expect_err("wrong shape must fail");
        assert!(format!("{err:?}").contains("shape mismatch for"), "{err:?}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
