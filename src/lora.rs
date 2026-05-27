use anyhow::Result;
use candle_core::{DType, Tensor};
use std::collections::HashMap;

pub fn apply_lora(
    mut tensors: HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    lora_scale: f64,
) -> Result<HashMap<String, Tensor>> {
    let mut applied = 0usize;

    // ── Pass 1: diffusers format ─────────────────────────────────────────────
    // UNet key "a.b.c.weight" → LoRA base "lora_unet_a_b_c"
    {
        let weight_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.ends_with(".weight"))
            .cloned()
            .collect();
        for weight_key in weight_keys {
            let base = weight_key.strip_suffix(".weight").expect("filtered");
            let lora_base = format!("lora_unet_{}", base.replace('.', "_"));
            if let Some(merged) =
                merge_lora_layer(&tensors, lora, &weight_key, &lora_base, lora_scale)?
            {
                tensors.insert(weight_key, merged);
                applied += 1;
            }
        }
    }

    // ── Pass 2: ldm / ComfyUI format ────────────────────────────────────────
    // LoRA base "lora_unet_input_blocks_N_M_..." → UNet key via block mapping.
    // Only runs when Pass 1 matched nothing (avoids double-applying).
    if applied == 0 {
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
            if !tensors.contains_key(&unet_key) {
                continue;
            }
            if let Some(merged) =
                merge_lora_layer(&tensors, lora, &unet_key, &lora_full_base, lora_scale)?
            {
                tensors.insert(unet_key, merged);
                applied += 1;
            }
        }
        if skipped > 0 {
            tracing::warn!("UNet LoRA: {skipped} ldm keys skipped (unmapped block indices)");
        }
    }

    if applied == 0 {
        let sample: Vec<_> = lora.keys().take(5).collect();
        anyhow::bail!("LoRA matched 0 UNet layers.\nFirst keys in file: {sample:?}");
    }
    tracing::info!("LoRA applied to {applied} layers");
    Ok(tensors)
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
    let (Some(lora_down), Some(lora_up)) = (
        lora.get(&format!("{lora_base}.lora_down.weight")),
        lora.get(&format!("{lora_base}.lora_up.weight")),
    ) else {
        return Ok(None);
    };
    let rank = lora_down.dim(0)?;
    let scale = if let Some(alpha) = lora.get(&format!("{lora_base}.alpha")) {
        lora_scale * alpha.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64 / rank as f64
    } else {
        lora_scale
    };
    let delta = (lora_up
        .to_dtype(DType::F32)?
        .matmul(&lora_down.to_dtype(DType::F32)?)?
        * scale)?;
    let w = tensors.get(weight_key).expect("caller checked");
    let orig = w.dtype();
    Ok(Some((w.to_dtype(DType::F32)? + delta)?.to_dtype(orig)?))
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
    } else if let Some(s) = base.strip_prefix("middle_block_") {
        let (n, s) = pop_num(s)?;
        let rest = s.strip_prefix('_').unwrap_or(s);
        (sdxl_middle_block(n)?, rest)
    } else {
        return None;
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
