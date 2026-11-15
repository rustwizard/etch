//! Parse a diffusers `unet/config.json` into candle's `UNet2DConditionModelConfig`.
//!
//! Both UNet load paths (with and without LoRA) must use the architecture
//! declared by the model itself instead of hardcoded Araminta/SDXL-base
//! values, otherwise a local `--local-model` with a different UNet silently
//! gets the wrong block layout.
use anyhow::{Context, Result};
use candle_transformers::models::stable_diffusion::unet_2d;
use std::path::Path;

/// UNet parameters taken from `unet/config.json`. Channel counts come along
/// with the architecture config because candle's model constructor takes
/// them separately.
#[derive(Debug)]
pub struct UNetConfigFile {
    pub config: unet_2d::UNet2DConditionModelConfig,
    pub in_channels: usize,
    pub out_channels: usize,
}

pub fn load_unet_config(path: &Path) -> Result<UNetConfigFile> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;
    parse_unet_config(&json)
        .with_context(|| format!("unsupported UNet config in {}", path.display()))
}

fn parse_unet_config(json: &serde_json::Value) -> Result<UNetConfigFile> {
    let in_channels = get_usize(json, "in_channels")?;
    let out_channels = get_usize(json, "out_channels")?;
    let block_out_channels = get_usize_array(json, "block_out_channels")?;
    let down_block_types = get_str_array(json, "down_block_types")?;
    anyhow::ensure!(
        block_out_channels.len() == down_block_types.len(),
        "block_out_channels ({} entries) and down_block_types ({} entries) must have the same length",
        block_out_channels.len(),
        down_block_types.len(),
    );
    let attention_head_dim = get_int_or_array(json, "attention_head_dim")?;
    let transformer_layers = get_int_or_array(json, "transformer_layers_per_block")?;
    let at = |i: usize, what: &str, values: &[usize]| -> Result<usize> {
        match values.len() {
            1 => Ok(values[0]),
            n if n == block_out_channels.len() => Ok(values[i]),
            n => anyhow::bail!(
                "{what} must be a single int or one per block ({}), got {n}",
                block_out_channels.len()
            ),
        }
    };

    let mut blocks = Vec::with_capacity(block_out_channels.len());
    for (i, kind) in down_block_types.iter().enumerate() {
        // Candle models cross-attention blocks; plain "DownBlock2D" has no
        // transformer at all. Self-attention-only blocks ("AttnDownBlock2D")
        // have no candle equivalent.
        let use_cross_attn = match *kind {
            "DownBlock2D" => None,
            "CrossAttnDownBlock2D" => {
                Some(at(i, "transformer_layers_per_block", &transformer_layers)?)
            }
            other => anyhow::bail!("unsupported down_block_type '{other}'"),
        };
        blocks.push(unet_2d::BlockConfig {
            out_channels: block_out_channels[i],
            use_cross_attn,
            attention_head_dim: at(i, "attention_head_dim", &attention_head_dim)?,
        });
    }

    Ok(UNetConfigFile {
        config: unet_2d::UNet2DConditionModelConfig {
            blocks,
            center_input_sample: get_bool_or(json, "center_input_sample", false)?,
            cross_attention_dim: get_usize(json, "cross_attention_dim")?,
            downsample_padding: get_usize_or(json, "downsample_padding", 1)?,
            flip_sin_to_cos: get_bool(json, "flip_sin_to_cos")?,
            freq_shift: get_f64(json, "freq_shift")?,
            layers_per_block: get_usize(json, "layers_per_block")?,
            mid_block_scale_factor: get_f64(json, "mid_block_scale_factor")?,
            norm_eps: get_f64(json, "norm_eps")?,
            norm_num_groups: get_usize(json, "norm_num_groups")?,
            // Matches the previous behavior: etch does not expose sliced
            // attention, and it is not part of the diffusers config.
            sliced_attention_size: None,
            use_linear_projection: get_bool(json, "use_linear_projection")?,
        },
        in_channels,
        out_channels,
    })
}

fn field<'v>(json: &'v serde_json::Value, key: &str) -> Result<&'v serde_json::Value> {
    json.get(key)
        .with_context(|| format!("missing field '{key}'"))
}

fn get_usize(json: &serde_json::Value, key: &str) -> Result<usize> {
    let v = field(json, key)?;
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .with_context(|| format!("field '{key}' must be a non-negative integer, got {v}"))
}

fn get_usize_or(json: &serde_json::Value, key: &str, default: usize) -> Result<usize> {
    match json.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(_) => get_usize(json, key),
    }
}

fn get_f64(json: &serde_json::Value, key: &str) -> Result<f64> {
    let v = field(json, key)?;
    v.as_f64()
        .with_context(|| format!("field '{key}' must be a number, got {v}"))
}

fn get_bool(json: &serde_json::Value, key: &str) -> Result<bool> {
    let v = field(json, key)?;
    v.as_bool()
        .with_context(|| format!("field '{key}' must be a boolean, got {v}"))
}

fn get_bool_or(json: &serde_json::Value, key: &str, default: bool) -> Result<bool> {
    match json.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(_) => get_bool(json, key),
    }
}

fn get_usize_array(json: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
    let v = field(json, key)?;
    v.as_array()
        .with_context(|| format!("field '{key}' must be an array, got {v}"))?
        .iter()
        .map(|e| {
            e.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .with_context(|| format!("field '{key}' must contain integers, got {e}"))
        })
        .collect()
}

fn get_str_array<'v>(json: &'v serde_json::Value, key: &str) -> Result<Vec<&'v str>> {
    let v = field(json, key)?;
    v.as_array()
        .with_context(|| format!("field '{key}' must be an array, got {v}"))?
        .iter()
        .map(|e| {
            e.as_str()
                .with_context(|| format!("field '{key}' must contain strings, got {e}"))
        })
        .collect()
}

/// diffusers writes some fields either as a single int (applied to every
/// block) or as an array with one entry per block.
fn get_int_or_array(json: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
    match field(json, key)? {
        serde_json::Value::Number(_) => Ok(vec![get_usize(json, key)?]),
        serde_json::Value::Array(_) => get_usize_array(json, key),
        v => anyhow::bail!("field '{key}' must be an int or an array of ints, got {v}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// stabilityai/stable-diffusion-xl-base-1.0 unet/config.json shape.
    fn sdxl_config_json() -> serde_json::Value {
        serde_json::json!({
            "_class_name": "UNet2DConditionModel",
            "act_fn": "silu",
            "addition_embed_type": "text_time",
            "block_out_channels": [320, 640, 1280],
            "cross_attention_dim": 2048,
            "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
            "downsample_padding": 1,
            "flip_sin_to_cos": true,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-5,
            "norm_num_groups": 32,
            "out_channels": 4,
            "attention_head_dim": [5, 10, 20],
            "transformer_layers_per_block": [1, 2, 10],
            "use_linear_projection": true
        })
    }

    #[test]
    fn parses_sdxl_unet_config() {
        let parsed = parse_unet_config(&sdxl_config_json()).expect("SDXL config parses");
        let cfg = &parsed.config;
        assert_eq!(parsed.in_channels, 4);
        assert_eq!(parsed.out_channels, 4);
        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[0].out_channels, 320);
        assert_eq!(cfg.blocks[0].use_cross_attn, None);
        assert_eq!(cfg.blocks[0].attention_head_dim, 5);
        assert_eq!(cfg.blocks[1].use_cross_attn, Some(2));
        assert_eq!(cfg.blocks[1].attention_head_dim, 10);
        assert_eq!(cfg.blocks[2].use_cross_attn, Some(10));
        assert_eq!(cfg.blocks[2].attention_head_dim, 20);
        assert_eq!(cfg.cross_attention_dim, 2048);
        assert_eq!(cfg.layers_per_block, 2);
        assert!(!cfg.center_input_sample);
        assert!(cfg.flip_sin_to_cos);
        assert_eq!(cfg.freq_shift, 0.0);
        assert_eq!(cfg.mid_block_scale_factor, 1.0);
        assert_eq!(cfg.norm_eps, 1e-5);
        assert_eq!(cfg.norm_num_groups, 32);
        assert!(cfg.use_linear_projection);
        assert_eq!(cfg.sliced_attention_size, None);
    }

    #[test]
    fn accepts_scalar_transformer_layers_and_attention_dim() {
        // Some diffusers configs write a single int instead of an array.
        let mut json = sdxl_config_json();
        json["transformer_layers_per_block"] = serde_json::json!(1);
        json["attention_head_dim"] = serde_json::json!(8);
        let cfg = parse_unet_config(&json).expect("scalar form parses");
        // Block 0 is DownBlock2D (no cross-attn); the scalar applies to the
        // CrossAttn blocks and to every attention_head_dim.
        assert_eq!(cfg.config.blocks[0].use_cross_attn, None);
        assert_eq!(cfg.config.blocks[1].use_cross_attn, Some(1));
        assert_eq!(cfg.config.blocks[2].use_cross_attn, Some(1));
        assert_eq!(cfg.config.blocks[2].attention_head_dim, 8);
    }

    #[test]
    fn rejects_unsupported_block_type() {
        let mut json = sdxl_config_json();
        json["down_block_types"] = serde_json::json!([
            "AttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ]);
        let err = parse_unet_config(&json).expect_err("AttnDownBlock2D is not supported");
        assert!(format!("{err:#}").contains("AttnDownBlock2D"));
    }

    #[test]
    fn rejects_mismatched_block_arrays() {
        let mut json = sdxl_config_json();
        json["block_out_channels"] = serde_json::json!([320, 640]);
        let err = parse_unet_config(&json).expect_err("length mismatch must fail");
        assert!(format!("{err:#}").contains("same length"), "{err}");
    }

    #[test]
    fn rejects_missing_required_fields() {
        for key in ["block_out_channels", "cross_attention_dim", "norm_eps"] {
            let mut json = sdxl_config_json();
            json.as_object_mut().expect("object").remove(key);
            assert!(
                parse_unet_config(&json).is_err(),
                "missing '{key}' must fail"
            );
        }
    }
}
