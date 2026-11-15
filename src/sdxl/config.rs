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
        !block_out_channels.is_empty(),
        "block_out_channels must not be empty"
    );
    anyhow::ensure!(
        block_out_channels.len() == down_block_types.len(),
        "block_out_channels ({} entries) and down_block_types ({} entries) must have the same length",
        block_out_channels.len(),
        down_block_types.len(),
    );
    validate_activation(json)?;
    validate_mirror_layout(json, &down_block_types)?;
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

fn optional_str<'v>(json: &'v serde_json::Value, key: &str) -> Result<Option<&'v str>> {
    match json.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(Some)
            .with_context(|| format!("field '{key}' must be a string, got {v}")),
    }
}

/// Candle hardcodes SiLU in the timestep embedding, ResNet blocks, and the
/// output block. Accepting another activation would silently compute a
/// different architecture with matching weight shapes, so reject it.
fn validate_activation(json: &serde_json::Value) -> Result<()> {
    if let Some(act_fn) = optional_str(json, "act_fn")? {
        anyhow::ensure!(
            act_fn == "silu",
            "unsupported act_fn '{act_fn}': candle implements SiLU only"
        );
    }
    Ok(())
}

/// Candle derives the up path as the mirror of the down path and always
/// builds a cross-attention mid block. A diffusers config that declares a
/// different layout would load the wrong architecture (or fail later with a
/// confusing missing-weight error), so verify the declared layout matches
/// what candle can actually build.
fn validate_mirror_layout(json: &serde_json::Value, down_types: &[&str]) -> Result<()> {
    if let Some(up_types) = optional_str_array(json, "up_block_types")? {
        let expected: Vec<&str> = down_types
            .iter()
            .rev()
            .map(|d| match *d {
                "DownBlock2D" => "UpBlock2D",
                "CrossAttnDownBlock2D" => "CrossAttnUpBlock2D",
                other => other, // rejected earlier with a dedicated error
            })
            .collect();
        anyhow::ensure!(
            up_types == expected,
            "up_block_types {up_types:?} do not mirror down_block_types {down_types:?} the way candle builds the up path"
        );
    }
    // An absent field means the diffusers default (UNetMidBlock2DCrossAttn,
    // which is what candle builds). An explicit null is different: diffusers
    // then creates NO mid block at all, which candle cannot represent — so
    // reject it instead of silently building a cross-attn mid.
    match json.get("mid_block_type") {
        None => {}
        Some(serde_json::Value::Null) => {
            anyhow::bail!(
                "mid_block_type: null means no mid block, which candle cannot build (it always constructs UNetMidBlock2DCrossAttn)"
            )
        }
        Some(v) => {
            let mid = v
                .as_str()
                .with_context(|| format!("field 'mid_block_type' must be a string, got {v}"))?;
            anyhow::ensure!(
                mid == "UNetMidBlock2DCrossAttn",
                "unsupported mid_block_type '{mid}': candle builds UNetMidBlock2DCrossAttn only"
            );
        }
    }
    if json
        .get("reverse_transformer_layers_per_block")
        .is_some_and(|v| !v.is_null())
    {
        anyhow::bail!(
            "reverse_transformer_layers_per_block is not supported: candle reuses the down-path transformer counts for the up path"
        );
    }
    Ok(())
}

fn optional_str_array<'v>(json: &'v serde_json::Value, key: &str) -> Result<Option<Vec<&'v str>>> {
    match json.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(_) => get_str_array(json, key).map(Some),
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

    #[test]
    fn rejects_empty_block_arrays() {
        // Empty arrays pass the length check but panic inside candle's
        // UNet2DConditionModel::new; the parser must reject them first.
        let mut json = sdxl_config_json();
        json["block_out_channels"] = serde_json::json!([]);
        json["down_block_types"] = serde_json::json!([]);
        let err = parse_unet_config(&json).expect_err("empty blocks must fail");
        assert!(format!("{err:#}").contains("must not be empty"), "{err}");
    }

    #[test]
    fn rejects_non_silu_activation() {
        // act_fn changes the computation, not just metadata; candle only
        // implements SiLU, so anything else must fail loudly.
        let mut json = sdxl_config_json();
        json["act_fn"] = serde_json::json!("gelu");
        let err = parse_unet_config(&json).expect_err("gelu must be rejected");
        assert!(format!("{err:#}").contains("act_fn"), "{err}");
    }

    #[test]
    fn accepts_silu_or_missing_activation() {
        let mut json = sdxl_config_json();
        json["act_fn"] = serde_json::json!("silu");
        assert!(parse_unet_config(&json).is_ok());
        json.as_object_mut().expect("object").remove("act_fn");
        assert!(parse_unet_config(&json).is_ok());
    }

    #[test]
    fn validates_up_and_mid_layout_against_candles_mirror_scheme() {
        // Correct mirror per the real SDXL-base config: up[i] is the Up
        // counterpart of down[n-1-i].
        let mut json = sdxl_config_json();
        json["up_block_types"] =
            serde_json::json!(["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]);
        json["mid_block_type"] = serde_json::json!("UNetMidBlock2DCrossAttn");
        json["reverse_transformer_layers_per_block"] = serde_json::Value::Null;
        assert!(
            parse_unet_config(&json).is_ok(),
            "mirrored layout must parse"
        );

        // Non-mirrored up path.
        let mut bad = json.clone();
        bad["up_block_types"] =
            serde_json::json!(["CrossAttnUpBlock2D", "UpBlock2D", "CrossAttnUpBlock2D"]);
        let err = parse_unet_config(&bad).expect_err("non-mirrored up types must fail");
        assert!(format!("{err:#}").contains("do not mirror"), "{err}");

        // Mid block without cross attention.
        let mut bad = json.clone();
        bad["mid_block_type"] = serde_json::json!("UNetMidBlock2DSimpleCrossAttn");
        let err = parse_unet_config(&bad).expect_err("non-cross-attn mid must fail");
        assert!(format!("{err:#}").contains("mid_block_type"), "{err}");

        // Explicit null means NO mid block in diffusers; candle always builds
        // one, so it must be rejected rather than treated as the default.
        let mut bad = json.clone();
        bad["mid_block_type"] = serde_json::Value::Null;
        let err = parse_unet_config(&bad).expect_err("null mid_block_type must fail");
        assert!(format!("{err:#}").contains("null"), "{err}");

        // Absent field keeps the diffusers default of a cross-attn mid (base
        // fixture has no mid_block_type at all and parses fine).
        let mut ok = json.clone();
        ok.as_object_mut().expect("object").remove("mid_block_type");
        assert!(
            parse_unet_config(&ok).is_ok(),
            "absent mid_block_type must parse"
        );

        // Reversed transformer counts are unsupported.
        let mut bad = json.clone();
        bad["reverse_transformer_layers_per_block"] = serde_json::json!([10, 2, 1]);
        let err = parse_unet_config(&bad).expect_err("reversed transformer counts must fail");
        assert!(
            format!("{err:#}").contains("reverse_transformer_layers_per_block"),
            "{err}"
        );
    }
}
