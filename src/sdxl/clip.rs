use crate::lora;
use anyhow::{Error as E, Result};
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub(crate) struct ClipEmbedCtx<'a> {
    pub prompt: &'a str,
    pub uncond_prompt: &'a str,
    pub clip_skip: usize,
    pub device: &'a Device,
    pub dtype: DType,
    pub use_guide_scale: bool,
}

pub(crate) fn sdxl_clip_emb(
    ctx: &ClipEmbedCtx,
    tokenizer: &Tokenizer,
    weights: std::path::PathBuf,
    clip_config: &stable_diffusion::clip::Config,
    lora: Option<(&HashMap<String, Tensor>, &str, f64)>,
) -> Result<Tensor> {
    let vocab = tokenizer.get_vocab(true);
    let pad_id = match &clip_config.pad_with {
        Some(p) => *vocab
            .get(p.as_str())
            .ok_or_else(|| anyhow::anyhow!("pad token '{p}' not in vocab"))?,
        None => *vocab
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("'<|endoftext|>' not in vocab"))?,
    };
    let max_len = clip_config.max_position_embeddings;

    let tokenize = |text: &str| -> Result<Tensor> {
        let mut ids = tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        ids.resize(max_len, pad_id);
        Ok(Tensor::new(ids.as_slice(), ctx.device)?.unsqueeze(0)?)
    };

    let vb = if let Some((lora_map, te_prefix, scale)) = lora {
        let tensors = candle_core::safetensors::load(&weights, &Device::Cpu)?;
        let (patched, applied) = lora::apply_te_lora(tensors, lora_map, te_prefix, scale)?;
        if applied > 0 {
            tracing::info!("TE LoRA ({te_prefix}): applied to {applied} layers");
        }
        VarBuilder::from_tensors(patched, ctx.dtype, ctx.device)
    } else {
        // SAFETY: file is owned by the HF cache or local model dir and not modified during inference.
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], ctx.dtype, ctx.device)? }
    };
    let model = stable_diffusion::clip::ClipTextTransformer::new(vb, clip_config)?;

    let encode = |tokens: &Tensor| -> Result<Tensor> {
        if ctx.clip_skip <= 1 {
            Ok(model.forward(tokens)?)
        } else {
            let layer_idx = -(ctx.clip_skip as isize);
            let (_final, intermediate) =
                model.forward_until_encoder_layer(tokens, usize::MAX, layer_idx)?;
            Ok(intermediate)
        }
    };

    let cond = encode(&tokenize(ctx.prompt)?)?;
    let emb = if ctx.use_guide_scale {
        let uncond = encode(&tokenize(ctx.uncond_prompt)?)?;
        Tensor::cat(&[uncond, cond], 0)?
    } else {
        cond
    };
    Ok(emb.to_dtype(ctx.dtype)?)
}
