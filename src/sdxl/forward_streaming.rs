use crate::ssd::layer_stream::LayerStream;
use anyhow::Context;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, Linear};

pub fn sdxl_forward_streaming(
    stream: &mut LayerStream,
    input: &Tensor,
    timestep: f64,
    encoder_hidden_states: &Tensor,
) -> anyhow::Result<Tensor> {
    let device = input.device().clone();
    let dtype = input.dtype();
    let bsz = input.dim(0)?;

    stream.begin();

    // time_embedding: sin(320) batched → MLP → [bsize, 1280]
    let temb = {
        let sin_emb = timestep_embedding(timestep, 320, bsz, &device, dtype)?;
        let te_weights = stream.next_layer()?;
        let te_l1 = Linear::new(
            w(&te_weights, "linear_1.weight")?.clone(),
            Some(w(&te_weights, "linear_1.bias")?.clone()),
        );
        let te_l2 = Linear::new(
            w(&te_weights, "linear_2.weight")?.clone(),
            Some(w(&te_weights, "linear_2.bias")?.clone()),
        );
        let mut temb = te_l1.forward(&sin_emb)?;
        temb = candle_nn::ops::silu(&temb)?;
        te_l2.forward(&temb)?
    };
    let mut sample = conv_in(&stream.next_layer()?, input)?;
    let mut down_samples: Vec<Tensor> = Vec::new();

    for bi in 0..3 {
        let out_c = [320, 640, 1280][bi];

        for _ri in 0..2 {
            sample = resnet_block(&stream.next_layer()?, &sample, &temb)?;
        }

        if bi >= 1 {
            for _ai in 0..2 {
                let sp_weights = stream.next_layer()?;
                let (_, _, sh, sw) = sample.dims4()?;
                sample = spatial_norm_proj(&sp_weights, &sample)?;

                let depth: usize = if bi == 1 { 2 } else { 10 };
                for _ in 0..depth {
                    sample = transformer_block(
                        &stream.next_layer()?,
                        &sample,
                        encoder_hidden_states,
                        out_c,
                        64,
                        bsz,
                    )?;
                }

                sample = spatial_proj_out(&sp_weights, &sample, sh, sw)?;
            }
        }

        down_samples.push(sample.clone());

        if bi < 2 {
            sample = downsample_block(&stream.next_layer()?, &sample)?;
        }
    }

    sample = resnet_block(&stream.next_layer()?, &sample, &temb)?;
    let mid_sp = stream.next_layer()?;
    let (_, _, mh, mw) = sample.dims4()?;
    sample = spatial_norm_proj(&mid_sp, &sample)?;
    for _ in 0..10 {
        sample = transformer_block(
            &stream.next_layer()?,
            &sample,
            encoder_hidden_states,
            1280,
            64,
            bsz,
        )?;
    }
    sample = spatial_proj_out(&mid_sp, &sample, mh, mw)?;

    for bi in (0..3).rev() {
        let out_c = [320, 640, 1280][bi];

        for _ri in 0..3 {
            let skip = down_samples.pop().expect("skip connection");
            let cat = Tensor::cat(&[&sample, &skip], 1)?;
            sample = resnet_block(&stream.next_layer()?, &cat, &temb)?;
        }

        if bi >= 1 {
            for _ai in 0..3 {
                let sp_weights = stream.next_layer()?;
                let (_, _, sh, sw) = sample.dims4()?;
                sample = spatial_norm_proj(&sp_weights, &sample)?;

                let depth: usize = if bi == 2 { 10 } else { 2 };
                for _ in 0..depth {
                    sample = transformer_block(
                        &stream.next_layer()?,
                        &sample,
                        encoder_hidden_states,
                        out_c,
                        64,
                        bsz,
                    )?;
                }

                sample = spatial_proj_out(&sp_weights, &sample, sh, sw)?;
            }
        }

        sample = upsample_block(&stream.next_layer()?, &sample)?;
    }

    // conv_norm_out: GroupNorm + silu
    sample = {
        let weights = stream.next_layer()?;
        let w = find_weight(&weights, "weight")?.clone();
        let b = find_weight(&weights, "bias")?.clone();
        let c = sample.dim(1)?;
        let gn = GroupNorm::new(w, b, c, 32, 1e-5)?;
        let s = gn.forward(&sample)?;
        candle_nn::ops::silu(&s)?
    };

    // conv_out: Conv2d 1280→4
    {
        let weights = stream.next_layer()?;
        let conv_w = find_weight(&weights, "weight")?.clone();
        let conv_b = find_weight(&weights, "bias")?.clone();
        Ok(conv_layer(&conv_w, Some(&conv_b), 1, 1).forward(&sample)?)
    }
}

fn w<'a>(weights: &'a [(String, Tensor)], suffix: &str) -> anyhow::Result<&'a Tensor> {
    let pattern = format!(".{suffix}");
    weights
        .iter()
        .find(|(n, _)| n.ends_with(&pattern))
        .map(|(_, t)| t)
        .with_context(|| format!("weight '{suffix}' not found"))
}

fn conv_layer(w_t: &Tensor, b_t: Option<&Tensor>, stride: usize, pad: usize) -> Conv2d {
    Conv2d::new(
        w_t.clone(),
        b_t.cloned(),
        Conv2dConfig {
            stride,
            padding: pad,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    )
}

fn timestep_embedding(
    timestep: f64,
    dim: usize,
    bsz: usize,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<Tensor> {
    let half = dim / 2;
    let log_max = 9.210340371976184_f64;
    let exponent = Tensor::arange(0, half as i64, device)?
        .to_dtype(DType::F32)?
        .affine(-log_max / half as f64, 0.)?
        .exp()?;
    // Create batch-size repeated timestep: [bsize] all = timestep
    let t = Tensor::full(timestep as f32, bsz, device)?;
    let emb = t.unsqueeze(1)?.broadcast_mul(&exponent.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[emb.cos()?, emb.sin()?], 1)?;
    Ok(emb.to_dtype(dtype)?)
}

fn conv_in(weights: &[(String, Tensor)], x: &Tensor) -> anyhow::Result<Tensor> {
    let conv = conv_layer(w(weights, "weight")?, Some(w(weights, "bias")?), 1, 1);
    Ok(conv.forward(x)?)
}

fn find_weight<'a>(weights: &'a [(String, Tensor)], ends_with: &str) -> anyhow::Result<&'a Tensor> {
    weights
        .iter()
        .find(|(n, _)| n.ends_with(ends_with))
        .map(|(_, t)| t)
        .with_context(|| format!("weight ending with '{ends_with}' not found"))
}

fn resnet_block(weights: &[(String, Tensor)], x: &Tensor, temb: &Tensor) -> anyhow::Result<Tensor> {
    let n1w = w(weights, "norm1.weight")?.clone();
    let n1b = w(weights, "norm1.bias")?.clone();
    let c1w = w(weights, "conv1.weight")?.clone();
    let c1b = w(weights, "conv1.bias")?.clone();
    let tp = Linear::new(w(weights, "time_emb_proj.weight")?.clone(), Some(w(weights, "time_emb_proj.bias")?.clone()));
    let n2w = w(weights, "norm2.weight")?.clone();
    let n2b = w(weights, "norm2.bias")?.clone();
    let c2w = w(weights, "conv2.weight")?.clone();
    let c2b = w(weights, "conv2.bias")?.clone();

    let bsz = x.dim(0)?;
    let c = x.dim(1)?;
    let gn1 = GroupNorm::new(n1w, n1b, c, 32, 1e-5)?;
    let mut h = gn1.forward(x)?;
    h = candle_nn::ops::silu(&h)?;
    h = conv_layer(&c1w, Some(&c1b), 1, 1).forward(&h)?;

    let temb_silu = candle_nn::ops::silu(temb)?;
    let temb_proj = tp.forward(&temb_silu)?;
    let temb_proj = temb_proj.reshape((bsz, c, 1, 1))?;
    h = h.broadcast_add(&temb_proj)?;

    let gn2 = GroupNorm::new(n2w, n2b, c, 32, 1e-5)?;
    h = gn2.forward(&h)?;
    h = candle_nn::ops::silu(&h)?;
    h = conv_layer(&c2w, Some(&c2b), 1, 1).forward(&h)?;

    let shortcut = if let (Ok(csw), Ok(csb)) = (w(weights, "conv_shortcut.weight"), w(weights, "conv_shortcut.bias")) {
        conv_layer(csw, Some(csb), 1, 0).forward(x)?
    } else {
        x.clone()
    };
    Ok((h + shortcut)?)
}

fn spatial_norm_proj(weights: &[(String, Tensor)], x: &Tensor) -> anyhow::Result<Tensor> {
    let nw = w(weights, "norm.weight")?.clone();
    let nb = w(weights, "norm.bias")?.clone();
    let pw = w(weights, "proj_in.weight")?.clone();
    let pb = w(weights, "proj_in.bias")?.clone();

    let c = x.dim(1)?;
    let x = x.flatten(2, 3)?;
    let x = x.transpose(1, 2)?.contiguous()?;

    let gn = GroupNorm::new(nw, nb, c, 32, 1e-5)?;
    let x = gn.forward(&x)?;

    Ok(Linear::new(pw, Some(pb)).forward(&x)?)
}

fn spatial_proj_out(weights: &[(String, Tensor)], x: &Tensor, h: usize, width: usize) -> anyhow::Result<Tensor> {
    let pw = w(weights, "proj_out.weight")?.clone();
    let pb = w(weights, "proj_out.bias")?.clone();

    let x = Linear::new(pw, Some(pb)).forward(x)?;
    let x = x.transpose(1, 2)?.contiguous()?;

    Ok(x.reshape((x.dim(0)?, x.dim(1)?, h, width))?)
}

fn transformer_block(
    weights: &[(String, Tensor)],
    x: &Tensor,
    context: &Tensor,
    dim: usize,
    head_dim: usize,
    bsz: usize,
) -> anyhow::Result<Tensor> {
    let n_heads = dim / head_dim;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let residual = x.clone();
    let normed = layer_norm(x, weights, "norm1")?;
    let attn_out = attention(&normed, &normed, weights, "attn1", n_heads, head_dim, scale, bsz)?;
    let mut x = (attn_out + residual)?;

    let residual = x.clone();
    let normed = layer_norm(&x, weights, "norm2")?;
    let attn_out = attention(&normed, context, weights, "attn2", n_heads, head_dim, scale, bsz)?;
    x = (attn_out + residual)?;

    let residual = x.clone();
    let normed = layer_norm(&x, weights, "norm3")?;
    let ff1 = Linear::new(w(weights, "ff.net.0.proj.weight")?.clone(), Some(w(weights, "ff.net.0.proj.bias")?.clone()));
    let ff2 = Linear::new(w(weights, "ff.net.2.weight")?.clone(), Some(w(weights, "ff.net.2.bias")?.clone()));
    let ff_out = ff1.forward(&normed)?;
    let ff_out = ff_out.gelu()?;
    let ff_out = ff2.forward(&ff_out)?;
    x = (ff_out + residual)?;

    Ok(x)
}

fn layer_norm(x: &Tensor, weights: &[(String, Tensor)], prefix: &str) -> anyhow::Result<Tensor> {
    let w_t = w(weights, &format!("{prefix}.weight"))?;
    let b_t = w(weights, &format!("{prefix}.bias"))?;
    Ok(candle_nn::ops::layer_norm(x, w_t, b_t, 1e-5f32)?)
}

fn attention(
    query: &Tensor,
    key: &Tensor,
    weights: &[(String, Tensor)],
    prefix: &str,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
    bsz: usize,
) -> anyhow::Result<Tensor> {
    let (b, t, _c) = query.dims3()?;
    let _ = bsz;

    let q_lin = Linear::new(w(weights, &format!("{prefix}.to_q.weight"))?.clone(), Some(w(weights, &format!("{prefix}.to_q.bias"))?.clone()));
    let k_lin = Linear::new(w(weights, &format!("{prefix}.to_k.weight"))?.clone(), Some(w(weights, &format!("{prefix}.to_k.bias"))?.clone()));
    let v_lin = Linear::new(w(weights, &format!("{prefix}.to_v.weight"))?.clone(), Some(w(weights, &format!("{prefix}.to_v.bias"))?.clone()));
    let o_lin = Linear::new(w(weights, &format!("{prefix}.to_out.0.weight"))?.clone(), Some(w(weights, &format!("{prefix}.to_out.0.bias"))?.clone()));

    let q = q_lin.forward(query)?;
    let k = k_lin.forward(key)?;
    let v = v_lin.forward(key)?;

    let kv_seq_len = key.dims()[1];

    // Merge batch and heads: [B, seq, dim*heads*head_dim] → [B*heads, seq, head_dim]
    let q = q.reshape((b * n_heads, t, head_dim))?;
    let k = k.reshape((b * n_heads, kv_seq_len, head_dim))?;
    let v = v.reshape((b * n_heads, kv_seq_len, head_dim))?;

    let attn_w = q.matmul(&k.transpose(1, 2)?)?;
    let attn_w = (attn_w * scale)?;
    let attn_w = candle_nn::ops::softmax(&attn_w, 2)?;

    let attn_out = attn_w.matmul(&v)?;
    let attn_out = attn_out.reshape((b, n_heads, t, head_dim))?.transpose(1, 2)?.contiguous()?.reshape((b, t, n_heads * head_dim))?;

    Ok(o_lin.forward(&attn_out)?)
}

fn downsample_block(weights: &[(String, Tensor)], x: &Tensor) -> anyhow::Result<Tensor> {
    let conv = conv_layer(w(weights, "conv.weight")?, w(weights, "conv.bias").ok(), 2, 1);
    Ok(conv.forward(x)?)
}

fn upsample_block(weights: &[(String, Tensor)], x: &Tensor) -> anyhow::Result<Tensor> {
    let (_, _, h, w_dim) = x.dims4()?;
    let up = x.upsample_nearest2d(h * 2, w_dim * 2)?;
    let conv = conv_layer(w(weights, "conv.weight")?, w(weights, "conv.bias").ok(), 1, 1);
    Ok(conv.forward(&up)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssd::layer_stream::LayerStream;
    use crate::ssd::offset_index::OffsetIndex;
    use crate::ssd::weight_descriptor::LayerStructure;
    use crate::sdxl::unet_structure::SdxlUnetStructure;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::stable_diffusion::{self, unet_2d};

    fn unet_path() -> std::path::PathBuf {
        let home = dirs::home_dir().expect("HOME");
        home.join(".cache/huggingface/hub/models--John6666--the-araminta-experiment-fv5-sdxl/snapshots/3d76a0d0b8845b7fdefcea1b7cc4f592a4003f7d/unet/diffusion_pytorch_model.safetensors")
    }

    #[test]
    fn compare_candle_vs_streaming() {
        let path = unet_path();
        if !path.exists() {
            eprintln!("UNet not cached, skipping test");
            return;
        }

        let device = Device::Cpu;
        let dtype = DType::F32;

        let latent = Tensor::randn(0f32, 1f32, (1, 4, 64, 64), &device).unwrap();
        let latent = Tensor::cat(&[&latent, &latent], 0).unwrap(); // expand to batch=2 for CFG
        let timestep: f64 = 500.0;
        let text_emb = Tensor::randn(0f32, 1f32, (2, 77, 2048), &device).unwrap(); // batch 2 for CFG

        // Candle forward
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl(None, Some(512), Some(512));
        let unet = sd_config.build_unet(&path, &device, 4, false, dtype).unwrap();
        let expected = unet.forward(&latent, timestep, &text_emb).unwrap();

        // Streaming forward
        let index = OffsetIndex::from_file(&path).unwrap();
        let structure = SdxlUnetStructure::from_offset_index(&index).unwrap();
        let layer_specs = structure.layer_specs();
        let mut stream = LayerStream::new(&path, layer_specs, &device, dtype).unwrap();

        let actual = sdxl_forward_streaming(&mut stream, &latent, timestep, &text_emb).unwrap();

        let diff = (expected - actual).unwrap();
        let max_diff = diff.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();

        eprintln!("max diff between candle and streaming: {max_diff}");
        assert!(max_diff < 1e-3, "output mismatch: max diff = {max_diff}");
    }
}
