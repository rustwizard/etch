use crate::cli::{Args, Model, Quantization};
use crate::image;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, t5};
use tokenizers::Tokenizer;
use tracing::info;

use super::gguf::load_gguf_with_spinner;
use super::model::FluxModel;

pub fn run_flux(args: &Args, device: &Device, dtype: DType) -> Result<()> {
    let height = args.height.unwrap_or(768);
    let width = args.width.unwrap_or(1360);
    let api = hf_hub::api::sync::Api::new()?;

    // GGUF weights live in CPU RAM; run the entire DiT on CPU to avoid device
    // mismatches (layer norms, biases all come from the CPU VarBuilder) and to
    // prevent Metal OOM from T5 (9.5 GB) and GGUF (7–12 GB) overlapping.
    // T5 and CLIP still encode on Metal for speed; only their small output
    // embeddings are moved to CPU before FLUX inference.
    let is_gguf = args.gguf.is_some() || matches!(args.model, Model::SchnellGguf | Model::DevGguf);
    let flux_device = if is_gguf { Device::Cpu } else { device.clone() };
    // GGUF weights dequantize to F32; mixing BF16 tensors with F32 weights fails in matmul.
    let dtype = if is_gguf { DType::F32 } else { dtype };

    let bf_repo = {
        let name = match args.model {
            Model::Dev | Model::DevGguf => "black-forest-labs/FLUX.1-dev",
            _ => "black-forest-labs/FLUX.1-schnell",
        };
        api.repo(hf_hub::Repo::model(name.to_string()))
    };

    // T5 and CLIP are independent — load and encode in parallel.
    let (t5_emb, clip_emb) = std::thread::scope(|s| {
        let t5_handle = s.spawn(|| -> Result<Tensor> {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model("mcmonkey/google_t5-v1_1-xxl_encoderonly".to_string());
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[repo.get("model.safetensors")?],
                    dtype,
                    device,
                )?
            };
            let config: t5::Config =
                serde_json::from_str(&std::fs::read_to_string(repo.get("config.json")?)?)?;
            let mut model = t5::T5EncoderModel::load(vb, &config)?;
            let tokenizer_file = api
                .model("lmz/mt5-tokenizers".to_string())
                .get("t5-v1_1-xxl.tokenizer.json")?;
            let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
            let mut tokens = tokenizer
                .encode(args.prompt.as_str(), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            tokens.resize(256, 0);
            let emb = model.forward(&Tensor::new(&tokens[..], device)?.unsqueeze(0)?)?;
            Ok(emb.to_device(&flux_device)?)
        });

        let clip_handle = s.spawn(|| -> Result<Tensor> {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.repo(hf_hub::Repo::model(
                "openai/clip-vit-large-patch14".to_string(),
            ));
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[repo.get("model.safetensors")?],
                    dtype,
                    device,
                )?
            };
            let config = clip::text_model::ClipTextConfig {
                vocab_size: 49408,
                projection_dim: 768,
                activation: clip::text_model::Activation::QuickGelu,
                intermediate_size: 3072,
                embed_dim: 768,
                max_position_embeddings: 77,
                pad_with: None,
                num_hidden_layers: 12,
                num_attention_heads: 12,
            };
            let model = clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config)?;
            let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(E::msg)?;
            let tokens = tokenizer
                .encode(args.prompt.as_str(), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let emb = model.forward(&Tensor::new(&tokens[..], device)?.unsqueeze(0)?)?;
            Ok(emb.to_device(&flux_device)?)
        });

        let t5_emb = t5_handle
            .join()
            .map_err(|e| anyhow::anyhow!("T5 thread panicked: {e:?}"))??;
        let clip_emb = clip_handle
            .join()
            .map_err(|e| anyhow::anyhow!("CLIP thread panicked: {e:?}"))??;
        Ok::<_, anyhow::Error>((t5_emb, clip_emb))
    })?;
    info!("T5: {:?}", t5_emb.shape());
    info!("CLIP: {:?}", clip_emb.shape());

    // FLUX DiT
    let img = {
        let cfg = match args.model {
            Model::Dev | Model::DevGguf => flux::model::Config::dev(),
            _ => flux::model::Config::schnell(),
        };
        let img = flux::sampling::get_noise(1, height, width, &flux_device)?.to_dtype(dtype)?;
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
        let n_steps = args.n_steps.unwrap_or(match args.model {
            Model::Dev | Model::DevGguf => 50,
            _ => 4,
        });
        let timesteps = match args.model {
            Model::Dev | Model::DevGguf => {
                flux::sampling::get_schedule(n_steps, Some((state.img.dim(1)?, 0.5, 1.15)))
            }
            _ => flux::sampling::get_schedule(n_steps, None),
        };
        let model = if let Some(gguf_path) = &args.gguf {
            let vb = load_gguf_with_spinner(gguf_path, gguf_path, device)?;
            FluxModel::Quantized(Box::new(flux::quantized_model::Flux::new(&cfg, vb)?))
        } else if matches!(args.model, Model::SchnellGguf | Model::DevGguf) {
            let q = match args.quantization {
                Quantization::Q8 => "Q8_0",
                Quantization::Q4 => "Q4_K_S",
            };
            let (gguf_repo, gguf_file) = match args.model {
                Model::DevGguf => ("city96/FLUX.1-dev-gguf", format!("flux1-dev-{q}.gguf")),
                _ => (
                    "city96/FLUX.1-schnell-gguf",
                    format!("flux1-schnell-{q}.gguf"),
                ),
            };
            let gguf_file = gguf_file.as_str();
            let path = api
                .repo(hf_hub::Repo::model(gguf_repo.to_string()))
                .get(gguf_file)?;
            let vb = load_gguf_with_spinner(&path, gguf_file, device)?;
            FluxModel::Quantized(Box::new(flux::quantized_model::Flux::new(&cfg, vb)?))
        } else {
            let model_file = match args.model {
                Model::Dev => bf_repo.get("flux1-dev.safetensors")?,
                _ => bf_repo.get("flux1-schnell.safetensors")?,
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
            FluxModel::Full(Box::new(flux::model::Flux::new(&cfg, vb)?))
        };
        let denoised = {
            let n_steps = timesteps.len().saturating_sub(1);
            let b_sz = state.img.dim(0)?;
            let guidance = Tensor::full(4f32, b_sz, &flux_device)?;
            let mut img = state.img.clone();
            let loop_start = std::time::Instant::now();
            for (i, window) in timesteps.windows(2).enumerate() {
                let (t_curr, t_prev) = (window[0], window[1]);
                let t_vec = Tensor::full(t_curr as f32, b_sz, &flux_device)?;
                let step_start = std::time::Instant::now();
                let pred = flux::WithForward::forward(
                    &model,
                    &img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance),
                )?;
                img = (img + (pred * (t_prev - t_curr))?)?;
                let step_secs = step_start.elapsed().as_secs_f32();
                let total_secs = loop_start.elapsed().as_secs_f32();
                let done = i + 1;
                let eta = step_secs * (n_steps - done) as f32;
                let bar_len = 20usize;
                let filled = bar_len * done / n_steps;
                let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
                print!(
                    "\rstep {done}/{n_steps} [{bar}] {step_secs:.1}s/step  {total_secs:.0}s elapsed  ETA {eta:.0}s"
                );
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
            }
            println!();
            img
        };
        let unpacked = flux::sampling::unpack(&denoised, height, width)?;
        unpacked.to_device(device)?
    };

    // VAE decode — always F32 for stable decode regardless of model dtype.
    // Optionally on CPU to keep Metal pool from growing with intermediate activations.
    let vae_device = if args.vae_cpu {
        Device::Cpu
    } else {
        device.clone()
    };
    let img = img.to_device(&vae_device)?;
    let img = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[bf_repo.get("ae.safetensors")?],
                DType::F32,
                &vae_device,
            )?
        };
        let cfg = match args.model {
            Model::Dev | Model::DevGguf => flux::autoencoder::Config::dev(),
            _ => flux::autoencoder::Config::schnell(),
        };
        flux::autoencoder::AutoEncoder::new(&cfg, vb)?.decode(&img)?
    };

    let img = img.to_device(&Device::Cpu)?;
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let out = args.output.as_deref().expect("output set in main");
    image::save_image(&img.i(0)?, out)?;
    info!("Saved to {out}");
    Ok(())
}
