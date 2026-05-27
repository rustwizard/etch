#![deny(clippy::unwrap_used)]

mod cli;
mod device;
mod image;
mod lora;
mod logger;
mod schedulers;

use cli::{Args, Model, Quantization, SamplerType};
use schedulers::{Dpm2mKarrasScheduler, KarrasEulerAScheduler};

use candle_transformers::models::{clip, flux, stable_diffusion, t5};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use stable_diffusion::{
    euler_ancestral_discrete::{
        EulerAncestralDiscreteScheduler, EulerAncestralDiscreteSchedulerConfig,
    },
    schedulers::{PredictionType, Scheduler},
    unet_2d,
};

use anyhow::{Error as E, Result};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();

    let device = device::pick_device(args.cpu);
    info!("Device: {:?}", device);

    let seeds = if let Some(ref range_str) = args.seed_range {
        cli::parse_seed_range(range_str)?
    } else {
        vec![args.seed.unwrap_or_else(rand::random)]
    };
    info!("Generating {} image(s)", seeds.len());

    let dtype: DType = args.dtype.map(DType::from).unwrap_or_else(|| {
        if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            DType::BF16
        }
    });
    info!("Dtype: {:?}", dtype);

    for seed in seeds {
        info!("--- Seed: {seed} ---");
        if !matches!(device, Device::Cpu) {
            if let Err(e) = device.set_seed(seed) {
                tracing::warn!("Failed to set seed {seed}: {e}");
            }
        }

        let output = cli::output_for_seed(&args.output, seed);
        if let Some(parent) = std::path::Path::new(&output).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let iter_args = Args {
            seed: Some(seed),
            output: Some(output),
            ..args.clone()
        };

        let t0 = std::time::Instant::now();
        let result = match iter_args.model {
            Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
                run_flux(&iter_args, &device, dtype)
            }
            Model::Araminta => run_sdxl(&iter_args, &device, dtype),
        };
        if let Err(e) = result {
            tracing::error!("Seed {seed} failed: {e}");
            continue;
        }
        info!("Total time: {:.1}s", t0.elapsed().as_secs_f32());

        let out_path = iter_args.output.as_deref().expect("output set above");
        logger::write_log_entry(out_path, &iter_args, seed)?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// FLUX pipeline
// ─────────────────────────────────────────────────────────────────────────────

enum FluxModel {
    Full(Box<flux::model::Flux>),
    Quantized(Box<flux::quantized_model::Flux>),
}

impl flux::WithForward for FluxModel {
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        match self {
            FluxModel::Full(m) => m.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance),
            FluxModel::Quantized(m) => {
                m.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance)
            }
        }
    }
}

fn load_gguf_with_spinner(
    path: impl AsRef<std::path::Path>,
    label: &str,
    _device: &Device,
) -> Result<QVarBuilder> {
    use std::io::Write as _;
    let spinner = ['|', '/', '-', '\\'];
    print!("Loading GGUF: {label}  ");
    let _ = std::io::stdout().flush();

    let path = path.as_ref().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel::<Result<QVarBuilder>>();
    std::thread::spawn(move || {
        let _ = tx.send(QVarBuilder::from_gguf(path, &Device::Cpu).map_err(Into::into));
    });

    let mut i = 0usize;
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(120)) {
            Ok(result) => {
                print!("\rLoading GGUF: {label}  done\n");
                let _ = std::io::stdout().flush();
                return result;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                print!("\rLoading GGUF: {label}  {}", spinner[i % 4]);
                let _ = std::io::stdout().flush();
                i += 1;
            }
            Err(e) => return Err(anyhow::anyhow!("GGUF loader thread error: {e}")),
        }
    }
}

fn run_flux(args: &Args, device: &Device, dtype: DType) -> Result<()> {
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

// ─────────────────────────────────────────────────────────────────────────────
// SDXL pipeline
// ─────────────────────────────────────────────────────────────────────────────

fn run_sdxl(args: &Args, device: &Device, dtype: DType) -> Result<()> {
    let height = args.height.unwrap_or(768);
    let width = args.width.unwrap_or(1024);
    let n_steps = args.n_steps.unwrap_or(20);
    let guidance_scale = args.guidance_scale;
    let use_guide_scale = guidance_scale > 1.0;

    let api = hf_hub::api::sync::Api::new()?;

    // Resolve a model-relative path: local dir or HuggingFace
    let model_file = {
        let local = args.local_model.clone();
        let repo = api.repo(hf_hub::Repo::model(
            "John6666/the-araminta-experiment-fv5-sdxl".to_string(),
        ));
        move |rel: &str| -> Result<std::path::PathBuf> {
            match &local {
                Some(dir) => Ok(std::path::PathBuf::from(dir).join(rel)),
                None => Ok(repo.get(rel)?),
            }
        }
    };

    let sd_config = stable_diffusion::StableDiffusionConfig::sdxl(None, Some(height), Some(width));
    let mut scheduler: Box<dyn Scheduler> = match args.scheduler {
        SamplerType::EulerA => {
            info!("Scheduler: Euler Ancestral");
            Box::new(EulerAncestralDiscreteScheduler::new(
                n_steps,
                EulerAncestralDiscreteSchedulerConfig {
                    prediction_type: PredictionType::Epsilon,
                    ..Default::default()
                },
            )?)
        }
        SamplerType::EulerAKarras => {
            info!("Scheduler: Euler Ancestral + Karras");
            Box::new(KarrasEulerAScheduler::new(n_steps)?)
        }
        SamplerType::Dpm2mKarras => {
            info!("Scheduler: DPM++ 2M Karras");
            Box::new(Dpm2mKarrasScheduler::new(n_steps)?)
        }
    };

    // Tokenizer 1: from local dir or CLIP HF repo
    let tok1 = {
        let path = match &args.local_model {
            Some(dir) => std::path::PathBuf::from(dir).join("tokenizer/tokenizer.json"),
            None => api
                .model("openai/clip-vit-large-patch14".to_string())
                .get("tokenizer.json")?,
        };
        Tokenizer::from_file(path).map_err(E::msg)?
    };

    // Tokenizer 2: from local dir or OpenCLIP HF repo
    let tok2 = {
        let path = match &args.local_model {
            Some(dir) => std::path::PathBuf::from(dir).join("tokenizer_2/tokenizer.json"),
            None => api
                .model("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string())
                .get("tokenizer.json")?,
        };
        Tokenizer::from_file(path).map_err(E::msg)?
    };

    // Load LoRA weights once; shared across UNet and both text encoders.
    let lora_map: Option<std::collections::HashMap<String, Tensor>> = if let Some(p) = &args.lora {
        info!("Loading LoRA: {p} (scale {})", args.lora_scale);
        Some(candle_core::safetensors::load(p, &Device::Cpu)?)
    } else {
        None
    };

    // Build embeddings from both encoders and cat along hidden dim
    let text_embeddings = {
        let ctx = ClipEmbedCtx {
            prompt: &args.prompt,
            uncond_prompt: &args.uncond_prompt,
            clip_skip: args.clip_skip,
            device,
            dtype,
            use_guide_scale,
        };
        let emb1 = sdxl_clip_emb(
            &ctx,
            &tok1,
            model_file("text_encoder/model.safetensors")?,
            &sd_config.clip,
            lora_map.as_ref().map(|m| (m, "lora_te1_", args.lora_scale)),
        )?;
        let clip2_config = sd_config
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SDXL config missing clip2"))?;
        let emb2 = sdxl_clip_emb(
            &ctx,
            &tok2,
            model_file("text_encoder_2/model.safetensors")?,
            clip2_config,
            lora_map.as_ref().map(|m| (m, "lora_te2_", args.lora_scale)),
        )?;
        // [batch, 77, 768] ++ [batch, 77, 1280] → [batch, 77, 2048]
        Tensor::cat(&[emb1, emb2], D::Minus1)?
    };
    info!("Text embeddings: {:?}", text_embeddings.shape());

    let unet = {
        let unet_weights = model_file("unet/diffusion_pytorch_model.safetensors")?;
        if let Some(lora) = &lora_map {
            info!("Applying UNet LoRA (scale {})", args.lora_scale);
            let mut tensors = candle_core::safetensors::load(&unet_weights, &Device::Cpu)?;
            tensors = lora::apply_lora(tensors, lora, args.lora_scale)?;
            let vb = VarBuilder::from_tensors(tensors, dtype, device);
            let bc = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
                out_channels,
                use_cross_attn,
                attention_head_dim,
            };
            let unet_cfg = unet_2d::UNet2DConditionModelConfig {
                blocks: vec![
                    bc(320, None, 5),
                    bc(640, Some(2), 10),
                    bc(1280, Some(10), 20),
                ],
                center_input_sample: false,
                cross_attention_dim: 2048,
                downsample_padding: 1,
                flip_sin_to_cos: true,
                freq_shift: 0.,
                layers_per_block: 2,
                mid_block_scale_factor: 1.,
                norm_eps: 1e-5,
                norm_num_groups: 32,
                sliced_attention_size: None,
                use_linear_projection: true,
            };
            unet_2d::UNet2DConditionModel::new(vb, 4, 4, false, unet_cfg)?
        } else {
            sd_config.build_unet(unet_weights, device, 4, false, dtype)?
        }
    };

    let vae_scale = 0.18215f64;
    let mut latents =
        (Tensor::randn(0f32, 1f32, (1usize, 4usize, height / 8, width / 8), device)?
            * scheduler.init_noise_sigma())?
        .to_dtype(dtype)?;

    let timesteps = scheduler.timesteps().to_vec();
    let loop_start = std::time::Instant::now();
    for (i, &timestep) in timesteps.iter().enumerate() {
        let t_step = std::time::Instant::now();
        let latent_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        let latent_input = scheduler.scale_model_input(latent_input, timestep)?;
        let noise_pred = unet.forward(&latent_input, timestep as f64, &text_embeddings)?;
        let noise_pred = if use_guide_scale {
            let chunks = noise_pred.chunk(2, 0)?;
            let (uncond, cond) = (&chunks[0], &chunks[1]);
            (uncond + ((cond - uncond)? * guidance_scale)?)?
        } else {
            noise_pred
        };
        latents = scheduler.step(&noise_pred, timestep, &latents)?;
        let step_secs = t_step.elapsed().as_secs_f32();
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

    drop(unet);
    drop(text_embeddings);

    // Load VAE only after UNet inference — avoids 335 MB Metal residency during the loop.
    // Optionally on CPU to keep Metal pool from growing with intermediate activations.
    let vae_device = if args.vae_cpu {
        Device::Cpu
    } else {
        device.clone()
    };
    let vae = sd_config.build_vae(
        model_file("vae/diffusion_pytorch_model.safetensors")?,
        &vae_device,
        DType::F32,
    )?;
    let latents = latents.to_device(&vae_device)?;
    let img = vae.decode(&(latents.to_dtype(DType::F32)? / vae_scale)?)?;
    drop(vae);
    let img = img.to_device(&Device::Cpu)?;
    let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
    let img = (img * 255.)?.to_dtype(DType::U8)?;
    let out = args.output.as_deref().expect("output set in main");
    image::save_image(&img.i(0)?, out)?;
    info!("Saved to {out}");
    Ok(())
}

struct ClipEmbedCtx<'a> {
    prompt: &'a str,
    uncond_prompt: &'a str,
    clip_skip: usize,
    device: &'a Device,
    dtype: DType,
    use_guide_scale: bool,
}

fn sdxl_clip_emb(
    ctx: &ClipEmbedCtx,
    tokenizer: &Tokenizer,
    weights: std::path::PathBuf,
    clip_config: &stable_diffusion::clip::Config,
    lora: Option<(&std::collections::HashMap<String, Tensor>, &str, f64)>,
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
            info!("TE LoRA ({te_prefix}): applied to {applied} layers");
        }
        VarBuilder::from_tensors(patched, ctx.dtype, ctx.device)
    } else {
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


