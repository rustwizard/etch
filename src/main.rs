#![deny(clippy::unwrap_used)]

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

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "A rusty robot walking on a beach")]
    prompt: String,

    /// Negative prompt (SDXL only).
    #[arg(long, default_value = "")]
    uncond_prompt: String,

    // (cpu flag defined below near metal)
    /// Height in pixels. Default: 768 (FLUX) or 1024 (SDXL).
    #[arg(long)]
    height: Option<usize>,

    /// Width in pixels. Default: 1360 (FLUX) or 1024 (SDXL).
    #[arg(long)]
    width: Option<usize>,

    /// Denoising steps. Default: 4 (schnell), 50 (dev), 20 (araminta).
    #[arg(long)]
    n_steps: Option<usize>,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long)]
    output: Option<String>,

    #[arg(long, value_enum, default_value = "schnell")]
    model: Model,

    /// Classifier-free guidance scale (SDXL only, default 7.5).
    #[arg(long, default_value_t = 7.5)]
    guidance_scale: f64,

    /// Force CPU instead of Metal.
    #[arg(long)]
    cpu: bool,

    /// Path to a LoRA .safetensors file (SDXL only).
    #[arg(long)]
    lora: Option<String>,

    /// LoRA strength, 0.0–1.0 (default 1.0).
    #[arg(long, default_value_t = 1.0)]
    lora_scale: f64,

    /// Path to a local model directory in diffusers format (SDXL only).
    /// Overrides HuggingFace download. Convert with:
    ///   diffusers StableDiffusionXLPipeline.from_single_file(...).save_pretrained(path)
    #[arg(long)]
    local_model: Option<String>,

    /// CLIP skip: number of layers to skip from the end of the text encoder (SDXL only).
    /// 1 = last layer (default), 2 = penultimate, 3–4 = earlier layers.
    #[arg(long, default_value_t = 1)]
    clip_skip: usize,

    /// Sampler (SDXL only).
    #[arg(long, value_enum, default_value = "euler-a")]
    scheduler: SamplerType,

    /// Path to a local FLUX GGUF file (e.g. flux1-schnell-Q8_0.gguf). Skips HF download.
    #[arg(long)]
    gguf: Option<String>,

    /// Quantization level for schnell-gguf / dev-gguf models (default: q8).
    #[arg(long, value_enum, default_value = "q8")]
    quantization: Quantization,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Default)]
enum Quantization {
    /// Q8_0 — ~12 GB, best quality
    #[default]
    Q8,
    /// Q4_K_S — ~7 GB, smaller but slightly lower quality
    Q4,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Default)]
enum SamplerType {
    /// Euler Ancestral — stochastic, fast, good all-rounder
    #[default]
    EulerA,
    /// Euler Ancestral + Karras sigma spacing
    EulerAKarras,
    /// DPM++ 2M Karras — smooth, great at 20–30 steps
    Dpm2mKarras,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// FLUX.1-schnell — 4 steps, fast, Apache 2.0 (~24 GB safetensors)
    Schnell,
    /// FLUX.1-dev — 50 steps, best quality, non-commercial (~24 GB safetensors)
    Dev,
    /// FLUX.1-schnell Q8_0 GGUF — 4 steps, ~12 GB
    SchnellGguf,
    /// FLUX.1-dev Q8_0 GGUF — 50 steps, ~12 GB, non-commercial
    DevGguf,
    /// John6666/the-araminta-experiment-fv5-sdxl — ~7 GB SDXL
    Araminta,
}

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

    let device = if args.cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or_else(|e| {
                tracing::warn!("Metal init failed: {e}. Falling back to CPU.");
                Device::Cpu
            })
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            Device::new_cuda(0).unwrap_or_else(|e| {
                tracing::warn!("CUDA init failed: {e}. Falling back to CPU.");
                Device::Cpu
            })
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            Device::Cpu
        }
    };
    info!("Device: {:?}", device);

    let seed = args.seed.unwrap_or_else(rand::random);
    info!("Seed: {seed}");
    if !matches!(device, Device::Cpu) {
        device.set_seed(seed)?;
    }

    let dtype = DType::F32;

    let output = args.output.clone().unwrap_or_else(|| {
        let n: u32 = rand::random();
        format!("out/out-{seed}-{n}.png")
    });
    if let Some(parent) = std::path::Path::new(&output).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let args = Args {
        output: Some(output),
        ..args
    };

    let t0 = std::time::Instant::now();
    match args.model {
        Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
            run_flux(&args, &device, dtype)?
        }
        Model::Araminta => run_sdxl(&args, &device, dtype)?,
    }
    info!("Total time: {:.1}s", t0.elapsed().as_secs_f32());

    let out_path = args.output.as_deref().expect("output set above");
    let log_path = std::path::Path::new(out_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("log.jsonl");
    let mut entry = serde_json::json!({
        "file": out_path,
        "seed": seed,
        "prompt": args.prompt,
        "model": format!("{:?}", args.model).to_lowercase(),
        "steps": args.n_steps,
    });
    if args.model == Model::Araminta {
        entry["scheduler"] = serde_json::json!(format!("{:?}", args.scheduler).to_lowercase());
        entry["guidance_scale"] = serde_json::json!(args.guidance_scale);
        if !args.uncond_prompt.is_empty() {
            entry["uncond_prompt"] = serde_json::json!(args.uncond_prompt);
        }
    }
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    use std::io::Write as _;
    writeln!(log, "{}", entry)?;

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

    // T5 text encoder
    let t5_emb = {
        let repo = api.model("mcmonkey/google_t5-v1_1-xxl_encoderonly".to_string());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], dtype, device)?
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
        emb.to_device(&flux_device)?
    };
    info!("T5: {:?}", t5_emb.shape());

    // CLIP pooled embeddings
    let clip_emb = {
        let repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], dtype, device)?
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
        emb.to_device(&flux_device)?
    };
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

    // VAE decode
    let img = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[bf_repo.get("ae.safetensors")?], dtype, device)?
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
    save_image(&img.i(0)?, out)?;
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

    let vae = sd_config.build_vae(
        model_file("vae/diffusion_pytorch_model.safetensors")?,
        device,
        DType::F32,
    )?;
    let unet = {
        let unet_weights = model_file("unet/diffusion_pytorch_model.safetensors")?;
        if let Some(lora) = &lora_map {
            info!("Applying UNet LoRA (scale {})", args.lora_scale);
            let mut tensors = candle_core::safetensors::load(&unet_weights, &Device::Cpu)?;
            tensors = apply_lora(tensors, lora, args.lora_scale)?;
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

    let img = vae.decode(&(latents.to_dtype(DType::F32)? / vae_scale)?)?;
    drop(vae);
    let img = img.to_device(&Device::Cpu)?;
    let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
    let img = (img * 255.)?.to_dtype(DType::U8)?;
    let out = args.output.as_deref().expect("output set in main");
    save_image(&img.i(0)?, out)?;
    info!("Saved to {out}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Karras utilities
// ─────────────────────────────────────────────────────────────────────────────

// ScaledLinear beta schedule used by both SDXL Karras schedulers.
// Returns (all_sigmas, sigma_max, sigma_min) for n_steps inference steps.
fn build_sdxl_sigmas(n_steps: usize) -> (Vec<f64>, f64, f64) {
    const BETA_START: f64 = 0.00085;
    const BETA_END: f64 = 0.012;
    const TRAIN_STEPS: usize = 1000;
    const STEPS_OFFSET: usize = 1;

    let mut cumprod = 1.0f64;
    let all_sigmas: Vec<f64> = (0..TRAIN_STEPS)
        .map(|i| {
            let t = i as f64 / (TRAIN_STEPS - 1) as f64;
            let b = BETA_START.sqrt() + t * (BETA_END.sqrt() - BETA_START.sqrt());
            cumprod *= 1.0 - b * b;
            ((1.0 - cumprod) / cumprod).sqrt()
        })
        .collect();

    let step_ratio = TRAIN_STEPS / n_steps;
    let sigma_max = all_sigmas[(n_steps - 1) * step_ratio + STEPS_OFFSET];
    let sigma_min = all_sigmas[step_ratio + STEPS_OFFSET];
    (all_sigmas, sigma_max, sigma_min)
}

// Map a Karras sigma back to the nearest discrete timestep in the original
// schedule. all_sigmas is monotone-increasing (sigma grows with timestep).
fn sigma_to_t(sigma: f64, all_sigmas: &[f64]) -> usize {
    let idx = all_sigmas.partition_point(|&s| s < sigma);
    if idx == 0 {
        return 0;
    }
    if idx >= all_sigmas.len() {
        return all_sigmas.len() - 1;
    }
    if (all_sigmas[idx - 1] - sigma).abs() <= (all_sigmas[idx] - sigma).abs() {
        idx - 1
    } else {
        idx
    }
}

// Build the Karras sigma schedule and the matching UNet timesteps.
fn build_karras_schedule(
    n_steps: usize,
    all_sigmas: &[f64],
    sigma_max: f64,
    sigma_min: f64,
) -> (Vec<f64>, Vec<usize>) {
    const RHO: f64 = 7.0;
    let min_inv_rho = sigma_min.powf(1.0 / RHO);
    let max_inv_rho = sigma_max.powf(1.0 / RHO);
    let mut sigmas: Vec<f64> = (0..n_steps)
        .map(|i| {
            let u = i as f64 / (n_steps - 1).max(1) as f64;
            (max_inv_rho + u * (min_inv_rho - max_inv_rho)).powf(RHO)
        })
        .collect();
    sigmas.push(0.0);
    // For each Karras sigma find the UNet timestep with the matching noise level.
    let timesteps: Vec<usize> = sigmas[..n_steps]
        .iter()
        .map(|&s| sigma_to_t(s, all_sigmas))
        .collect();
    (sigmas, timesteps)
}

// ─────────────────────────────────────────────────────────────────────────────
// Karras sigma schedule wrapped around EulerA steps
// ─────────────────────────────────────────────────────────────────────────────

struct KarrasEulerAScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    init_noise_sigma: f64,
}

impl KarrasEulerAScheduler {
    fn new(n_steps: usize) -> Result<Self> {
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
        let (sigmas, timesteps) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
        let init_noise_sigma = (sigma_max * sigma_max + 1.0).sqrt();
        Ok(Self { sigmas, timesteps, init_noise_sigma })
    }
}

impl Scheduler for KarrasEulerAScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        sample / (self.sigmas[i] * self.sigmas[i] + 1.0).sqrt()
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        let sigma_from = self.sigmas[i];
        let sigma_to = self.sigmas[i + 1];

        // Predicted denoised sample (epsilon prediction)
        let pred_x0 = (sample - (model_output * sigma_from)?)?;

        // Stochastic noise split: sigma_up^2 + sigma_down^2 = sigma_to^2
        let sigma_up = (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).sqrt();

        let derivative = ((sample - pred_x0)? / sigma_from)?;
        let prev_sample = (sample + (derivative * (sigma_down - sigma_from))?)?;
        let noise = prev_sample.randn_like(0.0, 1.0)?;
        prev_sample + (noise * sigma_up)?
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: Tensor,
        timestep: usize,
    ) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        original + (noise * self.sigmas[i])?
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DPM++ 2M Karras scheduler
// Pure-sigma parameterisation: x = x₀ + σ·ε (α ≡ 1).
// scale_model_input normalises to unit variance, matching the EulerA convention.
// ─────────────────────────────────────────────────────────────────────────────

struct Dpm2mKarrasScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    prev_denoised: Option<Tensor>,
}

impl Dpm2mKarrasScheduler {
    fn new(n_steps: usize) -> Result<Self> {
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
        let (sigmas, timesteps) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
        Ok(Self { sigmas, timesteps, prev_denoised: None })
    }
}

impl Scheduler for Dpm2mKarrasScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        let sigma = self.sigmas[i];
        sample / (sigma * sigma + 1.0).sqrt()
    }

    fn init_noise_sigma(&self) -> f64 {
        let s = self.sigmas[0];
        (s * s + 1.0).sqrt()
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        let sigma_from = self.sigmas[i];
        let sigma_to = self.sigmas[i + 1];

        // x₀ estimate: D₀ = x - σ·ε  (pure-sigma, α ≡ 1)
        let denoised = (sample - (model_output * sigma_from)?)?;

        let x_next = if sigma_to == 0.0 {
            denoised.clone()
        } else {
            // h = ln(σ_from/σ_to) > 0, ratio = σ_to/σ_from = exp(-h)
            let h = sigma_from.ln() - sigma_to.ln();
            let ratio = sigma_to / sigma_from;

            match self.prev_denoised.take() {
                None => {
                    // 1st order (exact solution to the sigma-space ODE with constant D₀)
                    ((sample * ratio)? + (&denoised * (1.0 - ratio))?)?
                }
                Some(prev_d) => {
                    // 2nd order DPM++ 2M midpoint correction
                    // D₁ = (D₀ - D₀_prev) / r,  denoised_d = D₀ + ½·D₁
                    let h_last = self.sigmas[i - 1].ln() - sigma_from.ln();
                    let r = h_last / h;
                    let c1 = 1.0 + 1.0 / (2.0 * r);
                    let c2 = 1.0 / (2.0 * r);
                    let denoised_d = ((&denoised * c1)? - (&prev_d * c2)?)?;
                    ((sample * ratio)? + (denoised_d * (1.0 - ratio))?)?
                }
            }
        };

        self.prev_denoised = Some(denoised);
        Ok(x_next)
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: Tensor,
        timestep: usize,
    ) -> candle_core::Result<Tensor> {
        let i = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in schedule"))
            })?;
        original + (noise * self.sigmas[i])?
    }
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
        let (patched, applied) = apply_te_lora(tensors, lora_map, te_prefix, scale)?;
        if applied > 0 {
            info!("TE LoRA ({te_prefix}): applied to {applied} layers");
        }
        VarBuilder::from_tensors(patched, DType::F32, ctx.device)
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, ctx.device)? }
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

// ─────────────────────────────────────────────────────────────────────────────
// LoRA merging
// ─────────────────────────────────────────────────────────────────────────────

fn apply_lora(
    mut tensors: std::collections::HashMap<String, Tensor>,
    lora: &std::collections::HashMap<String, Tensor>,
    lora_scale: f64,
) -> Result<std::collections::HashMap<String, Tensor>> {
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
    info!("LoRA applied to {applied} layers");
    Ok(tensors)
}

fn merge_lora_layer(
    tensors: &std::collections::HashMap<String, Tensor>,
    lora: &std::collections::HashMap<String, Tensor>,
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

fn apply_te_lora(
    mut tensors: std::collections::HashMap<String, Tensor>,
    lora: &std::collections::HashMap<String, Tensor>,
    te_prefix: &str,
    lora_scale: f64,
) -> Result<(std::collections::HashMap<String, Tensor>, usize)> {
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

/// Converts the inner part of a TE LoRA base (after stripping "lora_te1_" / "lora_te2_")
/// to the weight key used in the safetensors file.
/// e.g. "text_model_encoder_layers_0_self_attn_q_proj" → "text_model.encoder.layers.0.self_attn.q_proj.weight"
fn te_lora_base_to_weight_key(base: &str) -> Option<String> {
    const TE_TOKENS: &[(&str, &str)] = &[
        ("text_model_",  "text_model"),
        ("encoder_",     "encoder"),
        ("layers_",      "layers"),
        ("self_attn_",   "self_attn"),
        ("mlp_",         "mlp"),
        ("layer_norm1",  "layer_norm1"),
        ("layer_norm2",  "layer_norm2"),
        ("out_proj",     "out_proj"),
        ("q_proj",       "q_proj"),
        ("k_proj",       "k_proj"),
        ("v_proj",       "v_proj"),
        ("fc1",          "fc1"),
        ("fc2",          "fc2"),
    ];
    let result = greedy_tokenize(base, TE_TOKENS);
    if result.is_empty() { return None; }
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
                if !result.is_empty() { result.push('.'); }
                result.push_str(tok);
            }
            None => {
                let end = s.find('_').unwrap_or(s.len());
                if !result.is_empty() { result.push('.'); }
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

// ─────────────────────────────────────────────────────────────────────────────
// Image saving
// ─────────────────────────────────────────────────────────────────────────────

fn save_image(img: &Tensor, path: &str) -> Result<()> {
    let abs = std::env::current_dir().unwrap_or_default().join(path);
    let path = abs.as_path();
    let (_c, h, w) = img.dims3()?;
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
    info!("Saved: {}", path.display());
    Ok(())
}
