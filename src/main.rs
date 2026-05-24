#![deny(clippy::unwrap_used)]

use candle_transformers::models::{clip, flux, stable_diffusion, t5};
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
    /// FLUX.1-schnell — 4 steps, fast, Apache 2.0
    Schnell,
    /// FLUX.1-dev — 50 steps, best quality, non-commercial
    Dev,
    /// John6666/the-araminta-experiment-fv5-sdxl — ~7 GB SDXL
    Araminta,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "metal")]
        { Device::new_metal(0).unwrap_or_else(|e| { eprintln!("Metal init failed: {e}. Falling back to CPU."); Device::Cpu }) }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        { Device::new_cuda(0).unwrap_or_else(|e| { eprintln!("CUDA init failed: {e}. Falling back to CPU."); Device::Cpu }) }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        { Device::Cpu }
    };
    println!("Device: {:?}", device);

    let seed = args.seed.unwrap_or_else(rand::random);
    println!("Seed: {seed}");
    device.set_seed(seed)?;

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
    let args = Args { output: Some(output), ..args };

    let t0 = std::time::Instant::now();
    match args.model {
        Model::Schnell | Model::Dev => run_flux(&args, &device, dtype)?,
        Model::Araminta => run_sdxl(&args, &device, dtype)?,
    }
    println!("Total time: {:.1}s", t0.elapsed().as_secs_f32());

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
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)?;
    use std::io::Write as _;
    writeln!(log, "{}", entry)?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// FLUX pipeline
// ─────────────────────────────────────────────────────────────────────────────

fn run_flux(args: &Args, device: &Device, dtype: DType) -> Result<()> {
    let height = args.height.unwrap_or(768);
    let width = args.width.unwrap_or(1360);
    let api = hf_hub::api::sync::Api::new()?;

    let bf_repo = {
        let name = match args.model {
            Model::Dev => "black-forest-labs/FLUX.1-dev",
            _ => "black-forest-labs/FLUX.1-schnell",
        };
        api.repo(hf_hub::Repo::model(name.to_string()))
    };

    // T5 text encoder
    let t5_emb = {
        let repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
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
        model.forward(&Tensor::new(&tokens[..], device)?.unsqueeze(0)?)?
    };
    println!("T5: {:?}", t5_emb.shape());

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
        model.forward(&Tensor::new(&tokens[..], device)?.unsqueeze(0)?)?
    };
    println!("CLIP: {:?}", clip_emb.shape());

    // FLUX DiT
    let img = {
        let cfg = match args.model {
            Model::Dev => flux::model::Config::dev(),
            _ => flux::model::Config::schnell(),
        };
        let img = flux::sampling::get_noise(1, height, width, device)?.to_dtype(dtype)?;
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
        let n_steps = args.n_steps.unwrap_or(match args.model {
            Model::Dev => 50,
            _ => 4,
        });
        let timesteps = match args.model {
            Model::Dev => {
                flux::sampling::get_schedule(n_steps, Some((state.img.dim(1)?, 0.5, 1.15)))
            }
            _ => flux::sampling::get_schedule(n_steps, None),
        };
        let model_file = match args.model {
            Model::Dev => bf_repo.get("flux1-dev.safetensors")?,
            _ => bf_repo.get("flux1-schnell.safetensors")?,
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
        let model = flux::model::Flux::new(&cfg, vb)?;
        let denoised = flux::sampling::denoise(
            &model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            4.,
        )?;
        flux::sampling::unpack(&denoised, height, width)?
    };

    // VAE decode
    let img = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[bf_repo.get("ae.safetensors")?], dtype, device)?
        };
        let cfg = match args.model {
            Model::Dev => flux::autoencoder::Config::dev(),
            _ => flux::autoencoder::Config::schnell(),
        };
        flux::autoencoder::AutoEncoder::new(&cfg, vb)?.decode(&img)?
    };

    let img = img.to_device(&Device::Cpu)?;
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let out = args.output.as_deref().expect("output set in main");
    save_image(&img.i(0)?, out)?;
    println!("Saved to {out}");
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

    let sd_config =
        stable_diffusion::StableDiffusionConfig::sdxl(None, Some(height), Some(width));
    let mut scheduler: Box<dyn Scheduler> = match args.scheduler {
        SamplerType::EulerA => {
            println!("Scheduler: Euler Ancestral");
            Box::new(EulerAncestralDiscreteScheduler::new(
                n_steps,
                EulerAncestralDiscreteSchedulerConfig {
                    prediction_type: PredictionType::Epsilon,
                    ..Default::default()
                },
            )?)
        }
        SamplerType::EulerAKarras => {
            println!("Scheduler: Euler Ancestral + Karras");
            Box::new(KarrasEulerAScheduler::new(n_steps)?)
        }
        SamplerType::Dpm2mKarras => {
            println!("Scheduler: DPM++ 2M Karras");
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
        )?;
        let clip2_config = sd_config.clip2.as_ref().ok_or_else(|| anyhow::anyhow!("SDXL config missing clip2"))?;
        let emb2 = sdxl_clip_emb(
            &ctx,
            &tok2,
            model_file("text_encoder_2/model.safetensors")?,
            clip2_config,
        )?;
        // [batch, 77, 768] ++ [batch, 77, 1280] → [batch, 77, 2048]
        Tensor::cat(&[emb1, emb2], D::Minus1)?
    };
    println!("Text embeddings: {:?}", text_embeddings.shape());

    let vae = sd_config.build_vae(
        model_file("vae/diffusion_pytorch_model.safetensors")?,
        device,
        DType::F32,
    )?;
    let unet = {
        let unet_weights = model_file("unet/diffusion_pytorch_model.safetensors")?;
        if let Some(lora_path) = &args.lora {
            println!("Applying LoRA: {lora_path} (scale {})", args.lora_scale);
            let mut tensors = candle_core::safetensors::load(&unet_weights, &Device::Cpu)?;
            tensors = apply_lora(tensors, lora_path, args.lora_scale, &Device::Cpu)?;
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
        println!(
            "step {}/{n_steps} — {:.1}s",
            i + 1,
            t_step.elapsed().as_secs_f32()
        );
    }

    drop(unet);
    drop(text_embeddings);

    let img = vae.decode(&(latents.to_dtype(DType::F32)? / vae_scale)?)?;
    drop(vae);
    let img = img.to_device(&Device::Cpu)?;
    let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
    let img = (img * 255.)?.to_dtype(DType::U8)?;
    let out = args.output.as_deref().expect("output set in main");
    save_image(&img.i(0)?, out)?;
    println!("Saved to {out}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Karras utilities
// ─────────────────────────────────────────────────────────────────────────────

// Map a Karras sigma back to the nearest discrete timestep in the original
// schedule. all_sigmas is monotone-increasing (sigma grows with timestep).
fn sigma_to_t(sigma: f64, all_sigmas: &[f64]) -> usize {
    let idx = all_sigmas.partition_point(|&s| s < sigma);
    if idx == 0 { return 0; }
    if idx >= all_sigmas.len() { return all_sigmas.len() - 1; }
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
        const BETA_START: f64 = 0.00085;
        const BETA_END: f64 = 0.012;
        const TRAIN_STEPS: usize = 1000;
        const STEPS_OFFSET: usize = 1;

        // ScaledLinear betas (same as EulerA default for SDXL)
        let betas: Vec<f64> = (0..TRAIN_STEPS)
            .map(|i| {
                let t = i as f64 / (TRAIN_STEPS - 1) as f64;
                let b = BETA_START.sqrt() + t * (BETA_END.sqrt() - BETA_START.sqrt());
                b * b
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(TRAIN_STEPS);
        let mut cumprod = 1.0f64;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        // sigma_t = sqrt((1 - ᾱ_t) / ᾱ_t)
        let all_sigmas: Vec<f64> = alphas_cumprod.iter()
            .map(|&a| ((1.0 - a) / a).sqrt())
            .collect();

        // sigma_max / sigma_min from the standard linear SDXL timestep grid
        let step_ratio = TRAIN_STEPS / n_steps;
        let sigma_max = all_sigmas[(n_steps - 1) * step_ratio + STEPS_OFFSET];
        let sigma_min = all_sigmas[step_ratio + STEPS_OFFSET];

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
        let i = self.timesteps.iter().position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
        sample / (self.sigmas[i] * self.sigmas[i] + 1.0).sqrt()
    }

    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> candle_core::Result<Tensor> {
        let i = self.timesteps.iter().position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
        let sigma_from = self.sigmas[i];
        let sigma_to = self.sigmas[i + 1];

        // Predicted denoised sample (epsilon prediction)
        let pred_x0 = (sample - (model_output * sigma_from)?)?;

        // Stochastic noise split: sigma_up^2 + sigma_down^2 = sigma_to^2
        let sigma_up = (sigma_to * sigma_to
            * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).sqrt();

        let derivative = ((sample - pred_x0)? / sigma_from)?;
        let prev_sample = (sample + (derivative * (sigma_down - sigma_from))?)?;
        let noise = prev_sample.randn_like(0.0, 1.0)?;
        prev_sample + (noise * sigma_up)?
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = self.timesteps.iter().position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
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
        const BETA_START: f64 = 0.00085;
        const BETA_END: f64 = 0.012;
        const TRAIN_STEPS: usize = 1000;
        const STEPS_OFFSET: usize = 1;

        let betas: Vec<f64> = (0..TRAIN_STEPS)
            .map(|i| {
                let t = i as f64 / (TRAIN_STEPS - 1) as f64;
                let b = BETA_START.sqrt() + t * (BETA_END.sqrt() - BETA_START.sqrt());
                b * b
            })
            .collect();
        let mut alphas_cumprod = Vec::with_capacity(TRAIN_STEPS);
        let mut cumprod = 1.0f64;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }
        let all_sigmas: Vec<f64> =
            alphas_cumprod.iter().map(|&a| ((1.0 - a) / a).sqrt()).collect();

        let step_ratio = TRAIN_STEPS / n_steps;
        let sigma_max = all_sigmas[(n_steps - 1) * step_ratio + STEPS_OFFSET];
        let sigma_min = all_sigmas[step_ratio + STEPS_OFFSET];

        let (sigmas, timesteps) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
        Ok(Self { sigmas, timesteps, prev_denoised: None })
    }
}

impl Scheduler for Dpm2mKarrasScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = self.timesteps.iter().position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
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
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
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
            .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))?;
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
) -> Result<Tensor> {
    let vocab = tokenizer.get_vocab(true);
    let pad_id = match &clip_config.pad_with {
        Some(p) => *vocab.get(p.as_str()).ok_or_else(|| anyhow::anyhow!("pad token '{p}' not in vocab"))?,
        None => *vocab.get("<|endoftext|>").ok_or_else(|| anyhow::anyhow!("'<|endoftext|>' not in vocab"))?,
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

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, ctx.device)? };
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
    lora_path: &str,
    lora_scale: f64,
    device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let lora = candle_core::safetensors::load(lora_path, device)?;
    let mut applied = 0usize;

    let weight_keys: Vec<String> = tensors
        .keys()
        .filter(|k| k.ends_with(".weight"))
        .cloned()
        .collect();

    for weight_key in weight_keys {
        // Base key → LoRA key: replace '.' with '_', prepend "lora_unet_"
        // e.g. "down_blocks.0.attentions.0.to_q.weight"
        //   → "lora_unet_down_blocks_0_attentions_0_to_q"
        let base = weight_key.strip_suffix(".weight").expect("filtered by ends_with above");
        let lora_base = format!("lora_unet_{}", base.replace('.', "_"));

        let down_key = format!("{lora_base}.lora_down.weight");
        let up_key = format!("{lora_base}.lora_up.weight");

        let (Some(lora_down), Some(lora_up)) = (lora.get(&down_key), lora.get(&up_key)) else {
            continue;
        };

        let rank = lora_down.dim(0)?;
        let alpha_key = format!("{lora_base}.alpha");
        let scale = if let Some(alpha) = lora.get(&alpha_key) {
            let alpha_val = alpha.to_vec0::<f32>()? as f64;
            lora_scale * alpha_val / rank as f64
        } else {
            lora_scale
        };

        // delta = scale * (lora_up @ lora_down)  shape: [out, in]
        let down = lora_down.to_dtype(DType::F32)?;
        let up = lora_up.to_dtype(DType::F32)?;
        let delta = (up.matmul(&down)? * scale)?;

        let w = tensors.get(&weight_key).expect("key came from tensors");
        let orig_dtype = w.dtype();
        let merged = (w.to_dtype(DType::F32)? + delta)?.to_dtype(orig_dtype)?;
        tensors.insert(weight_key, merged);
        applied += 1;
    }

    println!("LoRA applied to {applied} layers");
    Ok(tensors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Image saving
// ─────────────────────────────────────────────────────────────────────────────

fn save_image(img: &Tensor, path: &str) -> Result<()> {
    let abs = std::env::current_dir().unwrap_or_default().join(path);
    let path = abs.as_path();
    println!("Saving image to: {}", path.display());
    let (_c, h, w) = img.dims3()?;
    // Single flat allocation (~3MB) instead of 1M Vec<u8> objects from to_vec2
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
    println!("Saved: {}", path.display());
    Ok(())
}
