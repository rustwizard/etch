#![deny(clippy::unwrap_used)]

mod cli;
mod device;
mod image;
mod lora;
mod logger;
mod schedulers;
mod sdxl;

use cli::{Args, Model, Quantization};

use candle_transformers::models::{clip, flux, t5};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
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
            Model::Araminta => sdxl::run_sdxl(&iter_args, &device, dtype),
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


