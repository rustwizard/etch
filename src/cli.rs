use anyhow::Result;
use candle_core::DType;
use clap::Parser;

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value = "A rusty robot walking on a beach")]
    pub prompt: String,

    /// Negative prompt (SDXL only).
    #[arg(long, default_value = "")]
    pub uncond_prompt: String,

    /// Height in pixels. Default: 768 (FLUX) or 1024 (SDXL).
    #[arg(long)]
    pub height: Option<usize>,

    /// Width in pixels. Default: 1360 (FLUX) or 1024 (SDXL).
    #[arg(long)]
    pub width: Option<usize>,

    /// Denoising steps. Default: 4 (schnell), 50 (dev), 20 (araminta).
    #[arg(long)]
    pub n_steps: Option<usize>,

    #[arg(long)]
    pub seed: Option<u64>,

    /// Generate a batch of images with seeds in range [start,end], e.g. 0-100
    #[arg(long, value_name = "START-END")]
    pub seed_range: Option<String>,

    #[arg(long)]
    pub output: Option<String>,

    #[arg(long, value_enum, default_value = "schnell")]
    pub model: Model,

    /// Classifier-free guidance scale (SDXL only, default 7.5).
    #[arg(long, default_value_t = 7.5)]
    pub guidance_scale: f64,

    /// Force CPU instead of Metal.
    #[arg(long)]
    pub cpu: bool,

    /// Path to a LoRA .safetensors file (SDXL only).
    #[arg(long)]
    pub lora: Option<String>,

    /// LoRA strength, 0.0–1.0 (default 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f64,

    /// Path to a local model directory in diffusers format (SDXL only).
    /// Overrides HuggingFace download. Convert with:
    ///   diffusers StableDiffusionXLPipeline.from_single_file(...).save_pretrained(path)
    #[arg(long)]
    pub local_model: Option<String>,

    /// CLIP skip: number of layers to skip from the end of the text encoder (SDXL only).
    /// 1 = last layer (default), 2 = penultimate, 3–4 = earlier layers.
    #[arg(long, default_value_t = 1)]
    pub clip_skip: usize,

    /// Sampler (SDXL only).
    #[arg(long, value_enum, default_value = "euler-a")]
    pub scheduler: SamplerType,

    /// Path to a local FLUX GGUF file (e.g. flux1-schnell-Q8_0.gguf). Skips HF download.
    #[arg(long)]
    pub gguf: Option<String>,

    /// Quantization level for schnell-gguf / dev-gguf models (default: q8).
    #[arg(long, value_enum, default_value = "q8")]
    pub quantization: Quantization,

    /// Tensor dtype for model weights (default: bf16 on Metal, f32 on CPU).
    /// BF16 halves memory vs F32 with negligible quality loss. GGUF ignores this flag.
    #[arg(long, value_enum)]
    pub dtype: Option<DtypeArg>,

    /// Force VAE decode on CPU. Slower (~10–30s on 1024×1024) but avoids the
    /// Metal pool peak from VAE intermediate activations — useful on tight memory.
    #[arg(long)]
    pub vae_cpu: bool,

    /// Guidance scale for FLUX.1-dev (ignored for schnell, which is distilled and needs no guidance).
    /// Typical range 1.0–5.0. Default: 3.5.
    #[arg(long, default_value_t = 3.5)]
    pub flux_guidance: f64,

    /// Enable verbose logging: show timestamps and module targets.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum DtypeArg {
    F32,
    Bf16,
    F16,
}

impl From<DtypeArg> for DType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => DType::F32,
            DtypeArg::Bf16 => DType::BF16,
            DtypeArg::F16 => DType::F16,
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Default)]
pub enum Quantization {
    /// Q8_0 — ~12 GB, best quality
    #[default]
    Q8,
    /// Q4_K_S — ~7 GB, smaller but slightly lower quality
    Q4,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Default)]
pub enum SamplerType {
    /// Euler Ancestral — stochastic, fast, good all-rounder
    #[default]
    EulerA,
    /// Euler Ancestral + Karras sigma spacing
    EulerAKarras,
    /// DPM++ 2M Karras — smooth, great at 20–30 steps
    Dpm2mKarras,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Model {
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

pub fn parse_seed_range(s: &str) -> Result<Vec<u64>> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        anyhow::bail!("seed-range must be in format START-END, e.g. 0-100");
    }
    let start: u64 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("seed-range start must be a number"))?;
    let end: u64 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("seed-range end must be a number"))?;
    if end < start {
        anyhow::bail!("seed-range end must be >= start");
    }
    Ok((start..=end).collect())
}

pub fn output_for_seed(base: &Option<String>, seed: u64) -> String {
    match base {
        Some(path) => {
            let p = std::path::Path::new(path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let dir = p
                .parent()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty());
            match dir {
                Some(d) => format!("{d}/{stem}-{seed}.{ext}"),
                None => format!("{stem}-{seed}.{ext}"),
            }
        }
        None => {
            let n: u32 = rand::random();
            format!("out/out-{seed}-{n}.png")
        }
    }
}
