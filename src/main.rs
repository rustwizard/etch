#![deny(clippy::unwrap_used)]

mod cache;
mod cli;
mod device;
mod flux;
mod hub;
mod image;
mod logger;
mod lora;
mod pipeline;
mod progress;
mod schedulers;
mod sdxl;
mod signals;
mod vae_tiling;

use cli::Args;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use tracing::info;

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        !(args.seed.is_some() && args.seed_range.is_some()),
        "--seed and --seed-range are mutually exclusive"
    );
    cli::validate_n_steps(args.n_steps)?;
    cli::validate_args(&args)?;
    signals::install();

    let builder = tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    );
    if args.verbose {
        builder.init();
    } else {
        builder.without_time().with_target(false).init();
    }

    for warning in cli::ignored_flag_warnings(&args) {
        tracing::warn!("{warning}");
    }

    let device = device::pick_device(args.cpu);
    info!("Device: {:?}", device);

    let (seeds, seed_source) = if let Some(ref range_str) = args.seed_range {
        (cli::parse_seed_range(range_str)?, "range")
    } else if args.seed.is_some() {
        (vec![args.seed.unwrap_or_else(rand::random)], "explicit")
    } else {
        (vec![rand::random::<u64>()], "auto")
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

    // Load model weights once and reuse them across all seeds.
    let mut pipeline = pipeline::for_model(args.model);
    pipeline.prepare(&args, &device, dtype)?;

    // Seed the device that actually creates the random tensors. For FLUX GGUF
    // that is the CPU (the DiT runs there), not the GPU picked above — seeding
    // the GPU would silently have no effect. Candle cannot seed the CPU RNG
    // at all, so CPU runs are inherently non-reproducible.
    let noise_device = pipeline.noise_device();
    let cpu_rng = matches!(noise_device, Device::Cpu);
    if cpu_rng {
        tracing::warn!(
            "The CPU random number generator cannot be seeded — results are not reproducible on CPU; use a GPU backend with non-GGUF weights for reproducible --seed runs"
        );
    }

    for seed in seeds {
        if signals::interrupted() {
            info!("Interrupted — stopping before seed {seed}");
            std::process::exit(130);
        }
        info!("--- Seed: {seed} ({seed_source}) ---");
        if !cpu_rng && let Err(e) = noise_device.set_seed(seed) {
            tracing::warn!("Failed to set seed {seed}: {e}");
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
        let result = pipeline.generate(&iter_args);
        let out_path = iter_args.output.as_deref().expect("output set above");
        if let Err(e) = result {
            if signals::interrupted() {
                info!("Interrupted during seed {seed} — stopping");
                std::process::exit(130);
            }
            tracing::error!("Seed {seed} failed: {e}");
            if let Err(log_err) = logger::write_log_failure(out_path, &iter_args, seed, &e) {
                tracing::warn!("Failed to write failure log: {log_err}");
            }
            continue;
        }
        let elapsed = t0.elapsed().as_secs_f32();
        info!("Total time: {:.1}s", elapsed);
        logger::write_log_entry(out_path, &iter_args, seed, elapsed)?;
    }

    Ok(())
}
