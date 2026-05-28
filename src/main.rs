#![deny(clippy::unwrap_used)]

mod cli;
mod device;
mod flux;
mod image;
mod logger;
mod lora;
mod schedulers;
mod sdxl;

use cli::{Args, Model};

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use tracing::info;

fn main() -> Result<()> {
    let args = Args::parse();

    let builder = tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    );
    if args.verbose {
        builder.init();
    } else {
        builder.without_time().with_target(false).init();
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

    for seed in seeds {
        info!("--- Seed: {seed} ({seed_source}) ---");
        if let Err(e) = device.set_seed(seed) {
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
        let result = match iter_args.model {
            Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
                flux::run_flux(&iter_args, &device, dtype)
            }
            Model::Araminta => sdxl::run_sdxl(&iter_args, &device, dtype),
        };
        let out_path = iter_args.output.as_deref().expect("output set above");
        if let Err(e) = result {
            tracing::error!("Seed {seed} failed: {e}");
            let _ = logger::write_log_failure(out_path, &iter_args, seed, &e);
            continue;
        }
        info!("Total time: {:.1}s", t0.elapsed().as_secs_f32());
        logger::write_log_entry(out_path, &iter_args, seed)?;
    }

    Ok(())
}
