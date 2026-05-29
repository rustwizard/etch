use crate::cli::{Args, SamplerType};
use crate::schedulers::{Dpm2mKarrasScheduler, KarrasEulerAScheduler};
use crate::{image, lora};
use anyhow::{Error as E, Result};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{
    self,
    euler_ancestral_discrete::{
        EulerAncestralDiscreteScheduler, EulerAncestralDiscreteSchedulerConfig,
    },
    schedulers::{PredictionType, Scheduler},
    unet_2d,
};
use tokenizers::Tokenizer;
use tracing::info;

use super::clip::{ClipEmbedCtx, sdxl_clip_emb};

pub fn run_sdxl(args: &Args, device: &Device, dtype: DType) -> Result<()> {
    let height = args.height.unwrap_or(768);
    let width = args.width.unwrap_or(1024);
    anyhow::ensure!(
        height.is_multiple_of(8),
        "--height must be a multiple of 8, got {height}"
    );
    anyhow::ensure!(
        width.is_multiple_of(8),
        "--width must be a multiple of 8, got {width}"
    );
    let n_steps = args.n_steps.unwrap_or(20);
    let guidance_scale = args.guidance_scale;
    anyhow::ensure!(
        guidance_scale > 0.0,
        "--guidance-scale must be positive, got {guidance_scale}"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&args.lora_scale),
        "--lora-scale must be in 0.0–1.0, got {}",
        args.lora_scale
    );
    anyhow::ensure!(
        args.clip_skip >= 1 && args.clip_skip <= 12,
        "--clip-skip must be in 1–12, got {}",
        args.clip_skip
    );
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
    let pb = crate::progress::denoising_bar(n_steps);
    for &timestep in &timesteps {
        let step_start = std::time::Instant::now();
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
        pb.set_message(format!("{:.1}s/step", step_start.elapsed().as_secs_f32()));
        pb.inc(1);
    }
    pb.finish_with_message("done");

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
