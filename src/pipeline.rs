use crate::cli::{Args, Model};
use anyhow::Result;
use candle_core::{DType, Device};

/// A text-to-image pipeline split into two phases so batch generation can
/// reuse the (expensive) model weights across seeds.
///
/// `prepare` loads all weights and reusable state once. `generate` produces a
/// single image and must be called after `prepare`. `generate` may be called
/// many times; the model weights stay resident for the lifetime of the pipeline.
pub trait Pipeline {
    fn prepare(&mut self, args: &Args, device: &Device, dtype: DType) -> Result<()>;
    fn generate(&self, args: &Args) -> Result<()>;
    /// Device on which random tensors (initial noise/latents) are created.
    /// Seeding any other device has no effect on generation. Only valid
    /// after `prepare`.
    fn noise_device(&self) -> &Device;
}

pub fn for_model(model: Model) -> Box<dyn Pipeline> {
    match model {
        Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
            Box::new(crate::flux::FluxPipeline::default())
        }
        Model::Araminta => Box::new(crate::sdxl::SdxlPipeline::default()),
    }
}
