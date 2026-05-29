use crate::cli::{Args, Model};
use anyhow::Result;
use candle_core::{DType, Device};

pub trait Pipeline {
    fn run(&self, args: &Args, device: &Device, dtype: DType) -> Result<()>;
}

pub fn for_model(model: Model) -> Box<dyn Pipeline> {
    match model {
        Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
            Box::new(crate::flux::FluxPipeline)
        }
        Model::Araminta => Box::new(crate::sdxl::SdxlPipeline),
    }
}
