use candle_core::Tensor;
use candle_transformers::models::flux;

pub enum FluxModel {
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
