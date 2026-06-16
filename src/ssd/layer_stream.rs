use crate::ssd::async_reader::{AsyncReader, IoResult};
use crate::ssd::offset_index::OffsetIndex;
use crate::ssd::weight_descriptor::LayerSpec;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

pub struct LayerStream {
    reader: AsyncReader,
    index: Arc<OffsetIndex>,
    device: Device,
    dtype: DType,
    layer_specs: Vec<LayerSpec>,
    total_layers: usize,
    current_layer_id: usize,
}

impl LayerStream {
    pub fn new(
        safetensors_path: &std::path::Path,
        layer_specs: Vec<LayerSpec>,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let index = Arc::new(OffsetIndex::from_file(safetensors_path)?);
        let reader = AsyncReader::spawn(safetensors_path);
        let total_layers = layer_specs.len();

        Ok(Self {
            reader,
            index,
            device: device.clone(),
            dtype,
            layer_specs,
            total_layers,
            current_layer_id: 0,
        })
    }

    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    pub fn begin(&mut self) {
        self.current_layer_id = 0;
        self.prefetch(0);
    }

    pub fn next_layer(&mut self) -> anyhow::Result<Vec<Tensor>> {
        let result = self.reader.wait_layer();
        let (lid, buffers) = match result {
            IoResult::LayerReady { layer_id, buffers } => (layer_id, buffers),
            IoResult::Error {
                layer_id,
                message,
            } => anyhow::bail!("SSD read error for layer {layer_id}: {message}"),
        };
        assert_eq!(lid, self.current_layer_id);

        let tensors: Vec<Tensor> = buffers
            .into_iter()
            .map(|(name, raw)| {
                let meta = self.index.tensor_meta(&name).expect("tensor meta in index");
                Tensor::from_raw_buffer(&raw, meta.dtype, &meta.shape, &self.device)
                    .expect("tensor from buffer")
                    .to_dtype(self.dtype)
                    .expect("cast to target dtype")
            })
            .collect();

        let next = self.current_layer_id + 1;
        if next < self.total_layers {
            self.prefetch(next);
        }

        self.current_layer_id = next;
        Ok(tensors)
    }

    fn prefetch(&self, layer_id: usize) {
        let spec = &self.layer_specs[layer_id];
        let metas: Vec<(String, _)> = spec
            .weights
            .iter()
            .map(|w| {
                let meta = self
                    .index
                    .tensor_meta(&w.name)
                    .expect("tensor meta in index")
                    .clone();
                (w.name.clone(), meta)
            })
            .collect();
        self.reader.request_layer(layer_id, metas);
    }
}
