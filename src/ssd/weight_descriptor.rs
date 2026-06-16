#[derive(Debug, Clone)]
pub struct WeightRef {
    pub name: String,
    pub description: &'static str,
}

impl WeightRef {
    pub fn new(name: impl Into<String>, description: &'static str) -> Self {
        Self {
            name: name.into(),
            description,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub id: usize,
    pub name: String,
    pub weights: Vec<WeightRef>,
    pub estimated_bytes: u64,
}

impl LayerSpec {
    pub fn new(id: usize, name: impl Into<String>, weights: Vec<WeightRef>) -> Self {
        Self {
            id,
            name: name.into(),
            weights,
            estimated_bytes: 0,
        }
    }

    pub fn with_estimated_bytes(mut self, bytes: u64) -> Self {
        self.estimated_bytes = bytes;
        self
    }
}

pub trait LayerStructure {
    fn layer_specs(&self) -> Vec<LayerSpec>;
}
