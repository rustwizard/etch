use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct CacheKey {
    prefix: String,
    hash: String,
}

impl CacheKey {
    pub fn from_parts(parts: &[&str]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for p in parts {
            hasher.update(p.as_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());
        CacheKey {
            prefix: parts.first().map(|s| s.to_string()).unwrap_or_default(),
            hash,
        }
    }
}

pub struct EmbeddingCache {
    root: PathBuf,
}

impl EmbeddingCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn default_dir() -> PathBuf {
        dirs::home_dir()
            .expect("no home directory")
            .join(".cache")
            .join("etch")
            .join("embeddings")
    }

    fn path_for(&self, key: &CacheKey) -> PathBuf {
        self.root
            .join(format!("{}-{}.safetensors", key.prefix, key.hash))
    }

    pub fn get(
        &self,
        key: &CacheKey,
        names: &[&str],
        device: &Device,
    ) -> Result<Option<HashMap<String, Tensor>>> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(None);
        }
        let tensors = candle_core::safetensors::load(&path, device)?;
        for name in names {
            if !tensors.contains_key(*name) {
                return Ok(None);
            }
        }
        Ok(Some(tensors))
    }

    pub fn set(&self, key: &CacheKey, tensors: &HashMap<String, Tensor>) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.path_for(key);
        let cpu_tensors: HashMap<String, Tensor> = tensors
            .iter()
            .map(|(k, v)| Ok((k.clone(), v.to_device(&Device::Cpu)?)))
            .collect::<Result<_>>()?;
        candle_core::safetensors::save(&cpu_tensors, &path)?;
        Ok(())
    }
}
