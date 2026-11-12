use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Bumped whenever the hash input layout changes. The version is mixed into
/// every key, so entries written by an older layout are never reused.
const KEY_FORMAT_VERSION: &str = "v2";

pub struct CacheKey {
    prefix: String,
    hash: String,
}

impl CacheKey {
    /// Hash the key parts unambiguously: the format version and the byte
    /// length of each part are mixed in before its bytes, so distinct
    /// sequences of parts ("ab","c" vs "a","bc" vs "abc") can never produce
    /// the same hash input.
    pub fn from_parts(parts: &[&str]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(KEY_FORMAT_VERSION.as_bytes());
        hasher.update([0]);
        for p in parts {
            hasher.update((p.len() as u64).to_le_bytes());
            hasher.update(p.as_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());
        CacheKey {
            prefix: parts.first().map(|s| s.to_string()).unwrap_or_default(),
            hash,
        }
    }
}

/// SHA-256 of a file's contents, as lowercase hex. Used to make cache keys
/// depend on what a LoRA/model file actually contains, not just on its path.
pub fn file_digest(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_splits_do_not_collide() {
        // Plain concatenation of parts would hash all of these identically.
        let a = CacheKey::from_parts(&["ab", "c"]);
        let b = CacheKey::from_parts(&["a", "bc"]);
        let c = CacheKey::from_parts(&["abc"]);
        assert_ne!(a.hash, b.hash);
        assert_ne!(a.hash, c.hash);
        assert_ne!(b.hash, c.hash);
    }

    #[test]
    fn part_count_and_empty_parts_do_not_collide() {
        let a = CacheKey::from_parts(&["a", "b"]);
        let b = CacheKey::from_parts(&["ab"]);
        let c = CacheKey::from_parts(&["a", "", "b"]);
        let d = CacheKey::from_parts(&["", "ab"]);
        let hashes = [a.hash, b.hash, c.hash, d.hash];
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "keys {i} and {j} collide");
            }
        }
    }

    #[test]
    fn same_parts_give_stable_key() {
        let a = CacheKey::from_parts(&["sdxl", "a cat", "cfg2", "BF16"]);
        let b = CacheKey::from_parts(&["sdxl", "a cat", "cfg2", "BF16"]);
        assert_eq!(a.hash, b.hash);
        assert_eq!(a.prefix, "sdxl");
    }

    #[test]
    fn changed_lora_content_changes_key() {
        // Regression: the SDXL cache key must depend on LoRA file contents,
        // so replacing the file at the same path invalidates the cache.
        let dir = std::env::temp_dir();
        let path_a = dir.join("etch-test-lora-a.safetensors");
        let path_b = dir.join("etch-test-lora-b.safetensors");
        std::fs::write(&path_a, b"lora-weights-v1").expect("write test file a");
        std::fs::write(&path_b, b"lora-weights-v2").expect("write test file b");

        let digest_a = file_digest(&path_a).expect("digest a");
        let digest_b = file_digest(&path_b).expect("digest b");
        assert_ne!(digest_a, digest_b);

        let key_a = CacheKey::from_parts(&["sdxl", "a cat", &digest_a, "1.0"]);
        let key_b = CacheKey::from_parts(&["sdxl", "a cat", &digest_b, "1.0"]);
        assert_ne!(key_a.hash, key_b.hash);

        // Same content hashed again must give the same key inputs.
        let digest_a2 = file_digest(&path_a).expect("digest a again");
        assert_eq!(digest_a, digest_a2);

        std::fs::remove_file(&path_a).expect("remove test file a");
        std::fs::remove_file(&path_b).expect("remove test file b");
    }
}
