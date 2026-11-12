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
        // Regression: the SDXL cache key must depend on file contents, so
        // replacing a LoRA file at the same path invalidates the cache. The
        // test rewrites one file in place — exactly that scenario. The temp
        // dir is unique per process so parallel test binaries cannot race.
        let dir = std::env::temp_dir().join(format!(
            "etch-cache-test-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).expect("create unique temp dir");
        let lora_path = dir.join("adapter.safetensors");

        std::fs::write(&lora_path, b"lora-weights-v1").expect("write v1");
        let digest_v1 = file_digest(&lora_path).expect("digest v1");
        std::fs::write(&lora_path, b"lora-weights-v2").expect("write v2");
        let digest_v2 = file_digest(&lora_path).expect("digest v2");
        assert_ne!(digest_v1, digest_v2);

        let key_v1 = CacheKey::from_parts(&["sdxl", "a cat", &digest_v1, "1.0"]);
        let key_v2 = CacheKey::from_parts(&["sdxl", "a cat", &digest_v2, "1.0"]);
        assert_ne!(key_v1.hash, key_v2.hash);

        // Same content hashed again must give the same digest.
        let digest_v2_again = file_digest(&lora_path).expect("digest v2 again");
        assert_eq!(digest_v2, digest_v2_again);

        std::fs::remove_dir_all(&dir).expect("remove unique temp dir");
    }
}
