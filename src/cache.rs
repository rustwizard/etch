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

    /// Look up a cache entry. `Ok(None)` is a silent miss (no entry exists).
    /// `Err` means an entry exists but is unreadable or incomplete — callers
    /// should warn with the reason and recompute, which also overwrites the
    /// bad entry via `set`.
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
        let tensors = candle_core::safetensors::load(&path, device)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        for name in names {
            if !tensors.contains_key(*name) {
                anyhow::bail!("cache entry {} is missing tensor '{name}'", path.display());
            }
        }
        Ok(Some(tensors))
    }

    pub fn set(&self, key: &CacheKey, tensors: &HashMap<String, Tensor>) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.path_for(key);
        // Write to a unique temp file in the same directory, then atomically
        // rename over the destination. A crash mid-write then leaves the
        // previous entry (or none) intact instead of a truncated safetensors
        // file that every later run would fail to read. The temp name is
        // unique per process and attempt so parallel runs never clash.
        let tmp = self.root.join(format!(
            ".{}.tmp-{}-{}",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("cache-entry"),
            std::process::id(),
            rand::random::<u32>()
        ));
        let result = (|| {
            let cpu_tensors: HashMap<String, Tensor> = tensors
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.to_device(&Device::Cpu)?)))
                .collect::<Result<_>>()?;
            candle_core::safetensors::save(&cpu_tensors, &tmp)?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            // Best-effort cleanup; a leftover temp file only wastes space
            // and is never read as a cache entry.
            let _ = std::fs::remove_file(&tmp);
        }
        result
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

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "etch-{name}-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ))
    }

    #[test]
    fn set_get_roundtrip() {
        let dir = unique_test_dir("cache-roundtrip");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let cache = EmbeddingCache::new(dir.clone());
        let key = CacheKey::from_parts(&["flux", "a cat", "F32"]);

        let mut tensors = HashMap::new();
        tensors.insert(
            "emb".to_string(),
            Tensor::full(0.5f32, (1, 4), &Device::Cpu).expect("test tensor"),
        );
        cache.set(&key, &tensors).expect("set");

        let got = cache
            .get(&key, &["emb"], &Device::Cpu)
            .expect("get")
            .expect("cache hit");
        let emb = got.get("emb").expect("emb present");
        assert_eq!(emb.dims(), &[1, 4]);
        assert_eq!(emb.to_vec2::<f32>().expect("vec"), vec![vec![0.5; 4]]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_replaces_corrupt_entry() {
        // Recovery: a truncated/partially written cache file (e.g. left by an
        // interrupted non-atomic write, or external corruption) must be
        // atomically replaced by set(), not block every later run.
        let dir = unique_test_dir("cache-corrupt");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let cache = EmbeddingCache::new(dir.clone());
        let key = CacheKey::from_parts(&["sdxl", "a cat", "cfg2", "BF16"]);

        let corrupt = cache.path_for(&key);
        std::fs::write(&corrupt, b"garbage-not-safetensors").expect("write corrupt file");

        let mut tensors = HashMap::new();
        tensors.insert(
            "emb".to_string(),
            Tensor::full(1.0f32, (2, 2), &Device::Cpu).expect("test tensor"),
        );
        cache.set(&key, &tensors).expect("set over corrupt entry");

        let got = cache
            .get(&key, &["emb"], &Device::Cpu)
            .expect("get after recovery")
            .expect("cache hit after recovery");
        assert_eq!(got.get("emb").expect("emb present").dims(), &[2, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_distinguishes_miss_from_corrupt_entry() {
        let dir = unique_test_dir("cache-get-errors");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let cache = EmbeddingCache::new(dir.clone());
        let key = CacheKey::from_parts(&["flux", "a cat", "F32"]);
        let device = Device::Cpu;

        // No entry yet: silent miss, not an error.
        assert!(
            cache
                .get(&key, &["emb"], &device)
                .expect("miss is fine")
                .is_none()
        );

        // Corrupt entry (e.g. interrupted write): an error with the reason,
        // so the caller can warn and recompute instead of silently loading.
        std::fs::write(cache.path_for(&key), b"garbage").expect("write corrupt entry");
        let err = cache
            .get(&key, &["emb"], &device)
            .expect_err("corrupt entry must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("cannot read"), "{msg}");

        // Well-formed but incomplete entry (missing expected tensor): also
        // an error, not a silent miss.
        let other = CacheKey::from_parts(&["flux", "a dog", "F32"]);
        let mut partial = HashMap::new();
        partial.insert(
            "other_tensor".to_string(),
            Tensor::full(0f32, (1,), &device).expect("test tensor"),
        );
        cache.set(&other, &partial).expect("set partial entry");
        let err = cache
            .get(&other, &["emb"], &device)
            .expect_err("missing tensor must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("missing tensor 'emb'"), "{msg}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
