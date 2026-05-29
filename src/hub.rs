use anyhow::Result;
use hf_hub::api::sync::ApiRepo;
use std::path::PathBuf;

/// Wraps `repo.get(filename)` with an info log so the user sees what's being
/// fetched. On first run this coincides with an actual download; on subsequent
/// runs files are returned instantly from the local HF cache.
pub fn fetch(repo: &ApiRepo, filename: &str) -> Result<PathBuf> {
    tracing::info!("Fetching {filename}...");
    Ok(repo.get(filename)?)
}
