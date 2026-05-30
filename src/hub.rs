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

/// Logs the on-disk size of a model weight file as an approximate loaded size.
pub fn log_model_size(path: &std::path::Path, label: &str) {
    if let Ok(meta) = std::fs::metadata(path) {
        let bytes = meta.len();
        let size_str = if bytes >= 1_073_741_824 {
            format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
        } else {
            format!("{:.0} MB", bytes as f64 / 1_048_576.0)
        };
        tracing::info!("{label}: ~{size_str} loaded");
    }
}
