use crate::cli::{Args, Model};
use anyhow::Result;

pub fn write_log_entry(out_path: &str, args: &Args, seed: u64) -> Result<()> {
    let log_path = std::path::Path::new(out_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("log.jsonl");
    let mut entry = serde_json::json!({
        "file": out_path,
        "seed": seed,
        "prompt": args.prompt,
        "model": format!("{:?}", args.model).to_lowercase(),
        "steps": args.n_steps,
    });
    if args.model == Model::Araminta {
        entry["scheduler"] =
            serde_json::json!(format!("{:?}", args.scheduler).to_lowercase());
        entry["guidance_scale"] = serde_json::json!(args.guidance_scale);
        if !args.uncond_prompt.is_empty() {
            entry["uncond_prompt"] = serde_json::json!(args.uncond_prompt);
        }
    }
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    use std::io::Write as _;
    writeln!(log, "{}", entry)?;
    Ok(())
}
