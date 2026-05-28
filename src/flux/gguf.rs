use anyhow::Result;
use candle_core::Device;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

pub(crate) fn load_gguf(path: impl AsRef<std::path::Path>, label: &str) -> Result<QVarBuilder> {
    use std::io::Write as _;
    print!("Loading GGUF: {label}...");
    let _ = std::io::stdout().flush();
    let vb = QVarBuilder::from_gguf(path, &Device::Cpu)?;
    println!(" done");
    Ok(vb)
}
