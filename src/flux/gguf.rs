use anyhow::Result;
use candle_core::Device;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

pub(crate) fn load_gguf_with_spinner(
    path: impl AsRef<std::path::Path>,
    label: &str,
    _device: &Device,
) -> Result<QVarBuilder> {
    use std::io::Write as _;
    let spinner = ['|', '/', '-', '\\'];
    print!("Loading GGUF: {label}  ");
    let _ = std::io::stdout().flush();

    let path = path.as_ref().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel::<Result<QVarBuilder>>();
    std::thread::spawn(move || {
        let _ = tx.send(QVarBuilder::from_gguf(path, &Device::Cpu).map_err(Into::into));
    });

    let mut i = 0usize;
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(120)) {
            Ok(result) => {
                print!("\rLoading GGUF: {label}  done\n");
                let _ = std::io::stdout().flush();
                return result;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                print!("\rLoading GGUF: {label}  {}", spinner[i % 4]);
                let _ = std::io::stdout().flush();
                i += 1;
            }
            Err(e) => return Err(anyhow::anyhow!("GGUF loader thread error: {e}")),
        }
    }
}
