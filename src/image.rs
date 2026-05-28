use anyhow::Result;
use candle_core::Tensor;

pub fn save_image(img: &Tensor, path: &str) -> Result<()> {
    let abs = std::env::current_dir().unwrap_or_default().join(path);
    let path = abs.as_path();
    let (c, h, w) = img.dims3()?;
    anyhow::ensure!(c == 3, "expected 3-channel RGB tensor, got {c} channels");
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
    tracing::info!("Saved: {}", path.display());
    Ok(())
}
