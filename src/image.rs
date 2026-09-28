use anyhow::Result;
use candle_core::Tensor;

pub fn save_image(img: &Tensor, path: &str) -> Result<()> {
    let abs = std::env::current_dir().unwrap_or_default().join(path);
    let path = abs.as_path();
    let (c, h, w) = img.dims3()?;
    anyhow::ensure!(c == 3, "expected 3-channel RGB tensor, got {c} channels");
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        // Fast PNG compression: ~3–5× quicker to encode than the default at
        // ~10–20% larger files. Lossless — pixels are identical either way,
        // which matters for batch runs where saving adds up.
        use image::ImageEncoder as _;
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        let file = std::fs::File::create(path)?;
        let encoder =
            PngEncoder::new_with_quality(file, CompressionType::Fast, FilterType::Adaptive);
        encoder.write_image(&pixels, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    } else {
        image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
    }
    tracing::info!("Saved: {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tensor() -> Tensor {
        // 3×2×2 RGB gradient, deterministic
        let data: Vec<u8> = (0..12).map(|i| (i * 20) as u8).collect();
        Tensor::from_slice(&data, (3, 2, 2), &candle_core::Device::Cpu).expect("valid test tensor")
    }

    #[test]
    fn saves_valid_png() {
        let dir = std::env::temp_dir().join(format!("etch-test-png-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("t.png");
        save_image(&test_tensor(), path.to_str().expect("utf8 path")).expect("save png");
        let img = image::open(&path).expect("readable png");
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        // Round-trip check: channel 0, pixel (0,0) is 0; channel 2, pixel (1,1) is 11*20
        let rgb = img.to_rgb8();
        assert_eq!(rgb.get_pixel(0, 0).0[0], 0);
        assert_eq!(rgb.get_pixel(1, 1).0[2], 220);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn saves_valid_jpeg() {
        let dir = std::env::temp_dir().join(format!("etch-test-jpg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("t.jpg");
        save_image(&test_tensor(), path.to_str().expect("utf8 path")).expect("save jpeg");
        let img = image::open(&path).expect("readable jpeg");
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_non_rgb() {
        let gray = Tensor::zeros((1, 2, 2), candle_core::DType::U8, &candle_core::Device::Cpu)
            .expect("valid tensor");
        let err = save_image(&gray, "/tmp/never-written.png").expect_err("must reject 1 channel");
        assert!(err.to_string().contains("3-channel"), "{err}");
    }
}
