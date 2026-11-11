use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub fn tiled_decode(
    latent: &Tensor,
    tile_size: usize,
    overlap: usize,
    output_h: usize,
    output_w: usize,
    decode: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let (_batch, _channels, h, w) = latent.dims4()?;

    anyhow::ensure!(
        tile_size > overlap,
        "--vae-tile-size ({tile_size}) must be greater than --vae-tile-overlap ({overlap})"
    );
    anyhow::ensure!(
        tile_size <= h && tile_size <= w,
        "--vae-tile-size ({tile_size}) must fit within latent ({h}x{w})"
    );

    let scale_h = output_h / h;
    let scale_w = output_w / w;
    anyhow::ensure!(
        output_h == h * scale_h,
        "output_h {output_h} must be an integer multiple of latent h {h}"
    );
    anyhow::ensure!(
        output_w == w * scale_w,
        "output_w {output_w} must be an integer multiple of latent w {w}"
    );

    let device = latent.device();

    let stride = tile_size.saturating_sub(overlap);
    let overlap_out_h = overlap * scale_h;
    let overlap_out_w = overlap * scale_w;

    let mut output = Tensor::zeros((1, 3, output_h, output_w), DType::F32, device)?;
    // Sum of feather weights per pixel. Overlaps blend with complementary
    // ramps, so without normalization the result darkens wherever the mask
    // sum is below 1.
    let mut weight_sum = Tensor::zeros((1, 1, output_h, output_w), DType::F32, device)?;

    // Feather masks repeat across the grid: every interior tile shares one
    // mask, and edge/corner tiles repeat along rows and columns. Cache them
    // by (out_h, out_w, feather bits) to avoid recomputing the CPU loop and
    // re-uploading to the device for every tile. Overlap sizes are constant
    // within a single call, so they need not be part of the key.
    let mut weight_cache: std::collections::HashMap<(usize, usize, u8), Tensor> =
        std::collections::HashMap::new();

    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + tile_size).min(h);
        let ty_h = y1 - y0;

        let feather_top = y0 > 0;
        let feather_bottom = y1 < h;

        let mut x0 = 0usize;
        while x0 < w {
            let x1 = (x0 + tile_size).min(w);
            let tx_w = x1 - x0;

            let feather_left = x0 > 0;
            let feather_right = x1 < w;

            let tile_latent = latent.narrow(2, y0, ty_h)?.narrow(3, x0, tx_w)?;
            let tile_decoded = decode(&tile_latent)?;

            let out_y0 = y0 * scale_h;
            let out_x0 = x0 * scale_w;
            let out_h = ty_h * scale_h;
            let out_w = tx_w * scale_w;

            let feather_bits = (feather_left as u8)
                | (feather_top as u8) << 1
                | (feather_right as u8) << 2
                | (feather_bottom as u8) << 3;
            let cache_key = (out_h, out_w, feather_bits);
            let weight = match weight_cache.get(&cache_key) {
                Some(w) => w.clone(),
                None => {
                    let w = tile_weight(
                        &TileParams {
                            h: out_h,
                            w: out_w,
                            overlap_h: overlap_out_h,
                            overlap_w: overlap_out_w,
                            feather_left,
                            feather_top,
                            feather_right,
                            feather_bottom,
                        },
                        device,
                    )?;
                    weight_cache.insert(cache_key, w.clone());
                    w
                }
            };

            let weighted = tile_decoded.broadcast_mul(&weight)?;
            let region = output.narrow(2, out_y0, out_h)?.narrow(3, out_x0, out_w)?;
            let updated = (region + weighted)?;
            output = output.slice_assign(
                &[0..1, 0..3, out_y0..out_y0 + out_h, out_x0..out_x0 + out_w],
                &updated,
            )?;
            let wregion = weight_sum
                .narrow(2, out_y0, out_h)?
                .narrow(3, out_x0, out_w)?;
            weight_sum = weight_sum.slice_assign(
                &[0..1, 0..1, out_y0..out_y0 + out_h, out_x0..out_x0 + out_w],
                &(wregion + weight)?,
            )?;

            x0 += stride;
        }
        y0 += stride;
    }

    // Every output pixel is covered by at least one tile whose feather ramp
    // is strictly positive (see `tile_weight`), so the divisor never vanishes.
    Ok(output.broadcast_div(&weight_sum)?)
}

struct TileParams {
    h: usize,
    w: usize,
    overlap_h: usize,
    overlap_w: usize,
    feather_left: bool,
    feather_top: bool,
    feather_right: bool,
    feather_bottom: bool,
}

fn tile_weight(params: &TileParams, device: &Device) -> Result<Tensor> {
    let h = params.h;
    let w = params.w;
    let mut weights = vec![1.0f32; h * w];

    let eff_overlap_h = params.overlap_h.min(h / 2);
    let eff_overlap_w = params.overlap_w.min(w / 2);

    // Ramps use (x+1)/(overlap+1) instead of x/overlap: the endpoints stay
    // strictly positive, so neighbouring tiles never both contribute zero
    // weight to the same pixel. For full-size tiles the ramps are exact
    // complements (summing to 1); truncated edge tiles still leave every
    // pixel covered with positive total weight, and `tiled_decode` divides
    // by the accumulated sum to remove any residual darkening.
    if params.feather_left {
        for y in 0..h {
            for x in 0..eff_overlap_w {
                let t = (x + 1) as f32 / (eff_overlap_w + 1) as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_right {
        for y in 0..h {
            for xi in 0..eff_overlap_w {
                let x = w - 1 - xi;
                let t = (xi + 1) as f32 / (eff_overlap_w + 1) as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_top {
        for y in 0..eff_overlap_h {
            for x in 0..w {
                let t = (y + 1) as f32 / (eff_overlap_h + 1) as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_bottom {
        for yi in 0..eff_overlap_h {
            let y = h - 1 - yi;
            for x in 0..w {
                let t = (yi + 1) as f32 / (eff_overlap_h + 1) as f32;
                weights[y * w + x] *= t;
            }
        }
    }

    let mask = Tensor::from_slice(&weights, (h, w), device)?;
    Ok(mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With an identity "decoder", a constant latent must decode to the same
    /// constant in every pixel — including tile seams, edges and corners.
    /// Before weight normalization the feather ramps darkened overlaps
    /// (and produced zero-weight seam columns), so this test failed.
    #[test]
    fn constant_latent_stays_constant() -> Result<()> {
        let device = Device::Cpu;
        let cases: &[(usize, usize, usize, usize)] = &[
            // (latent_h, latent_w, tile_size, overlap)
            (8, 8, 4, 2),
            (12, 10, 5, 2),
            (6, 6, 6, 1),
            (16, 16, 6, 4),
            (7, 13, 4, 1),
        ];
        for &(h, w, tile, overlap) in cases {
            let value = 0.7f32;
            let latent = Tensor::full(value, (1, 3, h, w), &device)?;
            let out = tiled_decode(&latent, tile, overlap, h, w, |t| Ok(t.clone()))?;
            anyhow::ensure!(
                out.dims4()? == (1, 3, h, w),
                "unexpected output dims for case {h}x{w} tile={tile} overlap={overlap}"
            );
            let data = out.flatten_all()?.to_vec1::<f32>()?;
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    (v - value).abs() < 1e-4,
                    "case {h}x{w} tile={tile} overlap={overlap}: pixel {i} = {v}, want {value}"
                );
            }
        }
        Ok(())
    }
}
