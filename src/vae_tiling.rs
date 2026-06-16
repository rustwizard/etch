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

            let weight = tile_weight(
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

            let weighted = tile_decoded.broadcast_mul(&weight)?;
            let region = output.narrow(2, out_y0, out_h)?.narrow(3, out_x0, out_w)?;
            let updated = (region + weighted)?;
            output = output.slice_assign(
                &[0..1, 0..3, out_y0..out_y0 + out_h, out_x0..out_x0 + out_w],
                &updated,
            )?;

            x0 += stride;
        }
        y0 += stride;
    }

    Ok(output)
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

    if params.feather_left {
        for y in 0..h {
            for x in 0..eff_overlap_w {
                let t = x as f32 / eff_overlap_w as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_right {
        for y in 0..h {
            for xi in 0..eff_overlap_w {
                let x = w - 1 - xi;
                let t = xi as f32 / eff_overlap_w as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_top {
        for y in 0..eff_overlap_h {
            for x in 0..w {
                let t = y as f32 / eff_overlap_h as f32;
                weights[y * w + x] *= t;
            }
        }
    }
    if params.feather_bottom {
        for yi in 0..eff_overlap_h {
            let y = h - 1 - yi;
            for x in 0..w {
                let t = yi as f32 / eff_overlap_h as f32;
                weights[y * w + x] *= t;
            }
        }
    }

    let mask = Tensor::from_slice(&weights, (h, w), device)?;
    Ok(mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?)
}
