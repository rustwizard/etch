use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::stable_diffusion::schedulers::Scheduler;

// ScaledLinear beta schedule used by both SDXL Karras schedulers.
// Returns (all_sigmas, sigma_max, sigma_min) for n_steps inference steps.
pub(crate) fn build_sdxl_sigmas(n_steps: usize) -> (Vec<f64>, f64, f64) {
    const BETA_START: f64 = 0.00085;
    const BETA_END: f64 = 0.012;
    const TRAIN_STEPS: usize = 1000;
    const STEPS_OFFSET: usize = 1;

    let mut cumprod = 1.0f64;
    let all_sigmas: Vec<f64> = (0..TRAIN_STEPS)
        .map(|i| {
            let t = i as f64 / (TRAIN_STEPS - 1) as f64;
            let b = BETA_START.sqrt() + t * (BETA_END.sqrt() - BETA_START.sqrt());
            cumprod *= 1.0 - b * b;
            ((1.0 - cumprod) / cumprod).sqrt()
        })
        .collect();

    let step_ratio = TRAIN_STEPS / n_steps;
    // Clamp indices: for n_steps=1 the raw index (step_ratio + STEPS_OFFSET)
    // is 1001, past the end of the 1000-entry table. The schedule degenerates
    // (sigma_min = sigma_max) but must not panic and must stay ordered.
    let min_idx = (step_ratio + STEPS_OFFSET).min(TRAIN_STEPS - 1);
    let max_idx = ((n_steps - 1) * step_ratio + STEPS_OFFSET)
        .min(TRAIN_STEPS - 1)
        .max(min_idx);
    let sigma_max = all_sigmas[max_idx];
    let sigma_min = all_sigmas[min_idx];
    (all_sigmas, sigma_max, sigma_min)
}

// Map a Karras sigma back to the nearest discrete timestep in the original
// schedule. all_sigmas is monotone-increasing (sigma grows with timestep).
fn sigma_to_t(sigma: f64, all_sigmas: &[f64]) -> usize {
    let idx = all_sigmas.partition_point(|&s| s < sigma);
    if idx == 0 {
        return 0;
    }
    if idx >= all_sigmas.len() {
        return all_sigmas.len() - 1;
    }
    if (all_sigmas[idx - 1] - sigma).abs() <= (all_sigmas[idx] - sigma).abs() {
        idx - 1
    } else {
        idx
    }
}

// Build the Karras sigma schedule and the matching UNet timesteps.
pub(crate) fn build_karras_schedule(
    n_steps: usize,
    all_sigmas: &[f64],
    sigma_max: f64,
    sigma_min: f64,
) -> (Vec<f64>, Vec<usize>) {
    const RHO: f64 = 7.0;
    let min_inv_rho = sigma_min.powf(1.0 / RHO);
    let max_inv_rho = sigma_max.powf(1.0 / RHO);
    let mut sigmas: Vec<f64> = (0..n_steps)
        .map(|i| {
            let u = i as f64 / (n_steps - 1).max(1) as f64;
            (max_inv_rho + u * (min_inv_rho - max_inv_rho)).powf(RHO)
        })
        .collect();
    sigmas.push(0.0);
    // For each Karras sigma find the UNet timestep with the matching noise level.
    let timesteps: Vec<usize> = sigmas[..n_steps]
        .iter()
        .map(|&s| sigma_to_t(s, all_sigmas))
        .collect();
    (sigmas, timesteps)
}

fn timestep_index(timesteps: &[usize], timestep: usize) -> candle_core::Result<usize> {
    timesteps
        .iter()
        .position(|&t| t == timestep)
        .ok_or_else(|| candle_core::Error::Msg(format!("timestep {timestep} not in schedule")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Karras sigma schedule wrapped around EulerA steps
// ─────────────────────────────────────────────────────────────────────────────

pub struct KarrasEulerAScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    init_noise_sigma: f64,
}

impl KarrasEulerAScheduler {
    pub fn new(n_steps: usize) -> Result<Self> {
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
        let (sigmas, timesteps) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
        let init_noise_sigma = (sigma_max * sigma_max + 1.0).sqrt();
        Ok(Self {
            sigmas,
            timesteps,
            init_noise_sigma,
        })
    }
}

impl Scheduler for KarrasEulerAScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        sample / (self.sigmas[i] * self.sigmas[i] + 1.0).sqrt()
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        let sigma_from = self.sigmas[i];
        let sigma_to = self.sigmas[i + 1];

        // Predicted denoised sample (epsilon prediction)
        let pred_x0 = (sample - (model_output * sigma_from)?)?;

        // Stochastic noise split: sigma_up^2 + sigma_down^2 = sigma_to^2
        let sigma_up = (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).sqrt();

        let derivative = ((sample - pred_x0)? / sigma_from)?;
        let prev_sample = (sample + (derivative * (sigma_down - sigma_from))?)?;
        let noise = prev_sample.randn_like(0.0, 1.0)?;
        prev_sample + (noise * sigma_up)?
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: Tensor,
        timestep: usize,
    ) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        original + (noise * self.sigmas[i])?
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DPM++ 2M Karras scheduler
// Pure-sigma parameterisation: x = x₀ + σ·ε (α ≡ 1).
// scale_model_input normalises to unit variance, matching the EulerA convention.
// ─────────────────────────────────────────────────────────────────────────────

pub struct Dpm2mKarrasScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    prev_denoised: Option<Tensor>,
}

impl Dpm2mKarrasScheduler {
    pub fn new(n_steps: usize) -> Result<Self> {
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
        let (sigmas, timesteps) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
        Ok(Self {
            sigmas,
            timesteps,
            prev_denoised: None,
        })
    }
}

impl Scheduler for Dpm2mKarrasScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        let sigma = self.sigmas[i];
        sample / (sigma * sigma + 1.0).sqrt()
    }

    fn init_noise_sigma(&self) -> f64 {
        let s = self.sigmas[0];
        (s * s + 1.0).sqrt()
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        let sigma_from = self.sigmas[i];
        let sigma_to = self.sigmas[i + 1];

        // x₀ estimate: D₀ = x - σ·ε  (pure-sigma, α ≡ 1)
        let denoised = (sample - (model_output * sigma_from)?)?;

        let x_next = if sigma_to == 0.0 {
            denoised.clone()
        } else {
            // h = ln(σ_from/σ_to) > 0, ratio = σ_to/σ_from = exp(-h)
            let h = sigma_from.ln() - sigma_to.ln();
            let ratio = sigma_to / sigma_from;

            match self.prev_denoised.take() {
                None => {
                    // 1st order (exact solution to the sigma-space ODE with constant D₀)
                    ((sample * ratio)? + (&denoised * (1.0 - ratio))?)?
                }
                Some(prev_d) => {
                    // 2nd order DPM++ 2M midpoint correction
                    // D₁ = (D₀ - D₀_prev) / r,  denoised_d = D₀ + ½·D₁
                    let h_last = self.sigmas[i - 1].ln() - sigma_from.ln();
                    let r = h_last / h;
                    let c1 = 1.0 + 1.0 / (2.0 * r);
                    let c2 = 1.0 / (2.0 * r);
                    let denoised_d = ((&denoised * c1)? - (&prev_d * c2)?)?;
                    ((sample * ratio)? + (denoised_d * (1.0 - ratio))?)?
                }
            }
        };

        self.prev_denoised = Some(denoised);
        Ok(x_next)
    }

    fn add_noise(
        &self,
        original: &Tensor,
        noise: Tensor,
        timestep: usize,
    ) -> candle_core::Result<Tensor> {
        let i = timestep_index(&self.timesteps, timestep)?;
        original + (noise * self.sigmas[i])?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_does_not_panic() {
        // n_steps=1 pushes the min index past the 1000-entry table; clamping
        // must keep it in bounds and the schedule ordered.
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(1);
        assert!(sigma_max >= sigma_min);
        let (sigmas, timesteps) = build_karras_schedule(1, &all_sigmas, sigma_max, sigma_min);
        assert_eq!(sigmas.len(), 2); // one step + terminal 0.0
        assert_eq!(sigmas[1], 0.0);
        assert_eq!(timesteps.len(), 1);
    }

    #[test]
    fn karras_sigmas_decrease_to_zero() {
        for n_steps in [2, 4, 20, 30, 50, 1000] {
            let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
            let (sigmas, _) = build_karras_schedule(n_steps, &all_sigmas, sigma_max, sigma_min);
            assert_eq!(sigmas.len(), n_steps + 1, "n_steps={n_steps}");
            assert_eq!(*sigmas.last().expect("non-empty"), 0.0);
            assert!(sigmas[0] > 0.0, "n_steps={n_steps}");
            for w in sigmas.windows(2) {
                // Non-strict: for small n_steps (e.g. 2) sigma_max == sigma_min
                // by construction of the timestep spacing, so the leading
                // sigmas can be equal. Strict decrease holds for typical counts.
                assert!(
                    w[0] >= w[1],
                    "sigmas must not increase (n_steps={n_steps}): {} < {}",
                    w[0],
                    w[1]
                );
            }
            if n_steps >= 4 {
                for w in sigmas[..n_steps].windows(2) {
                    assert!(w[0] > w[1], "strict decrease expected (n_steps={n_steps})");
                }
            }
        }
    }

    #[test]
    fn sigma_bounds_are_ordered_and_positive() {
        for n_steps in [1, 2, 20, 50] {
            let (_, sigma_max, sigma_min) = build_sdxl_sigmas(n_steps);
            assert!(sigma_min > 0.0, "n_steps={n_steps}");
            assert!(sigma_max >= sigma_min, "n_steps={n_steps}");
        }
    }

    #[test]
    fn timesteps_map_into_training_range() {
        let (all_sigmas, sigma_max, sigma_min) = build_sdxl_sigmas(20);
        let (_, timesteps) = build_karras_schedule(20, &all_sigmas, sigma_max, sigma_min);
        for t in timesteps {
            assert!(t < 1000, "timestep {t} outside training range");
        }
    }

    #[test]
    fn sigma_to_t_picks_nearest() {
        let grid = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(sigma_to_t(0.4, &grid), 0);
        assert_eq!(sigma_to_t(0.6, &grid), 1);
        assert_eq!(sigma_to_t(2.5, &grid), 2); // tie goes to the lower index
        assert_eq!(sigma_to_t(99.0, &grid), 3);
        assert_eq!(sigma_to_t(-99.0, &grid), 0);
    }
}
