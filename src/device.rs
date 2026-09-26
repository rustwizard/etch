use candle_core::Device;

// Without this guard, enabling both features would silently prefer Metal due
// to the cfg ordering below, even on a CUDA-only machine.
#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!(
    "features `metal` and `cuda` are mutually exclusive: pick exactly one GPU backend \
     (`--features metal` on Apple Silicon, `--features cuda` on NVIDIA)"
);

pub fn pick_device(cpu: bool) -> Device {
    if cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).unwrap_or_else(|e| {
            tracing::warn!("Metal init failed: {e}. Falling back to CPU.");
            Device::Cpu
        })
    }
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        Device::new_cuda(0).unwrap_or_else(|e| {
            tracing::warn!("CUDA init failed: {e}. Falling back to CPU.");
            Device::Cpu
        })
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Device::Cpu
    }
}
