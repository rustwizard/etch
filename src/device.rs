use candle_core::Device;

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
