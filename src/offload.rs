use anyhow::Result;
use candle_core::Device;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum OffloadTarget {
    Cpu,
    Disk,
}

pub trait Offloadable {
    fn to_compute(&mut self, device: &Device) -> Result<()>;
    fn to_offload(&mut self, device: &Device) -> Result<()>;
}
