//! YuE2-3B compute primitives on Candle: AR step with KV cache, NAR flow-matching velocity,
//! and the Oobleck VAE decoder. Everything that is *policy* (prompting, sampling, CFG, the ODE
//! solver, chunking, VAE tiling) lives in the host - see the C# app and [`ffi`].

mod config;
mod error;
pub mod ffi;
mod model;
mod vae;

use candle_core::{DType, Device};

pub use config::{VaeConfig, Yue2Config};
pub use error::{EngineError, Result};
pub use model::{Cache, Yue2Model};
pub use vae::VaeDecoder;

/// Where the tensors live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    pub fn device(self) -> Result<Device> {
        Ok(match self {
            Backend::Cpu => Device::Cpu,
            Backend::Metal => Device::new_metal(0)?,
            Backend::Cuda => Device::new_cuda(0)?,
        })
    }

    /// Precision the transformer runs in when the host doesn't care: BF16 is native on CUDA,
    /// Metal gets F16 (M1 has no bf16), CPU stays F32.
    pub fn default_dtype(self) -> DType {
        match self {
            Backend::Cpu => DType::F32,
            Backend::Metal => DType::F16,
            Backend::Cuda => DType::BF16,
        }
    }
}
