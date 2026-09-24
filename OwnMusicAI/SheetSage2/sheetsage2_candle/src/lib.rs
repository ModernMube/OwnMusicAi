//! SheetSage2 compute primitives on Candle: log-mel + MERT-v2 encoder for a 300 s window, and a
//! cached BART decoder step. Policy (audio loading, windowing, the event grammar, greedy
//! decoding, stitching, ABC/MIDI export) lives in the host - see the C# library and [`ffi`].

mod config;
mod decoder;
mod error;
pub mod ffi;
mod mel;
mod mert;
mod model;
mod nn;

use candle_core::{DType, Device};

pub use config::{MertConfig, SheetSage2Config, WeightsFormat};
pub use decoder::{BartDecoder, DecoderCache, Memory};
pub use error::{EngineError, Result};
pub use mel::MelFrontend;
pub use mert::MertEncoder;
pub use model::SheetSage2;

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

    /// BF16 is what the reference runs on CUDA. Metal stays F32: this model is small, and on an
    /// M1 Pro f16 came out slower (23 s vs 13 s per window, 40 vs 132 decode steps/s) and noisier.
    pub fn default_dtype(self) -> DType {
        match self {
            Backend::Cpu | Backend::Metal => DType::F32,
            Backend::Cuda => DType::BF16,
        }
    }
}
