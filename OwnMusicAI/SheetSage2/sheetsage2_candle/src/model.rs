use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    BartDecoder, DecoderCache, EngineError, MelFrontend, Memory, MertEncoder, Result, SheetSage2Config,
    config::{self, WeightsFormat},
    mert::Adapters,
};

/// Loaded SheetSage2: log-mel frontend, merged MERT-v2 encoder, BART decoder.
pub struct SheetSage2 {
    cfg: SheetSage2Config,
    mel: MelFrontend,
    encoder: MertEncoder,
    decoder: BartDecoder,
    dtype: DType,
}

impl SheetSage2 {
    /// `mert_dir` is the MERT-v2-FullSong snapshot; only an adapter-format checkpoint needs it.
    pub fn load(model_dir: &Path, mert_dir: Option<&Path>, device: &Device, dtype: DType) -> Result<Self> {
        let cfg: SheetSage2Config = config::read(model_dir)?;
        let backbone = &cfg.backbone_config;
        if cfg.sampling_rate != backbone.sampling_rate {
            return Err(EngineError::Invalid("processor and encoder sampling rates differ".into()));
        }

        // SAFETY: checkpoints are mmapped read-only and nobody rewrites them while we run.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, device)? };
        let (mert, adapters) = match cfg.weights_format {
            WeightsFormat::Merged => (vb.pp("encoder"), None),
            WeightsFormat::Adapter => {
                let dir = mert_dir.ok_or_else(|| EngineError::Invalid("adapter checkpoint needs the MERT-v2-FullSong folder".into()))?;
                // SAFETY: same as above.
                let parent = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)? };
                let adapters = Adapters { vb: vb.pp("adapter"), rank: cfg.lora_rank, scale: cfg.lora_alpha / cfg.lora_rank as f64 };
                (parent, Some(adapters))
            }
        };

        Ok(Self {
            mel: MelFrontend::load(mert.pp("feature_extractor"), backbone)?,
            encoder: MertEncoder::load(mert, adapters, vb.clone(), backbone, cfg.hidden_size)?,
            decoder: BartDecoder::load(vb, &cfg)?,
            dtype,
            cfg,
        })
    }

    pub fn config(&self) -> &SheetSage2Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn mel(&self) -> &MelFrontend {
        &self.mel
    }

    /// Encodes one mono 24 kHz window. Shorter audio is zero-padded to the full 300 s window,
    /// exactly like `_prepare_audio`: the encoder always attends to the whole window.
    pub fn encode(&self, samples: &[f32]) -> Result<Memory> {
        let window = self.cfg.window_samples();
        let minimum = self.cfg.backbone_config.n_fft / 2 + 1;
        if samples.len() < minimum || samples.len() > window {
            return Err(EngineError::Invalid(format!("a window takes {minimum}..={window} samples, got {}", samples.len())));
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(EngineError::Invalid("audio must contain only finite samples".into()));
        }
        let mut padded = samples.to_vec();
        padded.resize(window, 0.0);
        let (mel, frames) = self.mel.compute(&padded)?;
        self.decoder.memory(self.encoder.forward(&mel, frames)?)
    }

    pub fn new_cache(&self, capacity: usize) -> DecoderCache {
        self.decoder.new_cache(capacity)
    }

    /// Appends tokens to `cache`; last-position logits (vocab sized) land in `logits`.
    pub fn decode(&self, memory: &Memory, cache: &mut DecoderCache, tokens: &[u32], logits: &mut [f32]) -> Result<()> {
        self.decoder.forward(memory, cache, tokens, logits)
    }
}
