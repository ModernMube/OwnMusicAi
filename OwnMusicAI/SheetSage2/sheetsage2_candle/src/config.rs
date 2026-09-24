use std::path::Path;

use serde::Deserialize;

use crate::Result;

/// MERT-v2 backbone, as embedded in SheetSage2's config.json (`backbone_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct MertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub subsampling_channels: Vec<usize>,
    pub subsampling_depths: Vec<usize>,
    pub conv_depthwise_kernel_size: usize,
    pub rotary_embedding_base: f64,
    pub layer_norm_eps: f64,
    pub subsampling_layer_norm_eps: f64,
}

impl MertConfig {
    /// Waveform samples per encoder frame (hop * 4, the subsampler halves twice).
    pub fn samples_per_frame(&self) -> usize {
        self.hop_length * 4
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightsFormat {
    /// LoRA adapters only - the MERT-v2 parent is loaded next to it and merged.
    Adapter,
    /// Self-contained `save_pretrained` snapshot, encoder under `encoder.`.
    Merged,
}

/// The inference fields of SheetSage2's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct SheetSage2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub decoder_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub input_audio_length: f64,
    pub max_output_seq_len: usize,
    pub time_hz: usize,
    pub sampling_rate: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub weights_format: WeightsFormat,
    pub backbone_config: MertConfig,
}

impl SheetSage2Config {
    /// One model window in samples (300 s at 24 kHz), rounded up to a whole encoder frame.
    pub fn window_samples(&self) -> usize {
        let stride = self.backbone_config.samples_per_frame();
        let samples = (self.input_audio_length * self.sampling_rate as f64).round() as usize;
        samples.div_ceil(stride) * stride
    }
}

pub(crate) fn read<T: for<'de> Deserialize<'de>>(dir: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?)
}
