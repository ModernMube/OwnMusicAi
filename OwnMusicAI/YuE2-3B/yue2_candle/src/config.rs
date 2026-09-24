use std::path::Path;

use serde::Deserialize;

use crate::Result;

/// The inference fields of YuE2-3B's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Yue2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub latent_dim: usize,
    pub max_latent_frames: usize,
    pub timestep_shift: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub out_channels: usize,
    pub channels: usize,
    pub latent_dim: usize,
    pub c_mults: Vec<usize>,
    pub strides: Vec<usize>,
    pub use_snake: bool,
    #[serde(default)]
    pub final_tanh: bool,
}

/// YuE2-Vae / YuE2-Vae-legacy, decoder side only.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub decoder_config: DecoderConfig,
    pub sample_rate: usize,
    pub downsampling_ratio: usize,
    pub latent_dim: usize,
}

pub(crate) fn read<T: for<'de> Deserialize<'de>>(dir: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?)
}
