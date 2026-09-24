//! The 6-layer post-LN BART decoder of SheetSage2, with learned positions (offset 2) and the
//! output projection tied to the token embedding.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, kv_cache::KvCache};

use crate::{EngineError, Result, SheetSage2Config, nn::{LayerNorm, attend}};

/// BartLearnedPositionalEmbedding keeps two unused rows in front.
const POSITION_OFFSET: usize = 2;

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, width: usize, heads: usize) -> Result<Self> {
        let lin = |name: &str| candle_nn::linear(width, width, vb.pp(name));
        Ok(Self { q: lin("q_proj")?, k: lin("k_proj")?, v: lin("v_proj")?, out: lin("out_proj")?, heads, head_dim: width / heads })
    }

    /// [b, t, width] -> [b, heads, t, head_dim]
    fn split(&self, proj: &Linear, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        Ok(proj.forward(x)?.reshape((b, t, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?)
    }

    fn merge(&self, attended: &Tensor) -> Result<Tensor> {
        let (b, _, t, _) = attended.dims4()?;
        Ok(self.out.forward(&attended.transpose(1, 2)?.reshape((b, t, ()))?)?)
    }
}

struct Layer {
    self_attn: Attention,
    self_norm: LayerNorm,
    cross_attn: Attention,
    cross_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

/// One window's encoder output, plus every layer's cross-attention keys and values so a
/// decode step never re-projects the 7500 frames.
pub struct Memory {
    hidden: Tensor,
    cross: Vec<(Tensor, Tensor)>,
}

impl Memory {
    pub fn frames(&self) -> usize {
        self.hidden.dim(1).unwrap_or(0)
    }

    /// The encoder_last_hidden_state as f32 [frames * hidden].
    pub fn features(&self) -> Result<Vec<f32>> {
        Ok(self.hidden.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?)
    }
}

/// Self-attention KV state of one token sequence.
pub struct DecoderCache {
    layers: Vec<KvCache>,
    len: usize,
}

impl DecoderCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct BartDecoder {
    embed: Embedding,
    positions: Tensor,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    lm_head: Linear,
    max_positions: usize,
    device: Device,
}

impl BartDecoder {
    /// `vb` is the SheetSage2 root: `token_embedding` and `decoder.*`.
    pub fn load(vb: VarBuilder, cfg: &SheetSage2Config) -> Result<Self> {
        let (width, eps) = (cfg.hidden_size, 1e-5);
        let dvb = vb.pp("decoder");
        let tokens = vb.get((cfg.vocab_size, width), "token_embedding.weight")?;
        let layers = (0..cfg.decoder_layers)
            .map(|i| {
                let lvb = dvb.pp("layers").pp(i);
                Ok(Layer {
                    self_attn: Attention::load(lvb.pp("self_attn"), width, cfg.num_attention_heads)?,
                    self_norm: LayerNorm::load(lvb.pp("self_attn_layer_norm"), width, eps)?,
                    cross_attn: Attention::load(lvb.pp("encoder_attn"), width, cfg.num_attention_heads)?,
                    cross_norm: LayerNorm::load(lvb.pp("encoder_attn_layer_norm"), width, eps)?,
                    fc1: candle_nn::linear(width, cfg.intermediate_size, lvb.pp("fc1"))?,
                    fc2: candle_nn::linear(cfg.intermediate_size, width, lvb.pp("fc2"))?,
                    final_norm: LayerNorm::load(lvb.pp("final_layer_norm"), width, eps)?,
                })
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            embed: Embedding::new(tokens.clone(), width),
            positions: dvb.get((cfg.max_output_seq_len + POSITION_OFFSET, width), "embed_positions.weight")?,
            embed_norm: LayerNorm::load(dvb.pp("layernorm_embedding"), width, eps)?,
            layers,
            lm_head: Linear::new(tokens, None),
            max_positions: cfg.max_output_seq_len,
            device: vb.device().clone(),
        })
    }

    pub fn new_cache(&self, capacity: usize) -> DecoderCache {
        DecoderCache { layers: (0..self.layers.len()).map(|_| KvCache::new(2, capacity.max(1))).collect(), len: 0 }
    }

    /// Wraps the projected encoder output [1, frames, width] and precomputes the cross K/V.
    pub fn memory(&self, hidden: Tensor) -> Result<Memory> {
        let cross = self
            .layers
            .iter()
            .map(|layer| {
                let attn = &layer.cross_attn;
                Ok((attn.split(&attn.k, &hidden)?, attn.split(&attn.v, &hidden)?))
            })
            .collect::<Result<_>>()?;
        Ok(Memory { hidden, cross })
    }

    /// Appends `tokens` to `cache` and writes the last position's logits.
    pub fn forward(&self, memory: &Memory, cache: &mut DecoderCache, tokens: &[u32], logits: &mut [f32]) -> Result<()> {
        let n = tokens.len();
        if n == 0 || logits.len() != self.vocab_size()? {
            return Err(EngineError::Invalid("decode needs tokens and a vocab-sized logits buffer".into()));
        }
        if cache.len + n > self.max_positions {
            return Err(EngineError::Invalid(format!("position {} is past the {} token decoder context", cache.len + n, self.max_positions)));
        }

        let ids = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let positions = self.positions.narrow(0, cache.len + POSITION_OFFSET, n)?.unsqueeze(0)?;
        let mut x = self.embed_norm.forward(&self.embed.forward(&ids)?.broadcast_add(&positions)?)?;

        for ((layer, kv), (cross_k, cross_v)) in self.layers.iter().zip(&mut cache.layers).zip(&memory.cross) {
            let attn = &layer.self_attn;
            let (k, v) = kv.append(&attn.split(&attn.k, &x)?, &attn.split(&attn.v, &x)?)?;
            let attended = attend(&attn.split(&attn.q, &x)?, &k, &v, true)?;
            x = layer.self_norm.forward(&(x + attn.merge(&attended)?)?)?;

            let attn = &layer.cross_attn;
            let attended = attend(&attn.split(&attn.q, &x)?, cross_k, cross_v, false)?;
            x = layer.cross_norm.forward(&(x + attn.merge(&attended)?)?)?;

            let ff = layer.fc2.forward(&layer.fc1.forward(&x)?.gelu_erf()?)?;
            x = layer.final_norm.forward(&(x + ff)?)?;
        }
        cache.len += n;

        let values = self.lm_head.forward(&x.narrow(1, n - 1, 1)?)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        logits.copy_from_slice(&values);
        Ok(())
    }

    pub fn vocab_size(&self) -> Result<usize> {
        Ok(self.lm_head.weight().dim(0)?)
    }
}
