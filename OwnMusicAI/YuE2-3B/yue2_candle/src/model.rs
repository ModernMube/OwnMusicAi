use std::path::Path;

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, kv_cache::KvCache, ops, rotary_emb::rope};

use crate::{EngineError, Result, Yue2Config, config};

/// Query rows per attention block on the non-Metal path, keeps the score matrix bounded.
const QUERY_BLOCK: usize = 512;

/// RMSNorm that squares in f32 - the residual stream goes past 400 late in the stack,
/// which overflows fp16 once squared. Same upcast as the reference.
struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?.to_dtype(DType::F32)?, eps: eps as f32 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(ops::rms_norm(&x.to_dtype(DType::F32)?, &self.weight, self.eps)?.to_dtype(x.dtype())?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        Ok(self.down.forward(&gated)?)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// Projects, QK-normalizes and ropes; q is [b, heads, t, hd], k/v [b, kv_heads, t, hd].
    fn qkv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;
        let split = |proj: &Linear, heads: usize| proj.forward(x)?.reshape((b, t, heads, self.head_dim));
        let q = self.q_norm.forward(&split(&self.q, self.heads)?)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_norm.forward(&split(&self.k, self.kv_heads)?)?.transpose(1, 2)?.contiguous()?;
        let v = split(&self.v, self.kv_heads)?.transpose(1, 2)?.contiguous()?;
        Ok((rope(&q, cos, sin)?, rope(&k, cos, sin)?, v))
    }

    fn out(&self, attended: &Tensor) -> Result<Tensor> {
        let (b, _, t, _) = attended.dims4()?;
        Ok(self.o.forward(&attended.transpose(1, 2)?.reshape((b, t, ()))?)?)
    }
}

/// One of the two MoT paths inside a layer (AR or NAR weights).
struct Block {
    input_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: &VarBuilder, cfg: &Yue2Config, [input_norm, attn, mlp_norm, mlp]: [&str; 4]) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let lin = |name: &str, i, o| candle_nn::linear_no_bias(i, o, vb.pp(attn).pp(name));
        let mlp_vb = vb.pp(mlp);
        Ok(Self {
            input_norm: RmsNorm::load(vb.pp(input_norm), h, cfg.rms_norm_eps)?,
            attn: Attention {
                q: lin("q_proj", h, cfg.num_attention_heads * hd)?,
                k: lin("k_proj", h, cfg.num_key_value_heads * hd)?,
                v: lin("v_proj", h, cfg.num_key_value_heads * hd)?,
                o: lin("o_proj", cfg.num_attention_heads * hd, h)?,
                q_norm: RmsNorm::load(vb.pp(attn).pp("q_norm"), hd, cfg.rms_norm_eps)?,
                k_norm: RmsNorm::load(vb.pp(attn).pp("k_norm"), hd, cfg.rms_norm_eps)?,
                heads: cfg.num_attention_heads,
                kv_heads: cfg.num_key_value_heads,
                head_dim: hd,
            },
            mlp_norm: RmsNorm::load(vb.pp(mlp_norm), h, cfg.rms_norm_eps)?,
            mlp: Mlp {
                gate: candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp_vb.pp("gate_proj"))?,
                up: candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp_vb.pp("up_proj"))?,
                down: candle_nn::linear_no_bias(cfg.intermediate_size, h, mlp_vb.pp("down_proj"))?,
            },
        })
    }
}

struct Layer {
    ar: Block,
    nar: Block,
}

/// Per-sequence KV state: one per AR branch (positive / CFG negative), one per acoustic chunk.
pub struct Cache {
    layers: Vec<KvCache>,
    len: usize,
}

impl Cache {
    /// `capacity` is only the first allocation; it grows by the same amount when outrun.
    pub fn new(layers: usize, capacity: usize) -> Self {
        Self { layers: (0..layers).map(|_| KvCache::new(2, capacity.max(1))).collect(), len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct Yue2Model {
    cfg: Yue2Config,
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    llm2vae: Linear,
    vae2llm: Linear,
    time_in: Linear,
    time_out: Linear,
    cos: Tensor,
    sin: Tensor,
    pos_div: Tensor,
    dtype: DType,
    device: Device,
}

impl Yue2Model {
    pub fn load(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg: Yue2Config = config::read(dir)?;
        // SAFETY: the checkpoint is mmapped read-only; nobody rewrites model.safetensors under us.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)? };
        let h = cfg.hidden_size;

        let layers_vb = vb.pp("model.layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let lvb = layers_vb.pp(i);
                Ok(Layer {
                    ar: Block::load(&lvb, &cfg, ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"])?,
                    nar: Block::load(&lvb, &cfg, ["nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp"])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let half = cfg.head_dim / 2;
        let inv_freq: Vec<f32> =
            (0..half).map(|i| (1.0 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.head_dim as f64)) as f32).collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?.to_dtype(DType::F32)?;
        let angles = positions.unsqueeze(1)?.matmul(&inv_freq)?;

        let pos_step = -(10000f64.ln() / h as f64);
        let pos_div: Vec<f32> = (0..h).step_by(2).map(|i| (i as f32 * pos_step as f32).exp()).collect();

        Ok(Self {
            embed: candle_nn::embedding(cfg.vocab_size, h, vb.pp("model.embed_tokens"))?,
            layers,
            norm: RmsNorm::load(vb.pp("model.norm"), h, cfg.rms_norm_eps)?,
            lm_head: candle_nn::linear_no_bias(h, cfg.vocab_size, vb.pp("lm_head"))?,
            llm2vae: candle_nn::linear(h, cfg.latent_dim, vb.pp("llm2vae"))?,
            vae2llm: candle_nn::linear(cfg.latent_dim, h, vb.pp("vae2llm"))?,
            time_in: candle_nn::linear(256, h, vb.pp("time_embedder.mlp.0"))?,
            time_out: candle_nn::linear(h, h, vb.pp("time_embedder.mlp.2"))?,
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
            pos_div: Tensor::from_vec(pos_div, (1, h / 2), device)?,
            dtype,
            device: device.clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &Yue2Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn new_cache(&self, capacity: usize) -> Cache {
        Cache::new(self.cfg.num_hidden_layers, capacity)
    }

    fn rope_at(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        if start + len > self.cfg.max_position_embeddings {
            return Err(EngineError::Invalid(format!(
                "position {} is past the {} token context",
                start + len,
                self.cfg.max_position_embeddings
            )));
        }
        Ok((self.cos.narrow(0, start, len)?, self.sin.narrow(0, start, len)?))
    }

    /// Feeds `tokens` through the AR path (growing `cache`) and writes the last position's
    /// logits into `logits`. Prefill in several calls if you want - positions continue.
    pub fn forward(&self, cache: &mut Cache, tokens: &[u32], logits: &mut [f32]) -> Result<()> {
        if tokens.is_empty() || logits.len() != self.cfg.vocab_size {
            return Err(EngineError::Invalid("need tokens and a vocab-sized logits buffer".into()));
        }
        let n = tokens.len();
        let (cos, sin) = self.rope_at(cache.len, n)?;
        let mut x = self.embed.forward(&Tensor::new(tokens, &self.device)?.unsqueeze(0)?)?;

        for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
            let block = &layer.ar;
            let (q, k, v) = block.attn.qkv(&block.input_norm.forward(&x)?, &cos, &sin)?;
            let (k, v) = kv.append(&k, &v)?;
            x = (x + block.attn.out(&attend(&q, &k, &v, true)?)?)?;
            x = (&x + block.mlp.forward(&block.mlp_norm.forward(&x)?)?)?;
        }
        cache.len += n;

        let last = self.norm.forward(&x.narrow(1, n - 1, 1)?)?;
        let values = self.lm_head.forward(&last)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        logits.copy_from_slice(&values);
        Ok(())
    }

    /// v(x_t, t) for one acoustic chunk whose tokens (prefix + codec + MUSIC_END) were fed into
    /// `prefix` with [`Self::forward`]. `latents` is [frames * 64] row-major, `t_logit` is
    /// clamp(logit(t), -20, 20) as yue2/nar.py passes it.
    pub fn velocity(&self, prefix: &Cache, latents: &[f32], t_logit: f32, velocity: &mut [f32]) -> Result<()> {
        let dim = self.cfg.latent_dim;
        if prefix.is_empty() || latents.is_empty() || latents.len() % dim != 0 || velocity.len() != latents.len() {
            return Err(EngineError::Invalid("velocity needs a filled prefix cache and matching [frames*64] buffers".into()));
        }
        let frames = latents.len() / dim;
        let count = frames + 2;
        let (cos, sin) = self.rope_at(prefix.len, count)?;

        let state = Tensor::from_slice(latents, (1, frames, dim), &self.device)?.pad_with_zeros(1, 1, 1)?;
        let mut x = self.vae2llm.forward(&state.to_dtype(self.dtype)?)?;
        x = x.broadcast_add(&self.time_embedding(t_logit)?)?;
        x = x.broadcast_add(&self.latent_positions(count)?)?;

        for (layer, kv) in self.layers.iter().zip(&prefix.layers) {
            let block = &layer.nar;
            let (q, k, v) = block.attn.qkv(&block.input_norm.forward(&x)?, &cos, &sin)?;
            let (Some(pk), Some(pv)) = (kv.k()?, kv.v()?) else {
                return Err(EngineError::Invalid("prefix cache has an empty layer".into()));
            };
            let k = Tensor::cat(&[&pk, &k], 2)?;
            let v = Tensor::cat(&[&pv, &v], 2)?;
            x = (x + block.attn.out(&attend(&q, &k, &v, false)?)?)?;
            x = (&x + block.mlp.forward(&block.mlp_norm.forward(&x)?)?)?;
        }

        let out = self.llm2vae.forward(&self.norm.forward(&x)?)?.narrow(1, 1, frames)?;
        velocity.copy_from_slice(&out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?);
        Ok(())
    }

    /// Sigmoid-shifted t -> sinusoid -> MLP, [1, 1, hidden].
    fn time_embedding(&self, t_logit: f32) -> Result<Tensor> {
        let shift = self.cfg.timestep_shift;
        let t = 1.0 / (1.0 + (-(t_logit as f64)).exp());
        let t = (shift * t / (1.0 + (shift - 1.0) * t)) as f32;
        let half = 128;
        let args: Vec<f32> = (0..half).map(|i| t * (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let emb = Tensor::from_vec(emb, (1, 1, 2 * half), &self.device)?.to_dtype(self.dtype)?;
        Ok(self.time_out.forward(&self.time_in.forward(&emb)?.silu()?)?)
    }

    /// AudioPositionEmbedding rows 0..count, computed instead of shipping the 24576x2048 table.
    fn latent_positions(&self, count: usize) -> Result<Tensor> {
        let pos = Tensor::arange(0u32, count as u32, &self.device)?
            .to_dtype(DType::F32)?
            .clamp(0f32, (self.cfg.max_latent_frames - 1) as f32)?;
        let angles = pos.unsqueeze(1)?.broadcast_mul(&self.pos_div)?;
        let pe = Tensor::stack(&[angles.sin()?, angles.cos()?], 2)?.flatten_from(1)?;
        Ok(pe.unsqueeze(0)?.to_dtype(self.dtype)?)
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    Ok(Tensor::cat(&vec![x; groups], 2)?.reshape((b, h * groups, t, d))?)
}

/// Grouped-query attention, bottom-right causal when asked. Metal gets the fused kernel;
/// elsewhere queries go in blocks so a song-long NAR chunk never builds the full score matrix.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let q_len = q.dim(2)?;
    if q.device().is_metal() {
        let k_len = k.dim(2)?;
        if q_len == 1 || !causal || k_len == q_len {
            return Ok(ops::sdpa(q, k, v, None, causal && q_len > 1, scale as f32, 1.0)?);
        }
        // the fused do_causal is top-left aligned (NaNs once q is shorter than k), so a
        // continued prefill block gets its bottom-right mask spelled out
        let (b, heads, _, _) = q.dims4()?;
        let mask = causal_mask(q_len, k_len, q.device(), q.dtype())?.broadcast_as((b, heads, q_len, k_len))?;
        return Ok(ops::sdpa(q, k, v, Some(&mask), false, scale as f32, 1.0)?);
    }

    let groups = q.dim(1)? / k.dim(1)?;
    let k_t = repeat_kv(k, groups)?.t()?.contiguous()?;
    let v = repeat_kv(v, groups)?.contiguous()?;
    let k_len = k_t.dim(3)?;
    let offset = k_len - q_len;

    let mut blocks = Vec::with_capacity(q_len.div_ceil(QUERY_BLOCK));
    for start in (0..q_len).step_by(QUERY_BLOCK) {
        let len = QUERY_BLOCK.min(q_len - start);
        let mut scores = (q.narrow(2, start, len)?.contiguous()?.matmul(&k_t)? * scale)?;
        if causal {
            scores = scores.broadcast_add(&causal_mask_at(len, k_len, offset + start, q.device(), scores.dtype())?)?;
        }
        blocks.push(ops::softmax_last_dim(&scores)?.matmul(&v)?);
    }
    Ok(Tensor::cat(&blocks, 2)?)
}

/// Additive [q_len, k_len] mask, query i sees keys 0..=k_len-q_len+i.
fn causal_mask(q_len: usize, k_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    causal_mask_at(q_len, k_len, k_len - q_len, device, dtype)
}

/// Same, with query 0 sitting at absolute position `first`.
fn causal_mask_at(q_len: usize, k_len: usize, first: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let values: Vec<f32> = (0..q_len)
        .flat_map(|i| (0..k_len).map(move |j| if j > first + i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Ok(Tensor::from_vec(values, (q_len, k_len), device)?.to_dtype(dtype)?)
}
