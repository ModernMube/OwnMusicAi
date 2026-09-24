//! Pieces shared by the encoder and the decoder.

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{VarBuilder, ops};

use crate::Result;

/// Query rows per attention block off Metal, keeps the score matrix of a 7500 frame window bounded.
const QUERY_BLOCK: usize = 512;

/// LayerNorm that always runs in f32, the same thing autocast does in the reference.
/// The ConvNeXt/Conformer residual streams get large enough to hurt in f16.
pub(crate) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub(crate) fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?.to_dtype(DType::F32)?,
            bias: vb.get(dim, "bias")?.to_dtype(DType::F32)?,
            eps: eps as f32,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = ops::layer_norm(&x.to_dtype(DType::F32)?.contiguous()?, &self.weight, &self.bias, self.eps)?;
        Ok(normed.to_dtype(x.dtype())?)
    }
}

/// Multi-head attention over [b, heads, t, head_dim]. Metal gets the fused kernel; elsewhere
/// the queries go in blocks. `causal` is bottom-right aligned (continued prefill sees the cache).
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let (q_len, k_len) = (q.dim(2)?, k.dim(2)?);
    let causal = causal && q_len > 1;

    if q.device().is_metal() {
        if !causal || k_len == q_len {
            return Ok(ops::sdpa(q, k, v, None, causal, scale as f32, 1.0)?);
        }
        // do_causal is top-left aligned, so a prefill on top of a filled cache spells its mask out
        let (b, heads, _, _) = q.dims4()?;
        let mask = causal_mask(q_len, k_len, k_len - q_len, q.device(), q.dtype())?.broadcast_as((b, heads, q_len, k_len))?;
        return Ok(ops::sdpa(q, k, v, Some(&mask), false, scale as f32, 1.0)?);
    }

    let k_t = k.t()?.contiguous()?;
    let v = v.contiguous()?;
    let offset = k_len - q_len;
    let mut blocks = Vec::with_capacity(q_len.div_ceil(QUERY_BLOCK));
    for start in (0..q_len).step_by(QUERY_BLOCK) {
        let len = QUERY_BLOCK.min(q_len - start);
        let mut scores = (q.narrow(2, start, len)?.contiguous()?.matmul(&k_t)? * scale)?;
        if causal {
            scores = scores.broadcast_add(&causal_mask(len, k_len, offset + start, q.device(), scores.dtype())?)?;
        }
        // softmax in f32 like the reference sdpa under autocast
        let weights = ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        blocks.push(weights.matmul(&v)?);
    }
    Ok(Tensor::cat(&blocks, 2)?)
}

/// Additive [q_len, k_len] mask, query 0 sits at absolute position `first`.
fn causal_mask(q_len: usize, k_len: usize, first: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let values: Vec<f32> = (0..q_len)
        .flat_map(|i| (0..k_len).map(move |j| if j > first + i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Ok(Tensor::from_vec(values, (q_len, k_len), device)?.to_dtype(dtype)?)
}
