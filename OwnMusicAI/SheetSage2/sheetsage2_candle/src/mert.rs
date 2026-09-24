//! MERT-v2 FullSong encoder: ConvNeXt subsampler (100 Hz mel -> 25 Hz), 24 RoPE Conformer blocks,
//! and SheetSage2's softmax layer mix + projection on top. Attention LoRA adapters are merged
//! into the base weights at load time, in f32 on the CPU, before any cast.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, ops, rotary_emb::rope};

use crate::{EngineError, MertConfig, Result, nn::{LayerNorm, attend}};

/// Depthwise Conv1d over [b, t, c], as a sum of weighted shifted views. Candle runs a grouped conv
/// one group at a time, which is hopeless at groups = 1024.
struct Depthwise {
    taps: Vec<Tensor>,
    bias: Option<Tensor>,
    pad: usize,
}

impl Depthwise {
    fn load(vb: VarBuilder, channels: usize, kernel: usize, bias: bool) -> Result<Self> {
        let weight = vb.get((channels, 1, kernel), "weight")?;
        let taps = (0..kernel)
            .map(|i| weight.narrow(2, i, 1)?.reshape((1, 1, channels)))
            .collect::<candle_core::Result<_>>()?;
        let bias = bias.then(|| vb.get(channels, "bias")).transpose()?;
        Ok(Self { taps, bias, pad: (kernel - 1) / 2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let len = x.dim(1)?;
        let padded = x.pad_with_zeros(1, self.pad, self.pad)?;
        let mut out = match &self.bias {
            Some(bias) => bias.reshape((1, 1, ()))?.broadcast_as(x.shape())?.contiguous()?,
            None => x.zeros_like()?,
        };
        for (i, tap) in self.taps.iter().enumerate() {
            out = (out + padded.narrow(1, i, len)?.broadcast_mul(tap)?)?;
        }
        Ok(out)
    }
}

/// Global response norm; the L2 over time runs in f32 (it is a sum over 30000 frames).
struct Grn {
    weight: Tensor,
    bias: Tensor,
}

impl Grn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let magnitude = x.to_dtype(DType::F32)?.sqr()?.sum_keepdim(1)?.sqrt()?;
        let scale = magnitude.broadcast_div(&(magnitude.mean_keepdim(2)? + 1e-6)?)?.to_dtype(x.dtype())?;
        let gained = x.broadcast_mul(&scale)?.broadcast_mul(&self.weight)?;
        Ok((gained.broadcast_add(&self.bias)? + x)?)
    }
}

struct ConvNextLayer {
    depthwise: Depthwise,
    norm: LayerNorm,
    up: Linear,
    grn: Grn,
    down: Linear,
}

impl ConvNextLayer {
    fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let pw = vb.pp("pointwise_block");
        Ok(Self {
            depthwise: Depthwise::load(vb.pp("depthwise_block.1"), dim, 7, true)?,
            norm: LayerNorm::load(pw.pp("0"), dim, eps)?,
            up: candle_nn::linear(dim, 4 * dim, pw.pp("1"))?,
            grn: Grn { weight: pw.get((1, 1, 4 * dim), "3.weight")?, bias: pw.get((1, 1, 4 * dim), "3.bias")? },
            down: candle_nn::linear(4 * dim, dim, pw.pp("4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(&self.depthwise.forward(x)?)?;
        let h = self.grn.forward(&self.up.forward(&h)?.gelu_erf()?)?;
        Ok((x + self.down.forward(&h)?)?)
    }
}

/// LayerNorm + Conv1d(kernel 2, stride 2), written as a pairwise reshape and a Linear.
struct Downsample {
    norm: LayerNorm,
    proj: Linear,
}

impl Downsample {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let (b, t, c) = x.dims3()?;
        let half = t / 2;
        let pairs = x.narrow(1, 0, half * 2)?.reshape((b, half, 2, c))?.transpose(2, 3)?.reshape((b, half, 2 * c))?;
        Ok(self.proj.forward(&pairs)?)
    }
}

struct ConvNextBlock {
    downsample: Option<Downsample>,
    layers: Vec<ConvNextLayer>,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |proj: &Linear| proj.forward(x)?.reshape((b, t, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous();
        let q = rope(&split(&self.q)?, cos, sin)?;
        let k = rope(&split(&self.k)?, cos, sin)?;
        let attended = attend(&q, &k, &split(&self.v)?, false)?;
        Ok(self.out.forward(&attended.transpose(1, 2)?.reshape((b, t, ()))?)?)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, cfg: &MertConfig) -> Result<Self> {
        Ok(Self {
            up: candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("w_1"))?,
            down: candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("w_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.down.forward(&self.up.forward(x)?.gelu_erf()?)?)
    }
}

/// LN -> pointwise (GLU) -> depthwise k=31 -> LN -> GELU -> pointwise, all bias-free convs.
struct ConvModule {
    norm: LayerNorm,
    gate: Linear,
    depthwise: Depthwise,
    mid_norm: LayerNorm,
    out: Linear,
}

impl ConvModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let width = x.dim(2)?;
        let gated = self.gate.forward(&self.norm.forward(x)?)?;
        let h = (gated.narrow(2, 0, width)? * ops::sigmoid(&gated.narrow(2, width, width)?)?)?;
        let h = self.mid_norm.forward(&self.depthwise.forward(&h)?)?.gelu_erf()?;
        Ok(self.out.forward(&h)?)
    }
}

struct ConformerBlock {
    ffn1_norm: LayerNorm,
    ffn1: FeedForward,
    attn_norm: LayerNorm,
    attn: Attention,
    conv: ConvModule,
    ffn2_norm: LayerNorm,
    ffn2: FeedForward,
    final_norm: LayerNorm,
}

impl ConformerBlock {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = (x + (self.ffn1.forward(&self.ffn1_norm.forward(x)?)? * 0.5)?)?;
        let x = (self.attn.forward(&self.attn_norm.forward(&x)?, cos, sin)? + &x)?;
        let x = (self.conv.forward(&x)? + &x)?;
        let x = (&x + (self.ffn2.forward(&self.ffn2_norm.forward(&x)?)? * 0.5)?)?;
        self.final_norm.forward(&x)
    }
}

/// LoRA source for the attention projections: `adapter.layers.N.attn.<proj>.lora_{A,B}`.
pub struct Adapters<'a> {
    pub vb: VarBuilder<'a>,
    pub rank: usize,
    pub scale: f64,
}

pub struct MertEncoder {
    blocks: Vec<ConvNextBlock>,
    layers: Vec<ConformerBlock>,
    mix: Vec<f64>,
    projection: Linear,
    inv_freq: Vec<f32>,
    dtype: DType,
    device: Device,
}

impl MertEncoder {
    /// `vb` is the MERT-v2 root, `head` the SheetSage2 root (layer_weight, encoder_projection).
    pub fn load(vb: VarBuilder, adapters: Option<Adapters>, head: VarBuilder, cfg: &MertConfig, out_dim: usize) -> Result<Self> {
        let (h, dtype, device) = (cfg.hidden_size, head.dtype(), head.device().clone());
        if cfg.subsampling_channels.len() != 3 || cfg.subsampling_depths.len() != 3 {
            return Err(EngineError::Invalid("MERT-v2 subsampler needs three widths and depths".into()));
        }

        let mut width = cfg.num_mel_bins;
        let mut blocks = Vec::with_capacity(3);
        for (i, (&channels, &depth)) in cfg.subsampling_channels.iter().zip(&cfg.subsampling_depths).enumerate() {
            let bvb = vb.pp("subsampling_module").pp(i);
            let eps = cfg.subsampling_layer_norm_eps;
            let downsample = if width != channels || i > 0 {
                let rvb = bvb.pp("resampling_layer");
                let weight = rvb.get((channels, width, 2), "2.weight")?.reshape((channels, 2 * width))?;
                Some(Downsample {
                    norm: LayerNorm::load(rvb.pp("0"), width, eps)?,
                    proj: Linear::new(weight, Some(rvb.get(channels, "2.bias")?)),
                })
            } else {
                None
            };
            let layers = (0..depth)
                .map(|j| ConvNextLayer::load(bvb.pp("convnext_layers").pp(j), channels, eps))
                .collect::<Result<_>>()?;
            blocks.push(ConvNextBlock { downsample, layers });
            width = channels;
        }

        let eps = cfg.layer_norm_eps;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let lvb = vb.pp("layers").pp(i);
                let cvb = lvb.pp("conv_module.conv_block");
                let kernel = cfg.conv_depthwise_kernel_size;
                Ok(ConformerBlock {
                    ffn1_norm: LayerNorm::load(lvb.pp("ffn1_layer_norm"), h, eps)?,
                    ffn1: FeedForward::load(lvb.pp("ffn1"), cfg)?,
                    attn_norm: LayerNorm::load(lvb.pp("attn_layer_norm"), h, eps)?,
                    attn: Attention {
                        q: projection(&lvb, adapters.as_ref(), i, "query_proj", h, dtype, &device)?,
                        k: projection(&lvb, adapters.as_ref(), i, "key_proj", h, dtype, &device)?,
                        v: projection(&lvb, adapters.as_ref(), i, "value_proj", h, dtype, &device)?,
                        out: projection(&lvb, adapters.as_ref(), i, "out_proj", h, dtype, &device)?,
                        heads: cfg.num_attention_heads,
                        head_dim: cfg.head_dim(),
                    },
                    conv: ConvModule {
                        norm: LayerNorm::load(lvb.pp("conv_module.layer_norm"), h, eps)?,
                        gate: Linear::new(cvb.get((2 * h, h, 1), "1.weight")?.reshape((2 * h, h))?, None),
                        depthwise: Depthwise::load(cvb.pp("3"), h, kernel, false)?,
                        mid_norm: LayerNorm::load(cvb.pp("4.1"), h, eps)?,
                        out: Linear::new(cvb.get((h, h, 1), "6.weight")?.reshape((h, h))?, None),
                    },
                    ffn2_norm: LayerNorm::load(lvb.pp("ffn2_layer_norm"), h, eps)?,
                    ffn2: FeedForward::load(lvb.pp("ffn2"), cfg)?,
                    final_norm: LayerNorm::load(lvb.pp("final_layer_norm"), h, eps)?,
                })
            })
            .collect::<Result<_>>()?;

        let mix = head.clone().set_device(Device::Cpu).set_dtype(DType::F32).get(cfg.num_hidden_layers + 1, "layer_weight")?;
        let mix = ops::softmax_last_dim(&mix)?.to_vec1::<f32>()?.into_iter().map(f64::from).collect();

        let hd = cfg.head_dim();
        let inv_freq = (0..hd)
            .step_by(2)
            .map(|i| (1.0 / cfg.rotary_embedding_base.powf(i as f64 / hd as f64)) as f32)
            .collect();

        Ok(Self {
            blocks,
            layers,
            mix,
            projection: candle_nn::linear(h, out_dim, head.pp("encoder_projection"))?,
            inv_freq,
            dtype,
            device,
        })
    }

    /// [frames, mel_bins] row-major log-mel -> SheetSage2 memory [1, frames / 4, out_dim].
    pub fn forward(&self, mel: &[f32], frames: usize) -> Result<Tensor> {
        let bins = mel.len() / frames.max(1);
        let mut x = Tensor::from_slice(mel, (1, frames, bins), &self.device)?.to_dtype(self.dtype)?;
        for block in &self.blocks {
            if let Some(down) = &block.downsample {
                x = down.forward(&x)?;
            }
            for layer in &block.layers {
                x = layer.forward(&x)?;
            }
        }

        let (cos, sin) = self.rope_tables(x.dim(1)?)?;
        let mut mixed = (&x * self.mix[0])?;
        for (layer, weight) in self.layers.iter().zip(&self.mix[1..]) {
            x = layer.forward(&x, &cos, &sin)?;
            mixed = (mixed + (&x * *weight)?)?;
        }
        Ok(self.projection.forward(&mixed)?)
    }

    fn rope_tables(&self, len: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let inv_freq = Tensor::from_slice(&self.inv_freq, (1, half), &self.device)?;
        let positions = Tensor::arange(0u32, len as u32, &self.device)?.to_dtype(DType::F32)?;
        let angles = positions.unsqueeze(1)?.matmul(&inv_freq)?;
        Ok((angles.cos()?.to_dtype(self.dtype)?, angles.sin()?.to_dtype(self.dtype)?))
    }
}

/// q/k/v/out Linear of layer `index`, with its LoRA delta folded in when adapters are given.
fn projection(
    lvb: &VarBuilder,
    adapters: Option<&Adapters>,
    index: usize,
    name: &str,
    width: usize,
    dtype: DType,
    device: &Device,
) -> Result<Linear> {
    let pvb = lvb.pp("attn").pp(name);
    let bias = pvb.get(width, "bias")?;
    let Some(lora) = adapters else {
        return Ok(Linear::new(pvb.get((width, width), "weight")?, Some(bias)));
    };
    let base = pvb.clone().set_device(Device::Cpu).set_dtype(DType::F32).get((width, width), "weight")?;
    let avb = lora.vb.pp("layers").pp(index).pp("attn").pp(name).set_device(Device::Cpu).set_dtype(DType::F32);
    let a = avb.get((lora.rank, width), "lora_A.weight")?;
    let b = avb.get((width, lora.rank), "lora_B.weight")?;
    let merged = (base + (b.matmul(&a)? * lora.scale)?)?;
    Ok(Linear::new(merged.to_device(device)?.to_dtype(dtype)?, Some(bias)))
}
