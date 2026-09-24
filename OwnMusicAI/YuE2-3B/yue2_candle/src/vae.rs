use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{EngineError, Result, VaeConfig, config};

/// SnakeBeta with the log-scale params already exponentiated.
struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    fn load(vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get_unchecked("alpha")?;
        let channels = alpha.dim(0)?;
        let beta = vb.get_unchecked("beta")?.exp()?;
        Ok(Self {
            alpha: alpha.exp()?.reshape((1, channels, 1))?,
            inv_beta: (beta + 1e-9)?.recip()?.reshape((1, channels, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok((x + x.broadcast_mul(&self.alpha)?.sin()?.sqr()?.broadcast_mul(&self.inv_beta)?)?)
    }
}

/// Weight-normed Conv1d / ConvTranspose1d, g * v / ||v|| folded at load.
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
    transpose: bool,
}

impl Conv {
    fn load(vb: VarBuilder, padding: usize, stride: usize, dilation: usize, transpose: bool) -> Result<Self> {
        let g = vb.get_unchecked("weight_g")?;
        let v = vb.get_unchecked("weight_v")?;
        let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let bias = vb.get_unchecked("bias").ok().map(|b| b.reshape((1, (), 1))).transpose()?;
        Ok(Self { weight: v.broadcast_mul(&g.broadcast_div(&norm)?)?, bias, padding, stride, dilation, transpose })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = if self.transpose {
            x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, self.dilation, 1)?
        } else {
            x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?
        };
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct ResidualUnit {
    snake_in: Snake,
    conv: Conv,
    snake_mid: Snake,
    project: Conv,
}

impl ResidualUnit {
    fn load(vb: VarBuilder, dilation: usize) -> Result<Self> {
        let vb = vb.pp("layers");
        Ok(Self {
            snake_in: Snake::load(vb.pp(0))?,
            conv: Conv::load(vb.pp(1), 3 * dilation, 1, dilation, false)?,
            snake_mid: Snake::load(vb.pp(2))?,
            project: Conv::load(vb.pp(3), 0, 1, 1, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(&self.snake_in.forward(x)?)?;
        let y = self.project.forward(&self.snake_mid.forward(&y)?)?;
        Ok((x + y)?)
    }
}

struct UpBlock {
    snake: Snake,
    up: Conv,
    units: [ResidualUnit; 3],
}

impl UpBlock {
    fn load(vb: VarBuilder, stride: usize) -> Result<Self> {
        let vb = vb.pp("layers");
        Ok(Self {
            snake: Snake::load(vb.pp(0))?,
            up: Conv::load(vb.pp(1), stride.div_ceil(2), stride, 1, true)?,
            units: [ResidualUnit::load(vb.pp(2), 1)?, ResidualUnit::load(vb.pp(3), 3)?, ResidualUnit::load(vb.pp(4), 9)?],
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.up.forward(&self.snake.forward(x)?)?;
        for unit in &self.units {
            x = unit.forward(&x)?;
        }
        Ok(x)
    }
}

/// Oobleck decoder of YuE2-Vae: latent [1, 64, frames] -> stereo [1, 2, 1920 * frames - 64], FP32.
pub struct VaeDecoder {
    cfg: VaeConfig,
    input: Conv,
    blocks: Vec<UpBlock>,
    snake: Snake,
    output: Conv,
    tanh: bool,
    device: Device,
}

impl VaeDecoder {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let cfg: VaeConfig = config::read(dir)?;
        let dc = &cfg.decoder_config;
        if !dc.use_snake {
            return Err(EngineError::Invalid("only the snake (released) decoder is supported".into()));
        }
        // SAFETY: read-only mmap of the VAE checkpoint.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, device)? };
        let layers = vb.pp("decoder.layers");
        let depth = dc.c_mults.len() + 1;

        // c_mults gets a leading 1, blocks walk the strides backwards: 6, 5, 4, 4, 2, 2
        let blocks = (1..depth)
            .rev()
            .enumerate()
            .map(|(slot, i)| UpBlock::load(layers.pp(slot + 1), dc.strides[i - 1]))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            input: Conv::load(layers.pp(0), 3, 1, 1, false)?,
            snake: Snake::load(layers.pp(depth))?,
            output: Conv::load(layers.pp(depth + 1), 3, 1, 1, false)?,
            tanh: dc.final_tanh,
            blocks,
            device: device.clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    pub fn output_len(&self, frames: usize) -> usize {
        (self.cfg.downsampling_ratio * frames).saturating_sub(64)
    }

    /// `latents` is [frames * 64] row-major (one row per frame); `audio` gets planar
    /// [left..., right...] of [`Self::output_len`] samples each, unclipped.
    pub fn decode(&self, latents: &[f32], audio: &mut [f32]) -> Result<()> {
        let dim = self.cfg.latent_dim;
        let frames = latents.len() / dim;
        if frames == 0 || latents.len() % dim != 0 || audio.len() != 2 * self.output_len(frames) {
            return Err(EngineError::Invalid("vae decode needs [frames*64] latents and a [2*output_len] buffer".into()));
        }
        let mut x = Tensor::from_slice(latents, (1, frames, dim), &self.device)?.transpose(1, 2)?.contiguous()?;
        x = self.input.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.output.forward(&self.snake.forward(&x)?)?;
        if self.tanh {
            x = x.tanh()?;
        }
        audio.copy_from_slice(&x.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }
}
