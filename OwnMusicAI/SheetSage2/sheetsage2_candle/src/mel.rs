//! MERT2MelFrontend on the CPU: centered reflect-padded STFT, power spectrum, the checkpoint's
//! HTK mel filterbank, 10*log10 and the fixed per-bin normalization. Everything stays f32,
//! same as the reference which disables autocast for this part.

use std::{sync::Arc, thread};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use realfft::{RealFftPlanner, RealToComplex};

use crate::{EngineError, MertConfig, Result};

/// One triangular filter: first FFT bin it touches and its weights from there.
struct Filter {
    start: usize,
    weights: Vec<f32>,
}

pub struct MelFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    filters: Vec<Filter>,
    mean: Vec<f32>,
    std: Vec<f32>,
    hop: usize,
}

impl MelFrontend {
    /// `vb` points at `feature_extractor`; the buffers are read on the CPU in f32.
    pub fn load(vb: VarBuilder, cfg: &MertConfig) -> Result<Self> {
        if cfg.win_length != cfg.n_fft {
            return Err(EngineError::Invalid("mel frontend expects win_length == n_fft".into()));
        }
        let vb = vb.set_device(Device::Cpu).set_dtype(DType::F32);
        let bins = cfg.n_fft / 2 + 1;
        let fb = vb.get((bins, cfg.num_mel_bins), "mel_scale.fb")?.t()?.to_vec2::<f32>()?;
        let filters = fb
            .into_iter()
            .map(|column| {
                let start = column.iter().position(|w| *w != 0.0).unwrap_or(0);
                let end = column.iter().rposition(|w| *w != 0.0).map_or(start, |i| i + 1);
                Filter { start, weights: column[start..end].to_vec() }
            })
            .collect();

        Ok(Self {
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(cfg.n_fft),
            window: vb.get(cfg.win_length, "spectrogram.window")?.to_vec1()?,
            filters,
            mean: vb.get(cfg.num_mel_bins, "mel_mean")?.to_vec1()?,
            std: vb.get(cfg.num_mel_bins, "mel_std")?.to_vec1::<f32>()?.into_iter().map(|s| s.max(1e-5)).collect(),
            hop: cfg.hop_length,
        })
    }

    pub fn mel_bins(&self) -> usize {
        self.filters.len()
    }

    /// Normalized log-mel, [frames, mel_bins] row-major. frames = samples / hop: torch's centered
    /// STFT yields one more, which MERT2 drops.
    pub fn compute(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let n_fft = self.window.len();
        if samples.len() <= n_fft / 2 {
            return Err(EngineError::Invalid(format!("need more than {} samples for the STFT", n_fft / 2)));
        }
        let frames = samples.len() / self.hop;
        let bins = self.mel_bins();
        let mut mel = vec![0f32; frames * bins];

        let workers = thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let per_worker = frames.div_ceil(workers).max(1);
        thread::scope(|scope| {
            for (chunk, rows) in mel.chunks_mut(per_worker * bins).enumerate() {
                let first = chunk * per_worker;
                scope.spawn(move || self.fill(samples, first, rows));
            }
        });
        Ok((mel, frames))
    }

    fn fill(&self, samples: &[f32], first: usize, rows: &mut [f32]) {
        let n_fft = self.window.len();
        let half = (n_fft / 2) as isize;
        let last = samples.len() as isize - 1;
        let mut frame = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f32; spectrum.len()];

        for (index, row) in rows.chunks_mut(self.mel_bins()).enumerate() {
            let offset = ((first + index) * self.hop) as isize - half;
            for (i, (slot, w)) in frame.iter_mut().zip(&self.window).enumerate() {
                // reflect padding without the edge sample, torch's pad_mode="reflect"
                let mut at = offset + i as isize;
                if at < 0 {
                    at = -at;
                } else if at > last {
                    at = 2 * last - at;
                }
                *slot = samples[at as usize] * w;
            }
            self.fft
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .expect("buffers come from the same plan");
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.re * c.re + c.im * c.im;
            }
            for (j, (out, filter)) in row.iter_mut().zip(&self.filters).enumerate() {
                let energy: f32 = filter.weights.iter().zip(&power[filter.start..]).map(|(w, p)| w * p).sum();
                *out = (10.0 * energy.max(1e-10).log10() - self.mean[j]) / self.std[j];
            }
        }
    }
}
