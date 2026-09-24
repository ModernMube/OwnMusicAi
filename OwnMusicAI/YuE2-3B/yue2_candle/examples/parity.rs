//! Compares the engine against the FP32 PyTorch dumps of tools/make_reference.py.
//!
//!     cargo run --release --features metal --example parity -- <model_dir> <vae_dir> reference [metal|cpu] [f16|f32|bf16]

use std::{path::Path, time::Instant};

use candle_core::{DType, Device, Tensor};
use yue2_engine::{Backend, Cache, VaeDecoder, Yue2Model};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn npy<T: candle_core::WithDType>(dir: &Path, name: &str) -> Res<Vec<T>> {
    let t = Tensor::read_npy(dir.join(format!("{name}.npy")))?;
    Ok(t.flatten_all()?.to_vec1::<T>()?)
}

/// max |a - b| / max |b| and the RMS version of it.
fn rel_err(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    let scale = expected.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
    let (mut worst, mut sq, mut ref_sq) = (0f64, 0f64, 0f64);
    for (a, e) in actual.iter().zip(expected) {
        let d = (*a as f64 - *e as f64).abs();
        worst = worst.max(d);
        sq += d * d;
        ref_sq += (*e as f64).powi(2);
    }
    (worst / scale.max(1e-12), (sq / ref_sq.max(1e-24)).sqrt())
}

fn report(name: &str, actual: &[f32], expected: &[f32], limit: f64, failures: &mut Vec<String>) {
    let (max_rel, rms_rel) = rel_err(actual, expected);
    let finite = actual.iter().all(|v| v.is_finite());
    let ok = finite && actual.len() == expected.len() && rms_rel <= limit;
    println!("  {} {name}: max rel {max_rel:.2e}, rms rel {rms_rel:.2e}", if ok { "ok  " } else { "FAIL" });
    if !ok {
        failures.push(name.to_string());
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_dir, vae_dir, reference, rest @ ..] = args.as_slice() else {
        return Err("usage: parity <model_dir> <vae_dir> <reference_dir> [metal|cpu|cuda] [f16|f32|bf16]".into());
    };
    let backend = match rest.first().map(String::as_str) {
        Some("cpu") => Backend::Cpu,
        Some("cuda") => Backend::Cuda,
        _ => Backend::Metal,
    };
    let dtype = match rest.get(1).map(String::as_str) {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        _ => backend.default_dtype(),
    };
    let reference = Path::new(reference);
    let device: Device = backend.device()?;
    // f16/bf16 kernels round differently from torch fp32; f32 should land near 1e-4
    let limit = if dtype == DType::F32 { 1e-3 } else { 2e-2 };
    let mut failures = Vec::new();

    let clock = Instant::now();
    let model = Yue2Model::load(Path::new(model_dir), &device, dtype)?;
    println!("loaded YuE2 ({backend:?}, {dtype:?}) in {:.1}s", clock.elapsed().as_secs_f32());
    let vocab = model.config().vocab_size;

    println!("AR");
    let prefix: Vec<u32> = npy(reference, "prefix")?;
    let steps: Vec<u32> = npy(reference, "ar_steps")?;
    let expected_prefill: Vec<f32> = npy(reference, "ar_logits_prefill")?;
    let expected_steps: Vec<f32> = npy(reference, "ar_logits_step")?;

    let mut logits = vec![0f32; vocab];
    let mut cache = model.new_cache(prefix.len() + 8);
    model.forward(&mut cache, &prefix, &mut logits)?;
    report("prefill logits", &logits, &expected_prefill, limit, &mut failures);
    println!("       argmax {} vs {}", argmax(&logits), argmax(&expected_prefill));
    for (i, token) in steps.iter().enumerate() {
        model.forward(&mut cache, &[*token], &mut logits)?;
        report(&format!("decode step {} logits", i + 1), &logits, &expected_steps[i * vocab..(i + 1) * vocab], limit, &mut failures);
    }

    // same logits whether the prefix goes in at once or in two blocks
    let cut = prefix.len() / 2;
    let mut split: Cache = model.new_cache(4);
    let mut split_logits = vec![0f32; vocab];
    model.forward(&mut split, &prefix[..cut], &mut split_logits)?;
    model.forward(&mut split, &prefix[cut..], &mut split_logits)?;
    report("block prefill logits", &split_logits, &expected_prefill, limit, &mut failures);

    let clock = Instant::now();
    let mut warm = model.new_cache(64);
    model.forward(&mut warm, &prefix, &mut logits)?;
    for token in steps.iter().cycle().take(20) {
        model.forward(&mut warm, &[*token], &mut logits)?;
    }
    println!("       20 decode steps: {:.1} tok/s (incl. prefill)", 21.0 / clock.elapsed().as_secs_f32());

    println!("NAR");
    let tokens: Vec<u32> = npy(reference, "nar_tokens")?;
    let state: Vec<f32> = npy(reference, "nar_state")?;
    let mut chunk = model.new_cache(tokens.len());
    model.forward(&mut chunk, &tokens, &mut logits)?;
    let mut velocity = vec![0f32; state.len()];
    for (t, name) in [(1.3f32, "nar_velocity"), (20.0, "nar_velocity_t20")] {
        let clock = Instant::now();
        model.velocity(&chunk, &state, t, &mut velocity)?;
        let elapsed = clock.elapsed().as_secs_f32();
        report(&format!("velocity t_logit={t}"), &velocity, &npy::<f32>(reference, name)?, limit, &mut failures);
        println!("       {elapsed:.2}s");
    }
    drop((model, cache, split, warm, chunk));

    println!("VAE (f32)");
    let vae = VaeDecoder::load(Path::new(vae_dir), &device)?;
    let latent: Vec<f32> = npy(reference, "vae_latent")?;
    let expected_audio: Vec<f32> = npy(reference, "vae_audio")?;
    let mut audio = vec![0f32; 2 * vae.output_len(latent.len() / 64)];
    let clock = Instant::now();
    vae.decode(&latent, &mut audio)?;
    let elapsed = clock.elapsed().as_secs_f32();
    report("decode", &audio, &expected_audio, 1e-3, &mut failures);
    println!("       {elapsed:.2}s");

    if failures.is_empty() {
        println!("\nALL CHECKS PASSED");
        Ok(())
    } else {
        Err(format!("{} failed: {}", failures.len(), failures.join(", ")).into())
    }
}
