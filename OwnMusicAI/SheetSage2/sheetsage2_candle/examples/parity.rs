//! Compares the engine against the FP32 PyTorch dumps of tools/make_reference.py.
//!
//!     cargo run --release --features metal --example parity -- <model_dir> <mert_dir> reference [metal|cpu|cuda] [f16|f32|bf16]

use std::{path::Path, time::Instant};

use candle_core::{DType, Tensor};
use sheetsage2_engine::{Backend, SheetSage2};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn npy<T: candle_core::WithDType>(dir: &Path, name: &str) -> Res<Vec<T>> {
    Ok(Tensor::read_npy(dir.join(format!("{name}.npy")))?.flatten_all()?.to_vec1::<T>()?)
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
    let ok = actual.iter().all(|v| v.is_finite()) && actual.len() == expected.len() && rms_rel <= limit;
    println!("  {} {name}: max rel {max_rel:.2e}, rms rel {rms_rel:.2e}", if ok { "ok  " } else { "FAIL" });
    if !ok {
        failures.push(name.to_string());
    }
}

/// Greedy choice among the tokens the reference actually allows is not known here, so compare
/// the unconstrained argmax - good enough to spot a drifting decoder.
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_dir, mert_dir, reference, rest @ ..] = args.as_slice() else {
        return Err("usage: parity <model_dir> <mert_dir> <reference_dir> [metal|cpu|cuda] [f16|f32|bf16]".into());
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
    let limit = if dtype == DType::F32 { 1e-3 } else { 2e-2 };
    let mut failures = Vec::new();

    let clock = Instant::now();
    let model = SheetSage2::load(Path::new(model_dir), Some(Path::new(mert_dir)), &backend.device()?, dtype)?;
    println!("loaded SheetSage2 ({backend:?}, {dtype:?}) in {:.1}s", clock.elapsed().as_secs_f32());
    let vocab = model.config().vocab_size;

    println!("mel (cpu, f32)");
    let audio: Vec<f32> = npy(reference, "audio")?;
    let mut padded = audio.clone();
    padded.resize(model.config().window_samples(), 0.0);
    let clock = Instant::now();
    let (mel, frames) = model.mel().compute(&padded)?;
    println!("       {frames} frames in {:.2}s", clock.elapsed().as_secs_f32());
    report("log-mel", &mel, &npy::<f32>(reference, "mel")?, 1e-4, &mut failures);

    println!("encoder");
    let clock = Instant::now();
    let memory = model.encode(&audio)?;
    let features = memory.features()?;
    println!("       300 s window in {:.2}s", clock.elapsed().as_secs_f32());
    report("memory", &features, &npy::<f32>(reference, "memory")?, limit, &mut failures);

    println!("decoder");
    let tokens: Vec<u32> = npy(reference, "decoder_tokens")?;
    let prefix = npy::<u32>(reference, "decoder_prefix_len")?[0] as usize;
    let expected: Vec<f32> = npy(reference, "decoder_logits")?;
    let mut logits = vec![0f32; vocab];
    let mut cache = model.new_cache(tokens.len());
    let mut argmax_misses = 0;
    let clock = Instant::now();
    model.decode(&memory, &mut cache, &tokens[..prefix], &mut logits)?;
    report("prefill logits", &logits, &expected[..vocab], limit, &mut failures);
    for (step, token) in tokens[prefix..].iter().enumerate() {
        model.decode(&memory, &mut cache, &[*token], &mut logits)?;
        let row = &expected[(step + 1) * vocab..(step + 2) * vocab];
        let (_, rms) = rel_err(&logits, row);
        if rms > limit {
            report(&format!("step {} logits", step + 1), &logits, row, limit, &mut failures);
        }
        argmax_misses += usize::from(argmax(&logits) != argmax(row));
    }
    let steps = tokens.len() - prefix;
    println!(
        "  {} {steps} decode steps, argmax differs in {argmax_misses}, {:.1} steps/s",
        if argmax_misses == 0 { "ok  " } else { "WARN" },
        (steps + 1) as f32 / clock.elapsed().as_secs_f32()
    );

    // two-block prefill must land on the same logits
    let mut split = model.new_cache(4);
    model.decode(&memory, &mut split, &tokens[..prefix / 2], &mut logits)?;
    model.decode(&memory, &mut split, &tokens[prefix / 2..prefix + 8], &mut logits)?;
    report("block prefill logits", &logits, &expected[8 * vocab..9 * vocab], limit, &mut failures);

    if failures.is_empty() {
        println!("\nALL CHECKS PASSED");
        Ok(())
    } else {
        Err(format!("{} failed: {}", failures.len(), failures.join(", ")).into())
    }
}
