//! Song-sized timings: prefill in blocks, decode at a long context, one NAR velocity.
//!
//!     cargo run --release --features metal --example bench -- <model_dir> [frames] [prefix]

use std::{path::Path, time::Instant};

use yue2_engine::{Backend, Yue2Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_dir = args.first().ok_or("usage: bench <model_dir> [frames] [prefix]")?;
    let frames: usize = args.get(1).map_or(Ok(5000), |s| s.parse())?;
    let prefix: usize = args.get(2).map_or(Ok(3000), |s| s.parse())?;

    let backend = Backend::Metal;
    let model = Yue2Model::load(Path::new(model_dir), &backend.device()?, backend.default_dtype())?;
    let vocab = model.config().vocab_size;
    let mut logits = vec![0f32; vocab];

    // prefix + codec + MUSIC_END, like one acoustic chunk
    let tokens: Vec<u32> = (0..prefix + frames + 1).map(|i| if i < prefix { (i * 7919 % 150000) as u32 } else { 151853 + (i * 104729 % 32768) as u32 }).collect();
    let mut cache = model.new_cache(tokens.len() + 64);
    let clock = Instant::now();
    for block in tokens.chunks(1024) {
        model.forward(&mut cache, block, &mut logits)?;
    }
    println!("prefill {} tokens: {:.1}s", tokens.len(), clock.elapsed().as_secs_f32());

    let state: Vec<f32> = (0..frames * 64).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    let mut velocity = vec![0f32; state.len()];
    for run in 0..2 {
        let clock = Instant::now();
        model.velocity(&cache, &state, 0.3, &mut velocity)?;
        let s = clock.elapsed().as_secs_f32();
        println!("velocity #{run} ({frames} frames, prefix {}): {s:.2}s -> 64 evals ~ {:.1} min", cache.len(), 64.0 * s / 60.0);
    }
    println!("velocity finite: {}", velocity.iter().all(|v| v.is_finite()));

    let clock = Instant::now();
    for _ in 0..20 {
        model.forward(&mut cache, &[151853 + 17], &mut logits)?;
    }
    println!("decode at {} context: {:.1} tok/s", cache.len(), 20.0 / clock.elapsed().as_secs_f32());
    Ok(())
}
