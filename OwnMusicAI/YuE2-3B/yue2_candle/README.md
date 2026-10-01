# yue2_engine – YuE2-3B Rust/Candle engine

Runs the YuE2-3B neural networks natively with [Candle](https://github.com/huggingface/candle), exposed over a C ABI. The caller drives the pipeline (prompt, sampling, CFG, ODE solver, chunking, VAE tiling), see `../yue2_csharp`.

The engine provides three primitives:

| Primitive | Description |
|---|---|
| AR step | Appends tokens to the KV cache (which stays on the GPU) and returns the last position's logits for a window of the vocab. A batched variant decodes one token into each of several caches in one pass (CFG). |
| NAR velocity | The flow-matching velocity of an acoustic chunk against the chunk's prefix cache. |
| VAE decoder | Latents to 48 kHz stereo audio (fp32). |

Weights load straight from the `model.safetensors` files (mmap), no ONNX conversion.

## Build

```bash
cargo build --release --features metal        # macOS (Apple Silicon GPU)
cargo build --release --features cuda         # Windows / Linux, NVIDIA
cargo build --release                          # CPU only
```

The output is `target/release/libyue2_engine.dylib`, `.so` on Linux, `yue2_engine.dll` on Windows.

CI (`.github/workflows/build-native.yml`) builds this crate for `osx-arm64` (Metal) and `linux-x64`, `linux-arm64`, `win-x64`, `win-arm64` (CPU) and commits the binaries to `runtimes/<rid>/native/`. The .NET projects use those unless a local `target/release/` build exists. For NVIDIA GPUs see "CUDA builds" in the root README.

## Parity

`tools/make_reference.py` saves an fp32 reference from the official `yue2` PyTorch implementation, and `examples/parity.rs` checks the engine against it:

```bash
# one-off Python environment for the reference (torch + the yue2 package from the model's wheel)
uv venv tools/.venv --python 3.12
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt

tools/.venv/bin/python tools/make_reference.py --vae-dir <YuE2-Vae folder> --out reference
cargo run --release --features metal --example parity -- .. <YuE2-Vae folder> reference metal f16
```

The `reference/` folder is not in the repo (it is large and fully reproducible), so run `make_reference.py` once before the first parity run.

Measured on an M1 Pro (RMS relative error against the fp32 PyTorch reference):

| | Metal f16 | CPU f32 |
|---|---|---|
| AR prefill / decode logits | 5e-4 – 8e-4 (same argmax) | 4e-7 – 8e-7 |
| Blocked prefill | 5e-4 | 4e-7 |
| NAR velocity | 1.7e-3 – 2.4e-3 | 2e-4 – 3e-4 |
| VAE decoding | 2.5e-6 | 1.9e-6 |

CUDA uses the same blocked attention path as the CPU, but it has not been tested on hardware yet.

## Speed (M1 Pro, Metal, f16)

From `examples/bench.rs`:

| Operation | Time |
|---|---|
| Prefill, 8000 tokens | 9.5 s |
| Decoding at an 8000-token context | 30.4 tokens/s |
| Decoding at a 5000-token context, full vocab / codec window | 36.0 / 41.0 tokens/s |
| CFG pair at a 5000-token context, two passes / batched | 18.5 / 30.7 tokens/s |
| NAR velocity, 5000 frames (200 s of audio) + 8000-token prefix | 8.05 s (64 evaluations ≈ 8.6 min) |

## C ABI

Every call returns 0 on success and -1 on failure. The error message can be read from `yue2_last_error()` on the same thread. Rust panics never cross the boundary.

| Function | Description |
|---|---|
| `yue2_engine_create(model_dir, vae_dir?, backend, dtype, out engine)` | Load. `backend`: 0 cpu, 1 metal, 2 cuda. `dtype`: 0 auto, 1 f32, 2 f16, 3 bf16. |
| `yue2_engine_info(engine, out Yue2Info)` | vocab, latent_dim, layers, context, sample rate, dtype. |
| `yue2_cache_create(engine, capacity, out cache)` / `yue2_cache_len` / `yue2_cache_destroy` | KV cache handling. |
| `yue2_ar_forward(engine, cache, tokens, count, logits_first, logits, logits_len)` | AR step; logits of ids `logits_first..logits_first + logits_len`. |
| `yue2_ar_decode(engine, caches, tokens, count, logits_first, logits, logits_len)` | One token into each of `count` distinct caches in one pass; `logits` holds one window per cache. |
| `yue2_nar_velocity(engine, prefix_cache, latents, frames, t_logit, velocity)` | Latents are `[frames*64]` row-major; `t_logit = clamp(logit(t), -20, 20)`. |
| `yue2_vae_output_len(engine, frames)` / `yue2_vae_decode(engine, latents, frames, audio, audio_len)` | Planar stereo: left channel first, then right. |

## Implementation notes

- **RMSNorm:** computed in f32, because the residual stream grows past 400 in the deep layers and its square would overflow in fp16.
- **Metal causal attention:** the `do_causal` switch of Candle's `sdpa` kernel aligns top-left and returns NaN when the query is shorter than the keys. A continued prefill therefore gets an explicit bottom-right aligned mask.
- **Small-batch projections on Metal:** Candle runs a gemv for one row, but from two rows on it switches to a gemm that streams the weights at well under half the bandwidth. `src/gemv.rs` adds a kernel that reads each weight row once for 2-4 rows, which is what makes the batched CFG decode pay off.
- **Logit window:** the sampler only reads part of the vocab (the music phase 32769 of 184704 ids), so only those lm_head rows are computed.
- **CPU and CUDA attention:** computed in 512-query blocks, so even a song-long NAR chunk never builds the full attention matrix.
- **VAE:** weight norm is baked into the weights at load time, the SnakeBeta parameters are pre-exponentiated.
