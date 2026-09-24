# sheetsage2_engine – SheetSage2 Rust/Candle engine

Runs the SheetSage2 neural networks natively with [Candle](https://github.com/huggingface/candle), exposed over a C ABI. The caller drives the pipeline (audio loading, windowing, grammar constraints, greedy decoding, stitching, ABC/MIDI export), see `../sheetsage2_csharp`.

The engine provides two primitives:

| Primitive | Description |
|---|---|
| Encoder | 300 s audio window → log-mel → MERT-v2-FullSong (24 Conformer layers) → weighted layer mix and projection, 7500 frames × 512. |
| Decoder step | One BART decoder step with its own KV cache and the window's precomputed cross-attention keys; returns the logits of the last position. |

Weights load straight from the `model.safetensors` files (mmap), no ONNX conversion. The LoRA adapters are merged into the MERT-v2 weights at load time, on the CPU in f32 – the same way Python's `merge_lora()` does it.

## Prerequisites

- The SheetSage2 weights in the parent folder (`..`).
- MERT-v2-FullSong: `hf download m-a-p/MERT-v2-FullSong --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`. The program finds it in the Hugging Face cache, or you can pass it by hand.
- Rust 1.85+.

## Build

```bash
cargo build --release --features metal        # macOS (Apple Silicon GPU)
cargo build --release --features cuda         # Windows / Linux, NVIDIA
cargo build --release                          # CPU only
```

The output is `target/release/libsheetsage2_engine.dylib`, `.so` on Linux, `sheetsage2_engine.dll` on Windows.

CI (`.github/workflows/build-native.yml`) builds this crate for `osx-arm64` (Metal) and `linux-x64`, `linux-arm64`, `win-x64`, `win-arm64` (CPU) and commits the binaries to `runtimes/<rid>/native/`. The .NET projects use those unless a local `target/release/` build exists. For NVIDIA GPUs see "CUDA builds" in the root README.

## Parity

`tools/make_reference.py` saves an fp32 reference from the official PyTorch implementation, and `examples/parity.rs` checks the engine against it. The model's Python code is not downloaded by the app, so fetch it first:

```bash
hf download m-a-p/SheetSage2 --local-dir .. --include '*.py' requirements.txt
uv venv tools/.venv --python 3.11
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt

tools/.venv/bin/python tools/make_reference.py <mono 24 kHz wav> --out reference
cargo run --release --features metal --example parity -- .. <MERT-v2 folder> reference metal f32
```

Measured on an M1 Pro (RMS relative error against the fp32 PyTorch reference):

| | Metal f32 | Metal f16 | CPU f32 |
|---|---|---|---|
| log-mel | 1.6e-8 | 1.6e-8 | 1.6e-8 |
| Encoder output (memory) | 1.1e-4 | 6.5e-3 | 1.1e-5 |
| Decoder logits (prefill) | 8.6e-6 | 1.4e-3 | 2.8e-6 |
| Argmax match over 48 steps | 48/48 | 48/48 | 48/48 |

The log-mel always runs on the CPU in f32 (the reference turns autocast off for it too), so it does not depend on the device.

## Speed (M1 Pro, Metal, f32)

| Operation | Time |
|---|---|
| Encoding one 300 s window | 13 s |
| Decoding | 140–290 tokens/s (depending on context length) |

On this machine f16 is slower **and** less accurate (23 s per window, 40 tokens/s), so f32 is the default on Metal. On CUDA the default is bf16, matching the reference's autocast – not tested on hardware yet.

## C ABI

Every call returns 0 on success and -1 on failure. The error message can be read from `ss2_last_error()` on the same thread. Rust panics never cross the boundary.

| Function | Description |
|---|---|
| `ss2_engine_create(model_dir, mert_dir?, backend, dtype, out engine)` | Load and LoRA merge. `backend`: 0 cpu, 1 metal, 2 cuda. `dtype`: 0 auto, 1 f32, 2 f16, 3 bf16. |
| `ss2_engine_info(engine, out Ss2Info)` | vocab, sample rate, window size, frame count, decoder context, dtype. |
| `ss2_encode(engine, samples, count, out memory)` / `ss2_memory_destroy` | Encodes one window; shorter audio is padded with silence. |
| `ss2_memory_features(memory, out float*, len)` | The encoder output (`encoder_last_hidden_state`) in f32. |
| `ss2_cache_create(engine, capacity, out cache)` / `ss2_cache_len` / `ss2_cache_destroy` | Decoder KV cache handling. |
| `ss2_decode(engine, memory, cache, tokens, count, logits, logits_len)` | Decoder step, logits of the last position. |

## Implementation notes

- **Depthwise convolution:** Candle computes grouped convolution with one call per group, which is unusable at 1024 groups. It is a weighted sum of shifted slices instead.
- **LayerNorm and GRN:** computed in f32. The GRN L2 norm sums over 30000 frames, which would overflow in f16; the reference's autocast also promotes to f32.
- **Resampling convolution:** the k=2, stride 2 convolution is a pairwise reshape + Linear, so no channel permutation is needed.
- **Metal attention:** Candle's `sdpa` kernel. A continued prefill (the cache already has content) gets an explicit bottom-right aligned mask, because `do_causal` aligns top-left.
- **CPU and CUDA attention:** in 512-query blocks, so even the 7500-frame window never builds the full attention matrix.
