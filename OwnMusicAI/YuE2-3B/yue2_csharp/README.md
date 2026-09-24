# YuE2.Song – song generation from C# on a Rust/Candle engine

A .NET 10 console app. It drives the whole YuE2 pipeline in C#, while the `../yue2_candle` Rust engine runs the neural networks: Metal on macOS, CUDA GPUs on Windows and Linux.

The C# side handles:
- tokenization (`Microsoft.ML.Tokenizers`, `qwen.tiktoken`);
- prompt and protocol (`Protocol.cs`, ported from `yue2/protocol.py`);
- sampling and CFG (`TokenSampler.cs`, `SongGenerator.cs`);
- chunking, noise and the 32-step midpoint ODE (`AcousticSolver.cs`);
- VAE tiling and WAV writing (`VaeTiler.cs`, `WavWriter.cs`);
- saving and resuming (`SongState`, `--resume`).

The Rust engine only does the AR step, the NAR velocity and the VAE decoding (`Yue2Engine.cs`, `Native.cs`).

With `--abc-audio` the score is not generated but transcribed from a recording by the [SheetSage2](../../SheetSage2/sheetsage2_csharp) engine (without chords under `--cot melody`, with chord symbols under `--cot full`). The transcribed score is also saved to `<name>.source.abc`.

## Prerequisites

- The YuE2-3B weights in the parent folder (`..`), e.g. `hf download m-a-p/YuE2-3B --local-dir ..`.
- YuE2-Vae: `hf download m-a-p/YuE2-Vae`. The program finds it in the Hugging Face cache, or pass it with `--vae`.
- Rust (1.85+) and the .NET 10 SDK.

## Build

```bash
cd ../yue2_candle && cargo build --release --features metal   # Windows/Linux + NVIDIA: --features cuda
cd ../yue2_csharp && dotnet build -c Release
```

The csproj copies `libyue2_engine.dylib` / `.so` / `yue2_engine.dll` into the output automatically: your local cargo build if there is one, otherwise the CI-built binary from `../yue2_candle/runtimes/<rid>/native/` (see `OwnMusicAI/NativeEngines.targets`). So the cargo step is optional unless you change the Rust code.

## Run

```bash
# full song
bin/Release/net10.0/YuE2.Song --example ../examples/tonight-awake.json --out runs/song.wav

# cover: the score comes from an existing recording (SheetSage2)
bin/Release/net10.0/YuE2.Song --example ../examples/tonight-awake.json --cot melody --abc-audio ~/Music/original.mp3 --out runs/cover.wav

# quick test: no score, 8 s
bin/Release/net10.0/YuE2.Song --example ../examples/tonight-awake.json --cot off --max-seconds 8 --out runs/test.wav

# resume an interrupted run
bin/Release/net10.0/YuE2.Song --resume runs/song.state.json
```

The program saves after every phase:
- **After score planning:** `<name>.abc` and `<name>.state.json` (prompt tokens, score).
- **After the music tokens:** `<name>.state.json` is updated with the tokens.
- **After the audio latents:** `<name>.latents.f32`.

`--resume` continues from the first missing phase.

| Option | Meaning |
|---|---|
| `--model`, `--vae` | Model folders (default: `..` and the Hugging Face cache). |
| `--device metal\|cuda\|cpu`, `--dtype auto\|f32\|f16\|bf16` | Device and precision (auto: Metal f16, CUDA bf16, CPU f32). |
| `--example`, `--style`, `--lyrics` | Input. |
| `--cot off\|melody\|full`, `--abc` | Score planning mode, or a ready-made ABC score. |
| `--abc-audio <file>`, `--sheetsage <dir>` | Score from a recording via SheetSage2 (default folder: `../SheetSage2`). |
| `--seed`, `--cfg` | Seed and text guidance. |
| `--max-abc-tokens`, `--max-seconds` | For shortened test runs. |
| `--ode-steps`, `--context`, `--vae-tile` | Flow-matching steps; acoustic chunk context; VAE tile size (default: 256 frames). |

## Benchmarks (M1 Pro, 32 GB, Metal f16)

The ONNX columns come from an earlier, abandoned ONNX Runtime implementation, for comparison.

| Test | ONNX + CPU | ONNX + WebGPU | **Candle + Metal** |
|---|---|---|---|
| 8 s song (`--cot off`, with CFG), full run | 161 s | 70 s | **50 s** (15 s of it loading) |
| Decoding at an 8000-token context | – | ~5 tokens/s | **30 tokens/s** |
| NAR step, 200 s of audio | 70 s | 16 s | **8 s** |

## Notes

- **Seed:** the RNG is .NET's `System.Random`, so the same seed does not give a bit-identical song to the Python `yue2` package.
- **License:** the model weights are CC BY-NC 4.0, non-commercial use only.
