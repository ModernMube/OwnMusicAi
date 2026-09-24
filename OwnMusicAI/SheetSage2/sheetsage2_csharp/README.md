# SheetSage2 – audio recording to score from C#, on a Rust/Candle engine

A .NET 10 library (`SheetSage2.Transcription`) and console app (`SheetSage2.Cli`). It drives the whole SheetSage2 pipeline in C#, while the `../sheetsage2_candle` Rust engine runs the neural networks: Metal on macOS, CUDA GPUs on Windows and Linux.

The C# side handles:
- audio loading, downmixing and resampling (**OwnAudioSharp**, `AudioLoader.cs`) – this replaces the Python `ffmpeg` pipeline;
- the window plan and stitching overlapping windows (`WindowStitcher.cs`);
- the token vocabulary and the event grammar (`Tokenizer.cs`, `Grammar.cs`);
- grammar-constrained greedy decoding (`Generation.cs`);
- building, serializing and validating the ABC score (`Notation/`);
- LAB/TSV/JSON annotations and MIDI export (`Exports.cs`, `MidiExport.cs` – also OwnAudio).

OwnAudio comes from the `OwnAudioSharp.Basic` and `OwnAudioSharp.Midi` NuGet packages.

## Prerequisites

- The SheetSage2 weights in `..` (e.g. `hf download m-a-p/SheetSage2 --local-dir ..`), MERT-v2-FullSong in the Hugging Face cache (see `../sheetsage2_candle/README.md`).
- Rust (1.85+) and the .NET 10 SDK.

## Build

```bash
cd ../sheetsage2_candle && cargo build --release --features metal   # Windows/Linux + NVIDIA: --features cuda
cd ../sheetsage2_csharp/SheetSage2.Cli && dotnet build -c Release
```

## Run

```bash
# full transcription: score, MIDI, annotations
bin/Release/net10.0/SheetSage2 ~/Music/song.mp3 --out runs/song

# melody-only score (for covers, pairs with YuE2 --cot melody)
bin/Release/net10.0/SheetSage2 ~/Music/song.mp3 --out runs/song --melody-only

# quick test on the first 60 seconds
bin/Release/net10.0/SheetSage2 ~/Music/song.mp3 --out runs/test --max-seconds 60
```

| Option | Meaning |
|---|---|
| `--out <dir>` | Output folder (default: `output`). |
| `--model`, `--mert` | Model folders (default: the folder found above the binary, and the HF cache). |
| `--device metal\|cuda\|cpu`, `--dtype auto\|f32\|f16\|bf16` | Device and precision (auto: f32 on Metal and CPU, bf16 on CUDA). |
| `--melody-only` | Score and playback without chords; the raw annotations are unchanged. |
| `--max-seconds <s>` | Only the start of the song. |
| `--overlap <s>`, `--lookahead <s>` | Window overlap (200) and discarded window tail (100). |
| `--prompts a,b,c` | Task prompts (default: the full transcription). |
| `--replay <tokens.json>` | Exports the tokens of a reference run instead of decoding (parity tool). |

### Output

| File | Content |
|---|---|
| `score.abc` | Two-voice ABC score (Vocal + Ins) with chord symbols. |
| `transcription.mid`, `melody*.mid`, `chords.mid` | Playable MIDI with the original timing. |
| `events.json`, `events.tsv` | Every decoded event, tokens included. |
| `chord.lab`, `key.lab`, `structure.lab`, `beat.lab`, `downbeat.lab`, `melody_*.lab` | Timed annotations. |
| `tokens.txt`, `result.json` | Per-window tokens and a summary. |

## As a library

```csharp
using var transcriber = new Transcriber(modelDir, SheetSage2Engine.FindMert(modelDir), ComputeBackend.Metal, ComputeDType.Auto);
var result = transcriber.Transcribe("song.mp3", new TranscribeOptions { MelodyOnly = true });
string? abc = result.Abc;          // score as text
byte[]? midi = result.Midi;        // bytes of transcription.mid
transcriber.Save(result, "runs/song");
```

The YuE2 song generator uses this for `--abc-audio`, see `../../YuE2-3B/yue2_csharp`.

## Benchmarks (M1 Pro, 32 GB, Metal f32)

| Test | Python (CPU, fp32) | C# + CPU | **C# + Metal** |
|---|---|---|---|
| 162 s song (1 window, 2463 tokens) | 94 s | 245 s | **31 s** |
| 375 s song (2 windows, with overlap) | 108 s | – | **37 s** |

The Python reference runs on PyTorch's own multithreaded CPU kernels, so it beats the C# CPU mode; the Metal column is what matters.

## Matching the reference

**In CPU (f32) mode the chain reproduces the reference bit for bit.** On the same song the C# side decoded all 2463 tokens exactly like the official Python run, and from there `score.abc`, every `.lab`, `events.tsv`/`events.json` and every MIDI note are identical.

`--replay` mode (exporting the Python tokens through the C# chain) confirms the same for a two-window song stitched with overlap: identical score (full and melody-only), identical annotations, identical MIDI, and the same 1506-token overlap prefix.

On Metal the decoding **can drift**: the greedy choice often hangs by a hair (at the measured divergence point the top two token logits were 17.053 and 17.049), and GPU floating-point arithmetic adds more noise than that. The score stays musically the same, but the event count can differ by a few percent. If you need bit-identical output, use `--device cpu`.

## Notes

- **Audio loading:** OwnAudio's native (Symphonia) decoder reads mp3, flac, wav, aac, m4a, ogg and aiff. Stereo→mono follows ffmpeg's energy-preserving rule ((L+R)/√2), because the model was trained on absolute loudness – a plain average would feed it input 3 dB quieter.
- **Resampling:** for sources that are not 24 kHz, OwnAudio's FFT-based resampler leaves a delay of about 5 ms at the start of the output. That is negligible against the model's 10 ms time grid, but it can shift timestamps by that much compared to the Python run. 24 kHz mono input is bit-identical.
- **Not ported:** the `preset="paper"` benchmark mode, score and audio rendering (`render.py`, Playwright/Verovio), and the tensor exports (`--export-logits` and friends). `playback.json` is not written either, the MIDI files are.
- **License:** the model weights are CC BY-NC 4.0, non-commercial use only.
