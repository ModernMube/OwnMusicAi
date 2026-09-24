# Third-party notices

OwnMusicAI's own source code is licensed under CC BY-NC 4.0 (see `LICENSE`). The parts below come from,
or are derived from, other projects and keep their original licenses.

## Derived code

| Where | Derived from | License |
|---|---|---|
| `OwnMusicAI/YuE2-3B/yue2_candle/src/vae.rs` – Oobleck VAE decoder | [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (via YuE2's `modeling_vae.py`), Copyright (c) 2023 Stability AI | MIT, [`licenses/stable-audio-tools-MIT.txt`](licenses/stable-audio-tools-MIT.txt) |
| `OwnMusicAI/YuE2-3B/yue2_candle/src/vae.rs` – SnakeBeta activation | [BigVGAN](https://github.com/NVIDIA/BigVGAN), Copyright (c) 2022 NVIDIA CORPORATION | MIT, [`licenses/SnakeBeta-NVIDIA-MIT.txt`](licenses/SnakeBeta-NVIDIA-MIT.txt) |
| `OwnMusicAI/YuE2-3B/yue2_csharp/` – prompt protocol, sampling | The `yue2` inference package shipped with [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | see the model repository |
| `OwnMusicAI/SheetSage2/` – tokenizer, grammar, notation, exports | The reference implementation in [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | see the model repository |

## Model weights (downloaded at runtime, not in this repository)

| Model | License |
|---|---|
| [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | CC BY-NC 4.0 |
| [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | CC BY-NC 4.0 |
| [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | CC BY-NC 4.0 |

## Dependencies

| Package | License |
|---|---|
| [Candle](https://github.com/huggingface/candle) (`candle-core`, `candle-nn`) | MIT / Apache-2.0 |
| [OwnAudioSharp](https://github.com/ModernMube/OwnAudioSharp) (`OwnAudioSharp`, `.Basic`, `.Midi`) | MIT |
| [Avalonia](https://github.com/AvaloniaUI/Avalonia) | MIT |
| [CommunityToolkit.Mvvm](https://github.com/CommunityToolkit/dotnet) | MIT |
| [Microsoft.ML.Tokenizers](https://github.com/dotnet/machinelearning) | MIT |
| [Material.Icons.Avalonia](https://github.com/SKProCH/Material.Icons) | MIT |
| [Xaml.Behaviors](https://github.com/wieslawsoltes/Xaml.Behaviors) | MIT |

Other Rust crates and NuGet packages keep their respective licenses (see `Cargo.lock` and the NuGet
package metadata).
