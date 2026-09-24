"""Dump FP32 PyTorch reference tensors for the Candle parity check.

    uv venv tools/.venv --python 3.12 && uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
    tools/.venv/bin/python tools/make_reference.py --vae-dir <YuE2-Vae dir> --out reference

Uses the official yue2 package (modeling_yue2, nar.CachedNAR, modeling_vae) on CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=root)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reference"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    save = lambda name, value: np.save(args.out / f"{name}.npy", np.ascontiguousarray(value))

    from yue2.modeling_yue2 import StaticKVCache, YuE2ForCausalLM
    from yue2.nar import CachedNAR, Chunk
    from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer(args.model_dir / "qwen.tiktoken")
    prefix = token_prefixes(SongRequest(style="City pop, groovy bass, female vocal",
                                        lyrics="[Verse]\nneon lights are calling\n", cot="off"), tokenizer)
    model = YuE2ForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32, low_cpu_mem_usage=True,
                                            local_files_only=True).eval()
    cfg = model.config

    # AR: prefill, one decode step, and a second step to catch cache offsets
    cache = StaticKVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, len(prefix) + 4,
                          cfg.head_dim, torch.float32, "cpu")
    logits = model(torch.tensor([prefix]), past_key_values=cache, logits_to_keep=1).logits[0, -1]
    steps = [CODEC_OFFSET + 17, CODEC_OFFSET + 4242]
    step_logits = [model(torch.tensor([[t]]), past_key_values=cache, logits_to_keep=1).logits[0, -1] for t in steps]
    save("prefix", np.array(prefix, np.uint32))
    save("ar_steps", np.array(steps, np.uint32))
    save("ar_logits_prefill", logits.numpy())
    save("ar_logits_step", torch.stack(step_logits).numpy())

    codec = [5, 900, 32767, 12, 64, 1, 77, 3, 4096, 8191, 250, 31000]
    tokens = prefix + [CODEC_OFFSET + c for c in codec] + [MUSIC_END]
    noise = torch.randn(len(codec), 64, generator=torch.Generator().manual_seed(7))
    engine = CachedNAR(model, Chunk(tokens, noise))
    save("nar_tokens", np.array(tokens, np.uint32))
    save("nar_state", noise.numpy())
    save("nar_velocity", engine.velocity(noise, 1.3).numpy())
    save("nar_velocity_t20", engine.velocity(noise, 20.0).numpy())
    engine.close()
    del model, cache

    from yue2.modeling_vae import YuE2VAE
    vae = YuE2VAE.from_pretrained(args.vae_dir, decoder_only=True, device="cpu")
    latent = torch.randn(1, 64, 24, generator=torch.Generator().manual_seed(3))
    save("vae_latent", latent[0].T.contiguous().numpy())
    save("vae_audio", vae.decode(latent)[0].numpy())
    print(f"reference written to {args.out} ({len(prefix)} prefix tokens)")


if __name__ == "__main__":
    main()
