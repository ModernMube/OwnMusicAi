"""Dump FP32 PyTorch reference tensors for the Candle parity check.

    uv venv tools/.venv --python 3.11 && uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
    tools/.venv/bin/python tools/make_reference.py <mono 24 kHz wav> --out reference

Uses the official SheetSage2 remote code (MERT-v2 parent from the Hugging Face cache) on CPU.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model-dir", type=Path, default=root)
    parser.add_argument("--out", type=Path, default=Path("reference"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(args.model_dir))
    save = lambda name, value: np.save(args.out / f"{name}.npy", np.ascontiguousarray(value))

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True).eval()
    generation = importlib.import_module(type(model).__module__.replace("modeling_sheetsage2", "generation_sheetsage2"))

    rate, waveform = wavfile.read(args.audio)
    if rate != 24000 or waveform.ndim != 1 or waveform.dtype != np.float32:
        raise SystemExit("expected a mono 24 kHz float32 wav")
    audio = torch.from_numpy(waveform[: round(args.seconds * rate)].copy())
    save("audio", audio.numpy())

    padded, _ = model._prepare_audio(audio[None])
    save("mel", model.encoder.feature_extractor(padded)[0].numpy())
    memory = model.get_audio_features(audio[None]).encoder_last_hidden_state
    save("memory", memory[0].numpy())

    # a short greedy run gives realistic decoder inputs; then logits for prefill + each step
    tokenizer = model.tokenizer
    prefix = tokenizer.prompt_prefix(generation.FULL_TASK_PROMPTS)
    tokens = generation.constrained_prompt_generate(model, audio[None], generation.FULL_TASK_PROMPTS,
                                                    len(prefix) + args.steps, memory=memory, autocast_dtype=None)
    tokens = tokens.tolist()[: len(prefix) + args.steps]
    logits, cache = model.decode(memory, torch.tensor([prefix]), use_cache=True)
    rows = [logits[0, -1]]
    for token in tokens[len(prefix):]:
        logits, cache = model.decode(memory, torch.tensor([[token]]), use_cache=True, past_key_values=cache)
        rows.append(logits[0, -1])
    save("decoder_tokens", np.array(tokens, np.uint32))
    save("decoder_prefix_len", np.array([len(prefix)], np.uint32))
    save("decoder_logits", torch.stack(rows).numpy())
    print(f"reference written to {args.out} ({len(tokens)} decoder tokens)")


if __name__ == "__main__":
    main()
