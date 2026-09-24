"""Whole-song FP32 transcription with the official PyTorch SheetSage2, for comparing the C# pipeline.

    tools/.venv/bin/python tools/transcribe_reference.py song_24k.wav --out ref/song

Writes the standard output folder (score.abc, events.json, *.lab, tokens.json, ...) plus
score.melody_only.abc built from the same decoded events.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model-dir", type=Path, default=root)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.model_dir))

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True).eval()
    progress = lambda p: p["stage"] in ("encoding", "window_complete") and print(p, flush=True)
    result = model.transcribe(str(args.audio), output_dir=args.out, dtype="fp32", progress=progress)

    exports = importlib.import_module(type(model).__module__.replace("modeling_sheetsage2", "exports_sheetsage2"))
    decoded = dict(schema_version="v1", prompts=result["prompts"], events=result["events"], has_eos=True)
    melody = exports.export_result(decoded, model.tokenizer, None, result["duration_seconds"], melody_only=True)
    (args.out / "score.melody_only.abc").write_text(melody["payload"]["abc"] or "", encoding="utf-8")
    print(f"done: {len(result['events'])} events, abc_error={result.get('abc_error')}")


if __name__ == "__main__":
    main()
