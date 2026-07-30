"""Run the same protected diagnostic pipeline from inside the container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app import decode_pcm16_wav, run_diagnostics, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, help="16 kHz mono signed PCM16 WAV")
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--ayah", type=int, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--qari", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model_dir:
        os.environ["MODEL_DIR"] = str(args.model_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
    runtime.load()
    audio = decode_pcm16_wav(args.wav.read_bytes())
    result = run_diagnostics(
        audio,
        surah=args.surah,
        ayah=args.ayah,
        reference_text=args.reference,
        source=args.source,
        qari=args.qari,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
