"""Voice Isolator - local, offline speech/noise analysis pipeline (CLI).

Usage:
    python main.py path/to/recording.mp4
    python main.py clip.wav --language fa
    python main.py clip.wav --device cpu --outdir results

Runs entirely on your machine. See README.md for setup and the one-time
HuggingFace token needed for speaker diarization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from pipeline.runner import process_file
from pipeline.utils import log, resolve_device


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local voice isolator + analyser")
    ap.add_argument("input", help="path to a video or audio file")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--outdir", default=None, help="override output_dir")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--language", default=None,
                    choices=["auto", "fa", "he", "ar", "en"],
                    help="source language for the whole clip (ar = transcribe only)")
    ap.add_argument("--hf-token", default=None, help="HuggingFace token for diarization")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="stages to disable: separation noise diarization "
                         "transcription translation emotion")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        log(f"ERROR: input not found: {in_path}")
        return 2

    cfg = load_config(Path(args.config))
    if args.device:
        cfg["device"] = args.device
    if args.outdir:
        cfg["output_dir"] = args.outdir
    if args.language:
        cfg.setdefault("transcription", {})["language"] = args.language
    if args.hf_token:
        cfg.setdefault("diarization", {})["hf_token"] = args.hf_token
    for s in args.skip:
        cfg.setdefault(s, {})["enabled"] = False

    device = resolve_device(cfg.get("device", "auto"))
    log(f"Input: {in_path.name} | device: {device} | "
        f"language: {cfg['transcription'].get('language', 'auto')}")

    rep = process_file(in_path, cfg, device)
    log(f"\n✅ Report: {Path(cfg.get('output_dir','outputs')) / in_path.stem / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
