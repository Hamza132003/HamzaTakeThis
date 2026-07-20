"""Quick environment check: reports which pipeline stages are ready.

Run:  python check_env.py
"""
from __future__ import annotations

import importlib


CHECKS = [
    ("ffmpeg (bundled)", "imageio_ffmpeg", "audio extraction"),
    ("numpy", "numpy", "core"),
    ("librosa", "librosa", "audio I/O"),
    ("torch", "torch", "deep-learning backbone"),
    ("clearvoice", "clearvoice", "voice/noise separation (primary, 48 kHz)"),
    ("deepfilternet", "df", "separation fallback (optional)"),
    ("speechbrain", "speechbrain", "separation fallback + speaker matching"),
    ("noisereduce", "noisereduce", "separation last-resort fallback"),
    ("pyloudnorm", "pyloudnorm", "loudness normalization (EBU R128)"),
    ("panns_inference", "panns_inference", "noise identification"),
    # pyannote is NOT checked here: diarization runs in an isolated worker
    # environment (numpy 2), not in this one (numpy 1.26 for ClearVoice).
    # `python -m aegis doctor` reports the worker environment separately.
    ("faster_whisper", "faster_whisper", "transcription"),
    ("transformers", "transformers", "translation + emotion"),
    ("flask", "flask", "web dashboard"),
]


def main() -> None:
    print("Voice Isolator - environment check\n" + "-" * 40)
    ok = 0
    for name, mod, purpose in CHECKS:
        try:
            importlib.import_module(mod)
            print(f"  [ OK ] {name:<20} {purpose}")
            ok += 1
        except Exception as e:
            print(f"  [MISS] {name:<20} {purpose}  ->  {type(e).__name__}")

    print("-" * 40)
    print(f"{ok}/{len(CHECKS)} components available.")

    # GPU status + torch-build compatibility (fails loudly on mismatch: a build
    # without this GPU's arch reports cuda "available" but no kernel can run).
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB)")
            try:
                from aegis.gpu import check_torch_cuda
                probe = check_torch_cuda()
                status = "OK" if probe["compatible"] else "FAIL"
                print(f"GPU/torch compatibility: [{status}] {probe['reason']}")
                if not probe["compatible"]:
                    print("  → Do NOT run the pipeline. See docs/HARDWARE_SETUP.md "
                          "or run: python -m aegis doctor")
            except ImportError:
                pass
        else:
            print("GPU: not available (will run on CPU).")
    except Exception:
        print("GPU: torch not installed.")

    import os
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    print(f"HF token: {'set' if tok else 'NOT set (diarization will use fallback)'}")
    print("Diarization runs in a SEPARATE worker environment - run "
          "'python -m aegis doctor' to check it.")


if __name__ == "__main__":
    main()
