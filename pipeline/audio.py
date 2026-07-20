"""Stage 1: audio extraction — legacy behavior + AEGIS Phase 1 ingest façade.

`extract_audio` is preserved as a compatibility façade: it produces the exact
legacy artifacts (mono 16 kHz + mono 48 kHz pcm_s16le WAVs, same ffmpeg
commands, same filenames, same returned keys) so every downstream stage and
the dashboard work unchanged — and additionally runs the Phase 1 ingest
(immutable hashing, media probe, per-channel/mid/side preservation,
condition-vector diagnostics), writing `ingest_manifest.json` next to the
legacy files. Ingest failures degrade with a logged warning and a
`ingest_manifest` of None; they never break the legacy path. Disable via
config `ingest.enabled: false`.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system ffmpeg
install is required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import log


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fall back to a system ffmpeg if the user has one on PATH.
        return "ffmpeg"


def extract_audio_legacy(input_path: Path, out_dir: Path, sample_rate: int,
                         hq_sample_rate: int = 48000) -> dict:
    """The UNCHANGED legacy extraction (byte-identical commands and outputs
    to the Phase 0 checkpoint):

    - work_wav:  mono @ sample_rate (16k) for the speech models
    - hq_wav:    mono @ hq_sample_rate (48k) for full-band enhancement

    Returns a dict of paths. Works for video *or* audio inputs (including
    browser MediaRecorder .webm/.mp4 blobs).
    """
    ffmpeg = _ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_wav = out_dir / "audio_16k_mono.wav"
    hq_wav = out_dir / "audio_48k_mono.wav"

    for target, args in (
        (work_wav, ["-ac", "1", "-ar", str(sample_rate)]),
        (hq_wav, ["-ac", "1", "-ar", str(hq_sample_rate)]),
    ):
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn",
               "-acodec", "pcm_s16le", *args, str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"ffmpeg failed to extract audio to {target.name}.\n"
                f"{proc.stderr[-800:]}"
            )

    log(f"Extracted audio → {work_wav.name}, {hq_wav.name}")
    return {"work_wav": work_wav, "hq_wav": hq_wav}


def extract_audio(input_path: Path, out_dir: Path, sample_rate: int,
                  hq_sample_rate: int = 48000, cfg: dict | None = None) -> dict:
    """Compatibility façade: legacy outputs + additive Phase 1 ingest.

    Returned dict keeps the legacy keys ('work_wav', 'hq_wav') that every
    downstream consumer expects, plus 'ingest_manifest' (IngestManifest or
    None) for new-code access to channel assets and diagnostics.
    """
    out = extract_audio_legacy(input_path, out_dir, sample_rate, hq_sample_rate)

    ingest_cfg = (cfg or {}).get("ingest", {}) if cfg else {}
    manifest = None
    status: dict = {"state": "disabled", "ok": False, "stages_completed": [],
                    "failures": [], "failure_reason": None,
                    "store_relative_dir": None}

    if ingest_cfg.get("enabled", True):
        try:
            from pipeline.ingest.ingest import ingest_file
            manifest = ingest_file(Path(input_path), Path(out_dir), cfg)
            status = {
                "state": "ok" if manifest.ok else "degraded",
                "ok": manifest.ok,
                "stages_completed": list(manifest.stages_completed),
                "failures": [f.model_dump() for f in manifest.failures],
                "failure_reason": manifest.failure_reason,
                "store_relative_dir": manifest.store_relative_dir,
            }
            n_assets = len(manifest.derived)
            n_cond = len(manifest.condition_vector.conditions) \
                if manifest.condition_vector else 0
            log(f"Ingest [{status['state']}]: {n_assets} channel asset(s), "
                f"{n_cond} condition check(s), canonical store "
                f"{manifest.store_relative_dir}"
                + (f" [duplicate of {manifest.source.duplicate_of}]"
                   if manifest.source.duplicate_of else ""))
        except Exception as e:
            # ingest_file is designed not to raise; this is the last-resort
            # guard so the legacy path can never be taken down by ingest.
            status = {
                "state": "failed", "ok": False, "stages_completed": [],
                "failures": [{"stage": "unknown",
                              "exception_type": type(e).__name__,
                              "message": " ".join(str(e).split())[:300],
                              "occurred_utc": None, "source_content_id": None,
                              "recoverable": False, "legacy_continued": True,
                              "warnings": []}],
                "failure_reason": f"unexpected ingest error: {type(e).__name__}",
                "store_relative_dir": None,
            }
            log(f"WARNING: evidence-grade ingest did NOT complete "
                f"({type(e).__name__}); legacy outputs intact.")

    out["ingest_manifest"] = manifest
    out["ingest_status"] = status          # ALWAYS present and machine-readable
    return out


def duration_seconds(wav_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.frames / float(info.samplerate)
