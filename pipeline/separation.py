"""Stage 2: separate voice from noise.

Default method is ClearVoice MossFormer2_SE_48K — a full-band (48 kHz)
speech enhancer, so the isolated voice keeps sibilance/air above 8 kHz and
the noise track is the SAME model's residual (original - voice), which means
the voice does not bleed into noise.wav the way the old mixed-algorithm
derivation did.

Long files are enhanced in crossfaded chunks so a 6 GB GPU never OOMs.

Methods (config: separation.method):
  clearvoice    MossFormer2_SE_48K full-band enhancement  [default]
  deepfilternet DeepFilterNet3 48 kHz enhancement (lighter)
  enhance       SpeechBrain SepFormer 16 kHz enhancement (legacy)
  noisereduce   spectral-gating denoise (light, no model download)
  none          passthrough

Outputs: voice.wav + noise.wav at the processing rate, plus voice_16k.wav
for the downstream 16 kHz models (diarization / ASR / emotion).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .utils import log, free_cuda

HQ_SR = 48000
SR16 = 16000


def _save(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr)


def _load(path: Path, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def separate(paths: dict, out_dir: Path, cfg: dict, device: str,
             models=None) -> dict:
    if models is None:
        from .models import MANAGER as models

    voice_path = out_dir / "voice.wav"
    noise_path = out_dir / "noise.wav"
    voice16_path = out_dir / "voice_16k.wav"
    method = str(cfg.get("method", "clearvoice")).lower()
    if not cfg.get("enabled", True):
        method = "none"

    # Ordered fallback chain starting at the requested method.
    chain = ["clearvoice", "deepfilternet", "enhance", "noisereduce", "none"]
    if method in chain:
        chain = chain[chain.index(method):]
    else:
        log(f"Unknown separation.method '{method}', using clearvoice.")

    voice = None
    proc_sr = HQ_SR
    used = "none"
    for m in chain:
        try:
            if m == "clearvoice":
                voice = _clearvoice_enhance(paths["hq_wav"], out_dir, cfg, models)
                proc_sr = CV_MODEL_SR.get(
                    cfg.get("clearvoice_model", "MossFormer2_SE_48K"), HQ_SR)
            elif m == "deepfilternet":
                voice = _deepfilter_enhance(paths["hq_wav"], cfg, models)
                proc_sr = HQ_SR
            elif m == "enhance":
                voice = _speechbrain_enhance(paths["work_wav"], cfg, device, models)
                proc_sr = SR16
            elif m == "noisereduce":
                voice = _noisereduce(_load(paths["work_wav"], SR16), SR16)
                proc_sr = SR16
            else:  # none
                voice = _load(paths["hq_wav"], HQ_SR)
                proc_sr = HQ_SR
            used = m if m == method else f"{m} (fallback)"
            break
        except Exception as e:
            log(f"WARNING: '{m}' separation failed ({e}); trying next fallback.")
            voice = None

    original = _load(paths["hq_wav"] if proc_sr == HQ_SR else paths["work_wav"], proc_sr)
    if voice is None:
        voice = original.copy()
        used = "none (fallback)"

    # Noise = the SAME enhancer's residual, computed BEFORE any loudness
    # processing so the subtraction is scale-consistent. Written at natural
    # level — normalizing it just amplifies whatever leaked.
    m = min(len(original), len(voice))
    noise = (original[:m] - voice[:m]).astype(np.float32)

    voice = enhance_voice(voice, proc_sr, cfg)

    _save(voice_path, voice, proc_sr)
    _save(noise_path, noise, proc_sr)
    voice16 = _resample(voice, proc_sr, SR16)
    _save(voice16_path, voice16, SR16)
    free_cuda()
    return {"voice": voice_path, "noise": noise_path,
            "voice_16k": voice16_path, "method": used, "sample_rate": proc_sr}


# --------------------------------------------------------------- enhancers
# Native sample rate of each ClearVoice enhancement model.
CV_MODEL_SR = {
    "MossFormer2_SE_48K": 48000,   # full-band, best fidelity
    "MossFormerGAN_SE_16K": 16000, # strongest denoiser (best on hard audio)
    "FRCRN_SE_16K": 16000,
}


def _clearvoice_enhance(hq_wav: Path, out_dir: Path, cfg: dict, models) -> np.ndarray:
    """ClearVoice enhancement, chunked with crossfaded overlap-add.
    Optional multi-pass (separation.passes) for heavily degraded audio."""
    model_name = cfg.get("clearvoice_model", "MossFormer2_SE_48K")
    sr = CV_MODEL_SR.get(model_name, 48000)

    def loader():
        from clearvoice import ClearVoice
        return ClearVoice(task="speech_enhancement", model_names=[model_name])

    cv = models.get(f"clearvoice:{model_name}", loader, "cuda")

    y = _load(hq_wav, sr)
    passes = max(1, int(cfg.get("passes", 1)))
    for i in range(passes):
        y = _cv_run_chunked(cv, y, sr, cfg, out_dir)
        if passes > 1 and i < passes - 1:
            log(f"  (enhancement pass {i + 1}/{passes} done)")
    return y


def _cv_run_chunked(cv, y: np.ndarray, HQ_SR: int, cfg: dict,
                    out_dir: Path) -> np.ndarray:
    n = len(y)
    chunk = max(int(float(cfg.get("chunk_sec", 30)) * HQ_SR), HQ_SR)
    overlap = min(int(float(cfg.get("chunk_overlap_sec", 2)) * HQ_SR), chunk // 4)
    step = chunk - overlap

    tmp_dir = out_dir / "_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def run(seg: np.ndarray) -> np.ndarray:
        tmp = tmp_dir / "chunk.wav"
        _save(tmp, seg, HQ_SR)
        out = cv(input_path=str(tmp), online_write=False)
        out = np.squeeze(np.asarray(out)).astype(np.float32)
        if out.ndim > 1:
            out = out.mean(axis=0)
        if np.max(np.abs(out)) > 2.0:      # int16-scaled output guard
            out = out / 32768.0
        # Align to input length (models can trim a few samples).
        if len(out) < len(seg):
            out = np.pad(out, (0, len(seg) - len(out)))
        return out[:len(seg)]

    try:
        if n <= chunk + overlap:
            return run(y)

        acc = np.zeros(n, dtype=np.float64)
        wsum = np.zeros(n, dtype=np.float64)
        starts = list(range(0, n, step))
        for i, a in enumerate(starts):
            b = min(n, a + chunk)
            if b - a < int(0.25 * HQ_SR) and i > 0:
                break
            seg = run(y[a:b])
            w = np.ones(b - a, dtype=np.float64)
            if a > 0:
                ramp = min(overlap, b - a)
                w[:ramp] = np.linspace(0.0, 1.0, ramp)
            if b < n:
                ramp = min(overlap, b - a)
                w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp))
            acc[a:b] += seg * w
            wsum[a:b] += w
            if b >= n:
                break
        wsum[wsum == 0] = 1.0
        return (acc / wsum).astype(np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _deepfilter_enhance(hq_wav: Path, cfg: dict, models) -> np.ndarray:
    """DeepFilterNet3 full-band (48 kHz) enhancement."""
    import torch

    def loader():
        from df.enhance import init_df
        model, state, _ = init_df()
        return (model, state)

    model, state = models.get("deepfilternet", loader, "cuda")
    from df.enhance import enhance

    y = _load(hq_wav, state.sr() if hasattr(state, "sr") else HQ_SR)
    out = enhance(model, state, torch.from_numpy(y).unsqueeze(0))
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def _speechbrain_enhance(work_wav: Path, cfg: dict, device: str, models) -> np.ndarray:
    """Legacy 16 kHz SepFormer enhancement (fallback only)."""
    import torch
    import torchaudio
    from .utils import guard_speechbrain_lazy
    guard_speechbrain_lazy()   # incl. Windows no-symlink patch

    model_id = cfg.get("enhance_model", "speechbrain/sepformer-wham16k-enhancement")

    def loader():
        from speechbrain.inference.separation import SepformerSeparation
        savedir = Path.home() / ".cache" / "voice_isolator" / model_id.split("/")[-1]
        return SepformerSeparation.from_hparams(
            source=model_id, savedir=str(savedir),
            run_opts={"device": device})

    model = models.get(f"sepformer:{model_id}", loader, device)

    sig, sr = torchaudio.load(str(work_wav))
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != SR16:
        sig = torchaudio.functional.resample(sig, sr, SR16)
    if device == "cuda":
        sig = sig.to("cuda")

    with torch.no_grad():
        est = model.separate_batch(sig)          # (batch, time, n_src)
    return est[..., 0].squeeze(0).detach().cpu().numpy().astype(np.float32)


def _noisereduce(y: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=y, sr=sr, stationary=False,
                           prop_decrease=0.9).astype(np.float32)


# --------------------------------------------------------------- voice cleanup
def enhance_voice(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Post-separation cleanup: high-pass, EBU R128 loudness, VAD-guided gain.

    Order matters: boosting AFTER denoising raises the (already cleaned)
    voice, not the noise floor — which is what recovers quiet speakers.
    """
    if not cfg.get("enhance_voice", True) or y.size == 0:
        return y

    hp = float(cfg.get("highpass_hz", 80))
    if hp > 0:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, hp, btype="highpass", fs=sr, output="sos")
            y = sosfilt(sos, y).astype(np.float32)
        except Exception:
            pass

    y = _loudness_normalize(y, sr, cfg)

    if cfg.get("vad_gain", True):
        try:
            y = _vad_gain(y, sr, float(cfg.get("vad_gain_max_db", 9.0)))
        except Exception as e:
            log(f"  (vad gain skipped: {e})")

    return _apply_ceiling(y, float(cfg.get("peak_ceiling_dbfs", -1.0)))


def _loudness_normalize(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    target = float(cfg.get("loudness_lufs", -18.0))
    try:
        import pyloudnorm as pyln
        if len(y) < int(0.5 * sr):
            raise ValueError("clip too short for LUFS metering")
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if not np.isfinite(loudness):
            raise ValueError("non-finite loudness")
        y = (y * (10 ** ((target - loudness) / 20))).astype(np.float32)
    except Exception:
        # RMS fallback for very short / silent clips.
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        y = (y * ((10 ** (target / 20)) / rms)).astype(np.float32)
    return y


def _apply_ceiling(y: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-9
    ceiling = 10 ** (ceiling_dbfs / 20)
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def _vad_gain(y: np.ndarray, sr: int, max_db: float) -> np.ndarray:
    """Boost speech regions that are much quieter than the clip's median
    speech level (bounded, with short fades). Runs Silero VAD at 16 kHz."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    y16 = _resample(y, sr, SR16)
    regions = get_speech_timestamps(y16, VadOptions(min_silence_duration_ms=300))
    if len(regions) < 2:
        return y

    def rms_db(seg: np.ndarray) -> float:
        return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)

    scale = sr / SR16
    spans = []
    for r in regions:
        a, b = int(r["start"] * scale), min(len(y), int(r["end"] * scale))
        if b - a > int(0.2 * sr):
            spans.append((a, b, rms_db(y[a:b])))
    if len(spans) < 2:
        return y

    median = float(np.median([lv for _, _, lv in spans]))
    fade = int(0.05 * sr)
    out = y.copy()
    boosted = 0
    for a, b, lv in spans:
        deficit = median - lv
        if deficit <= 12.0:
            continue
        gain_db = min(deficit, max_db)
        g = 10 ** (gain_db / 20)
        env = np.full(b - a, g, dtype=np.float32)
        f = min(fade, (b - a) // 2)
        if f > 0:
            env[:f] = np.linspace(1.0, g, f)
            env[-f:] = np.linspace(g, 1.0, f)
        out[a:b] = y[a:b] * env
        boosted += 1
    if boosted:
        log(f"  (vad gain boosted {boosted} quiet speech region(s))")
    return out
