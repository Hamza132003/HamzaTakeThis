"""Deterministic signal diagnostics → ConditionVector (Phase 1).

Every formula, unit, window size, threshold, failure condition and known
false-positive/negative is documented in docs/DIAGNOSTICS.md. All constants
come from thresholds.THRESHOLDS (versioned). Nothing here modifies audio, uses
a model, or claims ground truth it cannot have: quantile SNR values are
labelled `proxy`, band/hum decisions `estimated`. No LLM assigns conditions.

Numerical safety: input converted to float64 for accumulations; NaN/Inf
counted then sanitized to 0 for downstream math; every divisor is guarded;
long files (> max_full_analysis_sec) are analysed via three deterministic
windows (start/middle/end) and flagged analysis_mode="windowed-sample".
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from pipeline.contracts import (
    CategoricalCondition,
    ChannelMetadata,
    ConditionVector,
    DiagnosticMeasurement,
)

from .thresholds import THRESHOLDS as T
from .thresholds import THRESHOLDS_VERSION

MeasurementKind = Literal["measured", "estimated", "proxy"]

_EPS = 1e-12


def _m(name: str, value, units: str, method: str,
       kind: MeasurementKind = "measured",
       valid: bool = True, reason: str | None = None,
       window: str | None = None) -> DiagnosticMeasurement:
    v = None
    if valid and value is not None:
        v = float(np.clip(value, -1e12, 1e12))
        if not np.isfinite(v):
            valid, v, reason = False, None, "non-finite result"
    return DiagnosticMeasurement(name=name, value=v, units=units, method=method,
                                 kind=kind, valid=valid, reason=reason,
                                 window=window)


# --------------------------------------------------------------- primitives
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + _EPS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, length)] of consecutive True runs."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    # Cast to Python ints: numpy integer scalars leak into the declared
    # tuple[int, int] contract and are rejected by stricter numpy stubs.
    starts = [int(i) for i in np.flatnonzero(d == 1) + 1]
    ends = [int(i) for i in np.flatnonzero(d == -1) + 1]
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(int(len(mask)))
    return [(s, e - s) for s, e in zip(starts, ends)]


def _frame_rms_db(x: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * T["snr_frame_ms"] / 1000.0))
    usable = (len(x) // n) * n
    if usable == 0:
        return np.array([])
    frames = x[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + _EPS)
    return 20.0 * np.log10(rms + _EPS)


def _spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean magnitude-squared spectrum via Hann STFT → (freqs, psd, n_frames)."""
    nfft, hop = int(T["fft_size"]), int(T["hop_size"])
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    win = np.hanning(nfft)
    n_frames = 1 + (len(x) - nfft) // hop
    acc = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + nfft] * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
    psd = acc / max(n_frames, 1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    return freqs, psd, n_frames


# ------------------------------------------------------------------- blocks
def _basic(x: np.ndarray, sr: int, out: list, nonfinite: int) -> dict:
    n = len(x)
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = _rms(x) if n else 0.0
    dc = float(np.mean(x, dtype=np.float64)) if n else 0.0
    zc = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8))))) if n > 1 else 0.0
    silent = float(np.mean(np.abs(x) < T["silence_level"])) if n else 1.0
    near_silent = float(np.mean(np.abs(x) < T["near_silence_level"])) if n else 1.0
    fr = _frame_rms_db(x, sr)
    dyn = float(np.percentile(fr, 95) - np.percentile(fr, 5)) if fr.size else 0.0
    out += [
        _m("duration", n / sr, "s", "N/sr"),
        _m("sample_count", n, "samples", "len(x)"),
        _m("sample_rate", sr, "Hz", "from decode"),
        _m("peak_amplitude", peak, "linear", "max|x|"),
        _m("rms", rms, "linear", "sqrt(mean(x^2))"),
        _m("dc_offset", dc, "linear", "mean(x)"),
        _m("crest_factor", peak / (rms + _EPS), "ratio", "peak/RMS"),
        _m("zero_crossing_rate", zc, "crossings/sample", "mean|diff(signbit)|"),
        _m("silent_sample_ratio", silent, "ratio", f"|x|<{T['silence_level']}"),
        _m("near_silence_ratio", near_silent, "ratio", f"|x|<{T['near_silence_level']}"),
        _m("nonfinite_sample_count", nonfinite, "samples", "count(~isfinite)"),
        _m("dynamic_range_proxy", dyn, "dB", "p95-p5 of frame RMS dB",
           kind="proxy", window=f"frame={T['snr_frame_ms']}ms"),
    ]
    return {"peak": peak, "rms": rms, "dc": dc, "silent": silent}


def _clipping(x: np.ndarray, sr: int, out: list) -> dict:
    a = np.abs(x)
    pos = x >= T["clip_level"]
    neg = x <= -T["clip_level"]
    both = pos | neg
    runs = [r for r in _runs(both) if r[1] >= T["clip_run_min_samples"]]
    longest = max((r[1] for r in runs), default=0) / sr
    # flat-top: consecutive identical extreme samples
    near = a >= T["near_clip_level"]
    flat = 0
    if len(x) > 1:
        same = np.diff(x) == 0
        for s, ln in _runs(same & near[1:]):
            if ln + 1 >= T["flat_top_min_samples"]:
                flat += 1
    out += [
        _m("positive_clipping_ratio", float(np.mean(pos)), "ratio",
           f"x>=+{T['clip_level']}"),
        _m("negative_clipping_ratio", float(np.mean(neg)), "ratio",
           f"x<=-{T['clip_level']}"),
        _m("clipping_ratio", float(np.mean(both)), "ratio", "pos|neg"),
        _m("clipping_run_count", len(runs), "runs",
           f">={T['clip_run_min_samples']} consecutive clipped samples"),
        _m("max_clipping_run", longest, "s", "longest run / sr"),
        _m("near_clipping_ratio", float(np.mean(near)), "ratio",
           f"|x|>={T['near_clip_level']}"),
        _m("flat_top_runs", flat, "runs",
           f">={T['flat_top_min_samples']} identical extreme samples",
           kind="estimated"),
    ]
    return {"clip_ratio": float(np.mean(both)), "clip_runs": len(runs)}


def _dropouts(x: np.ndarray, sr: int, out: list) -> dict:
    lo = int(sr * T["dropout_min_ms"] / 1000.0)
    hi = int(sr * T["dropout_max_ms"] / 1000.0)
    quiet = np.abs(x) < T["dropout_level"]
    all_runs = _runs(quiet)
    drops = [r for r in all_runs if lo <= r[1] <= hi]
    exact_zero = [r for r in _runs(x == 0.0) if lo <= r[1] <= hi]
    jumps = int(np.sum(np.abs(np.diff(x)) > T["discontinuity_jump"])) if len(x) > 1 else 0
    longest = max((r[1] for r in drops), default=0) / sr
    out += [
        _m("dropout_ratio", sum(r[1] for r in drops) / max(len(x), 1), "ratio",
           f"|x|<{T['dropout_level']} runs of {T['dropout_min_ms']}–{T['dropout_max_ms']}ms",
           kind="estimated"),
        _m("dropout_run_count", len(drops), "runs", "same detector",
           kind="estimated"),
        _m("max_dropout", longest, "s", "longest qualifying run",
           kind="estimated"),
        _m("discontinuity_count", jumps, "events",
           f"|x[n]-x[n-1]|>{T['discontinuity_jump']}", kind="estimated"),
        _m("repeated_zero_runs", len(exact_zero), "runs",
           "exact-zero runs in dropout window (packet-loss-like)",
           kind="estimated"),
    ]
    return {"dropout_runs": len(drops)}


def _spectral(x: np.ndarray, sr: int, out: list) -> dict:
    freqs, psd, n_frames = _spectrum(x, sr)
    tot = float(np.sum(psd))
    win = f"fft={int(T['fft_size'])} hop={int(T['hop_size'])} hann, {n_frames} frames"
    if tot <= _EPS:
        out.append(_m("spectral_centroid", None, "Hz", "sum(f*P)/sum(P)",
                      valid=False, reason="zero spectral energy", window=win))
        return {"low_edge": 0.0, "high_edge": 0.0, "valid": False}
    centroid = float(np.sum(freqs * psd) / tot)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / tot))
    flatness = float(np.exp(np.mean(np.log(psd + _EPS))) / (np.mean(psd) + _EPS))
    cum = np.cumsum(psd) / tot
    rolloff = float(freqs[int(np.searchsorted(cum, 0.85))])
    low_edge = float(freqs[int(np.searchsorted(cum, T["band_edge_low_q"]))])
    high_edge = float(freqs[int(np.searchsorted(cum, T["band_edge_high_q"]))])
    bands = [(0, 300), (300, 3400), (3400, sr / 2)]
    ratios = []
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        ratios.append(float(np.sum(psd[sel]) / tot))
    rumble = float(np.sum(psd[freqs < T["rumble_hz"]]) / tot)
    out += [
        _m("spectral_centroid", centroid, "Hz", "sum(f*P)/sum(P)", window=win),
        _m("spectral_bandwidth", bandwidth, "Hz", "sqrt(sum((f-c)^2*P)/sum(P))",
           window=win),
        _m("spectral_flatness", flatness, "ratio", "geomean(P)/mean(P)", window=win),
        _m("spectral_rolloff_85", rolloff, "Hz", "85% cumulative PSD", window=win),
        _m("low_band_ratio", ratios[0], "ratio", "P(0-300Hz)/P", window=win),
        _m("mid_band_ratio", ratios[1], "ratio", "P(300-3400Hz)/P", window=win),
        _m("high_band_ratio", ratios[2], "ratio", "P(3400-Nyq)/P", window=win),
        _m("occupied_band_low_edge", low_edge, "Hz",
           f"{T['band_edge_low_q']:.0%} cumulative PSD", kind="estimated", window=win),
        _m("occupied_band_high_edge", high_edge, "Hz",
           f"{T['band_edge_high_q']:.0%} cumulative PSD", kind="estimated", window=win),
        _m("rumble_ratio", rumble, "ratio", f"P(<{T['rumble_hz']}Hz)/P",
           kind="estimated", window=win),
    ]
    return {"low_edge": low_edge, "high_edge": high_edge, "valid": True}


def _hum(x: np.ndarray, sr: int, out: list) -> dict:
    """Mains-hum scoring with a persistence check so voiced speech harmonics
    (which modulate over time) are less likely to masquerade as hum."""
    nfft, hop = int(T["fft_size"]), int(T["hop_size"])
    win = np.hanning(nfft)
    n_frames = max(1, 1 + (len(np.pad(x, (0, max(0, nfft - len(x))))) - nfft) // hop)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    scores = {}
    persist = {}
    for fund in (50.0, 60.0):
        frame_scores = []
        for i in range(min(n_frames, 200)):        # bounded, deterministic
            seg = x[i * hop:i * hop + nfft]
            if len(seg) < nfft:
                seg = np.pad(seg, (0, nfft - len(seg)))
            p = np.abs(np.fft.rfft(seg * win)) ** 2
            tp = float(np.sum(p))
            if tp <= _EPS:
                frame_scores.append(0.0)
                continue
            tone = 0.0
            for h in range(1, int(T["hum_harmonics"]) + 1):
                f0 = fund * h
                sel = np.abs(freqs - f0) <= T["hum_bin_bw_hz"]
                tone += float(np.sum(p[sel]))
            frame_scores.append(tone / tp)
        fs = np.asarray(frame_scores)
        scores[fund] = float(np.median(fs)) if fs.size else 0.0
        persist[fund] = float(np.mean(fs > T["hum_score_warn"] / 2)) if fs.size else 0.0
    dominant = 50.0 if scores[50.0] >= scores[60.0] else 60.0
    reliable = persist[dominant] >= T["hum_persistence_min"]
    out += [
        _m("hum_50hz_score", scores[50.0], "tonal fraction",
           f"median over frames of P(50Hz harmonics ±{T['hum_bin_bw_hz']}Hz)/P",
           kind="estimated"),
        _m("hum_60hz_score", scores[60.0], "tonal fraction", "same for 60Hz",
           kind="estimated"),
        _m("hum_harmonic_energy", max(scores.values()), "tonal fraction",
           "max of family scores", kind="estimated"),
        _m("hum_dominant_family", dominant, "Hz", "argmax family",
           kind="estimated"),
        _m("hum_persistence", persist[dominant], "fraction of frames",
           "frames with tonal fraction above half-threshold; guards against "
           "time-varying voiced-speech harmonics", kind="estimated"),
    ]
    return {"hum_score": max(scores.values()), "hum_reliable": reliable}


def _snr_proxies(x: np.ndarray, sr: int, out: list) -> dict:
    fr = _frame_rms_db(x, sr)
    win = f"frame={T['snr_frame_ms']}ms"
    if fr.size < 4:
        out.append(_m("snr_quantile_proxy", None, "dB", "q90-q10 frame RMS",
                      kind="proxy", valid=False,
                      reason="too few frames", window=win))
        return {"snr_proxy": None}
    sig = float(np.quantile(fr, T["snr_signal_q"]))
    noise = float(np.quantile(fr, T["snr_noise_q"]))
    k = max(1, int(0.3 * fr.size))
    seg = float(np.mean(np.sort(fr)[-k:]) - np.mean(np.sort(fr)[:k]))
    out += [
        _m("noise_floor_proxy", noise, "dBFS",
           f"q{T['snr_noise_q']:.2f} of frame RMS dB", kind="proxy", window=win),
        _m("stationary_noise_proxy", float(np.std(np.sort(fr)[:k])), "dB",
           "std of quietest 30% frames (low = stationary)", kind="proxy",
           window=win),
        _m("snr_quantile_proxy", sig - noise, "dB",
           f"q{T['snr_signal_q']:.2f} - q{T['snr_noise_q']:.2f} frame RMS dB "
           "(NOT true SNR: no clean reference)", kind="proxy", window=win),
        _m("snr_segmental_proxy", seg, "dB",
           "mean(top 30%) - mean(bottom 30%) frame RMS dB (NOT true SNR)",
           kind="proxy", window=win),
    ]
    return {"snr_proxy": sig - noise}


def _channels(audio: np.ndarray, sr: int, out: list,
              ch_meta: list) -> dict:
    n_ch = audio.shape[0]
    for c in range(n_ch):
        x = audio[c]
        clip = float(np.mean(np.abs(x) >= T["clip_level"]))
        lo = int(sr * T["dropout_min_ms"] / 1000.0)
        hi = int(sr * T["dropout_max_ms"] / 1000.0)
        drops = [r for r in _runs(np.abs(x) < T["dropout_level"])
                 if lo <= r[1] <= hi]
        ch_meta.append(ChannelMetadata(index=c, rms=_rms(x),
                                       peak=float(np.max(np.abs(x))),
                                       clipping_ratio=clip,
                                       dropout_runs=len(drops)))
    if n_ch < 2:
        return {"imbalance_db": 0.0, "corr": None}
    a, b = audio[0], audio[1]
    ra, rb = _rms(a), _rms(b)
    imbalance = abs(20.0 * np.log10((ra + _EPS) / (rb + _EPS)))
    denom = float(np.std(a) * np.std(b))
    corr = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > _EPS else 0.0
    identical = bool(corr > T["identical_corr"]
                     and float(np.max(np.abs(a - b))) < T["identical_max_diff"])
    mid, side = (a + b) / 2.0, (a - b) / 2.0
    ms_ratio = (_rms(mid) + _EPS) / (_rms(side) + _EPS)
    # Delay estimate only when correlation supports it ("safely measurable").
    delay_ms = None
    reason = None
    max_lag = int(sr * T["delay_max_ms"] / 1000.0)
    if abs(corr) >= T["delay_min_corr"] and len(a) > 4 * max_lag:
        seg = slice(max_lag, min(len(a) - max_lag, max_lag + 10 * sr))
        ref = a[seg]
        xc = [float(np.dot(ref, b[seg.start - lag: seg.stop - lag]))
              for lag in range(-max_lag, max_lag + 1)]
        delay_ms = (int(np.argmax(np.abs(np.asarray(xc)))) - max_lag) / sr * 1000.0
    else:
        reason = "interchannel correlation too low for a safe delay estimate"
    out += [
        _m("channel_imbalance", imbalance, "dB", "|20log10(RMS_0/RMS_1)|"),
        _m("interchannel_correlation", corr, "Pearson r",
           "cov(ch0,ch1)/(std0*std1)"),
        _m("polarity_inversion_indicator", 1.0 if corr < T["polarity_corr"] else 0.0,
           "bool", f"r<{T['polarity_corr']}", kind="estimated"),
        _m("identical_channels_indicator", 1.0 if identical else 0.0, "bool",
           f"r>{T['identical_corr']} and max|diff|<{T['identical_max_diff']}"),
        _m("mid_side_energy_ratio", ms_ratio, "ratio", "RMS(mid)/RMS(side)"),
        _m("channel_delay_estimate", delay_ms, "ms",
           f"argmax cross-correlation ±{T['delay_max_ms']}ms",
           kind="estimated", valid=delay_ms is not None, reason=reason),
    ]
    return {"imbalance_db": float(imbalance), "corr": corr}


def _conditions(stats: dict, out: list[CategoricalCondition]) -> None:
    def cond(name, present, severity, reason, support, indeterminate=False):
        out.append(CategoricalCondition(
            condition=name,
            status="indeterminate" if indeterminate else
                   ("present" if present else "absent"),
            severity=float(np.clip(severity, 0.0, 1.0)) if present else 0.0,
            reason=reason, supporting_measurements=support,
            thresholds_version=THRESHOLDS_VERSION))

    sp = stats.get("spectral", {})
    if sp.get("valid"):
        nb = (sp["low_edge"] > T["narrowband_low_hz"]
              and sp["high_edge"] < T["narrowband_high_hz"])
        cond("probable_narrowband_radio", nb,
             0.8 if nb else 0.0,
             f"occupied band {sp['low_edge']:.0f}–{sp['high_edge']:.0f} Hz vs "
             f"window {T['narrowband_low_hz']:.0f}–{T['narrowband_high_hz']:.0f} Hz",
             ["occupied_band_low_edge", "occupied_band_high_edge"])
        hf = sp["high_edge"] < T["hf_loss_hz"]
        hf_sev = min(1.0, T["hf_loss_hz"] / max(sp["high_edge"], 1.0) - 1.0)
        cond("probable_severe_lowpass", hf, hf_sev,
             f"upper occupied edge {sp['high_edge']:.0f} Hz < {T['hf_loss_hz']:.0f} Hz",
             ["occupied_band_high_edge"])
        hp = sp["low_edge"] > 300.0 and not nb
        cond("probable_severe_highpass", hp, 0.5,
             f"lower occupied edge {sp['low_edge']:.0f} Hz", ["occupied_band_low_edge"])
    else:
        cond("probable_narrowband_radio", False, 0.0,
             "no spectral energy — band edges indeterminate",
             ["occupied_band_low_edge"], indeterminate=True)

    clip = stats["clip"]["clip_ratio"]
    cond("probable_clipping", clip > T["clip_ratio_warn"],
         min(1.0, clip / (10 * T["clip_ratio_warn"])),
         f"clipping ratio {clip:.5f} vs {T['clip_ratio_warn']}",
         ["clipping_ratio", "clipping_run_count", "flat_top_runs"])

    silent = stats["basic"]["silent"]
    cond("probable_silence_no_information", silent > T["silence_ratio_warn"],
         silent, f"silent-sample ratio {silent:.3f}",
         ["silent_sample_ratio", "rms"])

    dc = abs(stats["basic"]["dc"])
    cond("probable_dc_offset", dc > T["dc_offset_warn"], min(1.0, dc * 10),
         f"|mean(x)|={dc:.4f} vs {T['dc_offset_warn']}", ["dc_offset"])

    hum = stats["hum"]
    cond("probable_mains_hum", hum["hum_score"] > T["hum_score_warn"]
         and hum["hum_reliable"],
         hum["hum_score"],
         f"tonal fraction {hum['hum_score']:.3f}, persistence "
         f"{'ok' if hum['hum_reliable'] else 'insufficient'}",
         ["hum_50hz_score", "hum_60hz_score", "hum_persistence"],
         indeterminate=hum["hum_score"] > T["hum_score_warn"]
         and not hum["hum_reliable"])

    snr = stats["snr"]["snr_proxy"]
    if snr is None:
        cond("probable_severe_noise", False, 0.0, "too few frames",
             ["snr_quantile_proxy"], indeterminate=True)
    else:
        cond("probable_severe_noise", snr < T["snr_low_warn_db"],
             min(1.0, max(0.0, (T["snr_low_warn_db"] - snr) / T["snr_low_warn_db"])),
             f"quantile SNR proxy {snr:.1f} dB (< {T['snr_low_warn_db']} dB); "
             "PROXY, not true SNR", ["snr_quantile_proxy", "noise_floor_proxy"])

    imb = stats["ch"]["imbalance_db"]
    cond("probable_channel_imbalance", imb > T["imbalance_warn_db"],
         min(1.0, imb / 24.0), f"imbalance {imb:.1f} dB",
         ["channel_imbalance"])

    drops = stats["drop"]["dropout_runs"]
    cond("probable_dropout_corruption", drops > T["dropout_runs_warn"],
         min(1.0, drops / 20.0), f"{drops} dropout runs",
         ["dropout_run_count", "repeated_zero_runs"])


# ------------------------------------------------------------------- entry
def analyse(audio: np.ndarray, sr: int) -> ConditionVector:
    """audio: (C, N) float array (any float dtype), NOT modified."""
    audio = np.atleast_2d(np.asarray(audio))
    warnings: list[str] = []
    nonfinite = int(np.sum(~np.isfinite(audio)))

    # Window BEFORE the float64 copy so long files never spike RAM with a
    # full-length high-precision duplicate (bounded-memory requirement).
    dur = audio.shape[1] / float(sr)
    mode: Literal["full-file", "windowed-sample"] = "full-file"
    if dur > T["max_full_analysis_sec"]:
        mode = "windowed-sample"
        w = int(T["sample_window_sec"] * sr)
        n = audio.shape[1]
        idx = np.r_[0:w, n // 2 - w // 2: n // 2 + w // 2, n - w: n]
        audio = audio[:, np.clip(idx, 0, n - 1)]
        warnings.append(
            f"file >{T['max_full_analysis_sec']}s: diagnostics computed on 3 "
            f"deterministic windows (start/mid/end, {T['sample_window_sec']}s each)")

    audio = audio.astype(np.float64, copy=True)     # never mutates caller data
    if nonfinite:
        warnings.append(f"{nonfinite} non-finite samples sanitized to 0 for analysis")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    mono = audio.mean(axis=0)
    meas: list[DiagnosticMeasurement] = []
    ch_meta: list[ChannelMetadata] = []
    stats = {
        "basic": _basic(mono, sr, meas, nonfinite),
        "clip": _clipping(mono, sr, meas),
        "drop": _dropouts(mono, sr, meas),
        "spectral": _spectral(mono, sr, meas),
        "hum": _hum(mono, sr, meas),
        "snr": _snr_proxies(mono, sr, meas),
        "ch": _channels(audio, sr, meas, ch_meta),
    }
    conditions: list[CategoricalCondition] = []
    _conditions(stats, conditions)
    return ConditionVector(thresholds_version=THRESHOLDS_VERSION,
                           analysis_mode=mode, measurements=meas,
                           channel_metadata=ch_meta, conditions=conditions,
                           warnings=warnings)
