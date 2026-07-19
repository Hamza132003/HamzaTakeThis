"""Render polished mel-spectrogram PNGs for the dashboard.

Dark theme to match the web UI. One image per track (original / voice /
noise / per-speaker) plus a stacked comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import log

SR = 16000
BG = "#0f1523"
FG = "#eaf0f7"
MUTED = "#93a1b5"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_mel(wav_path: Path, out_png: Path, title: str,
               cmap: str = "magma") -> str | None:
    """Render a single log-mel spectrogram to a PNG. Returns the path or None."""
    try:
        import librosa
        import librosa.display
        plt = _setup()

        # Load at the file's NATIVE rate so 48 kHz tracks show the full band
        # (a fixed 16 kHz load would hide everything above 8 kHz).
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        sr = int(sr or SR)
        if y.size == 0:
            y = np.zeros(sr // 10, dtype=np.float32)
        hop = max(256, sr // 62)   # ~62 frames/s regardless of rate
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=sr // 2, hop_length=hop)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(9, 2.8), dpi=120)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                                       x_axis="time", y_axis="mel",
                                       fmax=sr // 2, cmap=cmap, ax=ax)
        ax.set_title(title, color=FG, fontsize=12, loc="left", pad=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2c3a52")
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
        cbar.outline.set_edgecolor("#2c3a52")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_png), facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return str(out_png)
    except Exception as e:
        log(f"WARNING: spectrogram '{title}' failed ({e}).")
        return None


def render_wave(wav_path: Path, out_png: Path, title: str,
                color: str = "#35d2cb") -> str | None:
    """Render an amplitude waveform (min/max envelope) to a PNG."""
    try:
        import librosa
        plt = _setup()

        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        sr = int(sr or SR)
        if y.size == 0:
            y = np.zeros(sr // 10, dtype=np.float32)

        # Downsample to a min/max envelope (~1600 columns) so long files
        # render fast and the plot still shows every transient.
        cols = 1600
        n = len(y)
        hop = max(1, n // cols)
        trimmed = y[: (n // hop) * hop].reshape(-1, hop)
        t = (np.arange(trimmed.shape[0]) * hop) / sr
        lo, hi = trimmed.min(axis=1), trimmed.max(axis=1)

        fig, ax = plt.subplots(figsize=(9, 1.9), dpi=120)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.fill_between(t, lo, hi, color=color, linewidth=0.4, alpha=0.9)
        ax.axhline(0, color="#2c3a52", linewidth=0.6)
        ax.set_xlim(0, max(t[-1] if len(t) else 1, 0.1))
        lim = max(1e-3, float(np.max(np.abs(y))) * 1.05)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, color=FG, fontsize=12, loc="left", pad=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2c3a52")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_png), facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return str(out_png)
    except Exception as e:
        log(f"WARNING: waveform '{title}' failed ({e}).")
        return None


def render_speaker_timeline(segments: list, duration: float,
                            out_png: Path) -> str | None:
    """Who-spoke-when activity plot: one 0/1 step lane per speaker."""
    try:
        plt = _setup()
        speakers = sorted({s["speaker"] for s in segments})
        if not speakers or duration <= 0:
            return None

        colors = ["#35d2cb", "#f2b45a", "#6ed69a", "#e78a7b",
                  "#8fa8ff", "#d98fff"]
        res = 0.05                      # 50 ms grid
        n = max(2, int(duration / res))
        t = np.arange(n) * res

        fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(speakers)), dpi=120)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        for i, spk in enumerate(speakers):
            active = np.zeros(n)
            for s in segments:
                if s["speaker"] != spk:
                    continue
                a, b = int(s["start"] / res), min(n, int(s["end"] / res) + 1)
                active[a:b] = 1
            base = (len(speakers) - 1 - i) * 1.4
            c = colors[i % len(colors)]
            ax.fill_between(t, base, base + active, step="mid",
                            color=c, alpha=0.85, linewidth=0)
            ax.text(0, base + 1.12, spk, color=c, fontsize=9, fontweight="bold")

        ax.set_xlim(0, duration)
        ax.set_ylim(-0.2, len(speakers) * 1.4)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
        ax.set_title("Speaker activity (who spoke when)", color=FG,
                     fontsize=12, loc="left", pad=8)
        for spine in ax.spines.values():
            spine.set_color("#2c3a52")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_png), facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return str(out_png)
    except Exception as e:
        log(f"WARNING: speaker timeline failed ({e}).")
        return None


WAVE_COLORS = {"original": "#8fa8ff", "voice": "#35d2cb", "noise": "#e78a7b"}


def render_all(tracks: dict, out_dir: Path) -> tuple[dict, dict]:
    """tracks: {name: (wav_path, title, cmap)}.
    Returns ({name: spectrogram_png}, {name: waveform_png})."""
    spec_dir = out_dir / "spectrograms"
    specs, waves = {}, {}
    for name, (wav, title, cmap) in tracks.items():
        if wav and Path(wav).exists():
            png = render_mel(Path(wav), spec_dir / f"{name}.png", title, cmap)
            if png:
                specs[name] = png
            wpng = render_wave(Path(wav), spec_dir / f"{name}_wave.png",
                               f"{title} — waveform",
                               WAVE_COLORS.get(name, "#35d2cb"))
            if wpng:
                waves[name] = wpng
    return specs, waves
