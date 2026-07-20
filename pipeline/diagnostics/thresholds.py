"""Versioned diagnostic thresholds.

Single source of truth for every constant used by pipeline.diagnostics.signal.
Changing ANY value requires bumping THRESHOLDS_VERSION (contracts record the
version used, so results stay auditable). Rationale, units, and known failure
modes for each threshold are documented in docs/DIAGNOSTICS.md — no
unexplained constants may appear in signal.py.
"""

# diag-v2 (Phase 1 repair): `max_full_analysis_sec` lowered 600 → 120 so that
# full-file diagnostics can never allocate more than a documented bound
# (120 s x 48 kHz x 2 ch x 4 B ≈ 46 MB). Longer recordings use the bounded
# 3-window sampling path, which reads only those windows from disk.
THRESHOLDS_VERSION = "diag-v2"

THRESHOLDS: dict[str, float] = {
    # --- amplitude ---------------------------------------------------------
    "clip_level": 0.999,          # |x| ≥ this (full-scale=1.0) counts as clipped
    "near_clip_level": 0.98,      # near-clipping band
    "clip_run_min_samples": 3,    # ≥3 consecutive clipped samples = one run
    "flat_top_min_samples": 4,    # constant-extreme plateau indicating saturation
    "silence_level": 1e-4,        # |x| below → silent sample (-80 dBFS)
    "near_silence_level": 1e-3,   # |x| below → near-silent (-60 dBFS)
    "dc_offset_warn": 0.01,       # |mean| above → DC-offset condition
    # --- dropouts ----------------------------------------------------------
    "dropout_level": 1e-5,        # |x| below → candidate dropout sample
    "dropout_min_ms": 20.0,       # run shorter than this is not a dropout
    "dropout_max_ms": 500.0,      # longer runs are treated as silence, not dropout
    "discontinuity_jump": 0.5,    # |x[n]-x[n-1]| above → sudden discontinuity
    # --- spectral ----------------------------------------------------------
    "band_edge_low_q": 0.01,      # occupied band = cumulative-PSD quantiles
    "band_edge_high_q": 0.99,
    "narrowband_low_hz": 200.0,   # lower edge above this AND
    "narrowband_high_hz": 4000.0, # upper edge below this → narrow-band/radio
    "hf_loss_hz": 5000.0,         # upper edge below this → high-frequency loss
    "rumble_hz": 60.0,            # low-band boundary for rumble ratio
    "rumble_ratio_warn": 0.4,     # low-band energy fraction above → rumble
    # --- hum ---------------------------------------------------------------
    "hum_harmonics": 5,           # harmonics scored per mains family
    "hum_bin_bw_hz": 2.0,         # tolerance around each harmonic
    "hum_score_warn": 0.25,       # tonal-fraction score above → hum condition
    "hum_persistence_min": 0.6,   # fraction of frames tone must persist
    # --- noise / SNR proxies ----------------------------------------------
    "snr_frame_ms": 50.0,         # frame size for energy quantiles
    "snr_signal_q": 0.90,         # signal proxy quantile (frame RMS dB)
    "snr_noise_q": 0.10,          # noise-floor proxy quantile
    "snr_low_warn_db": 6.0,       # proxy below → probable severe noise
    # --- channels ----------------------------------------------------------
    "imbalance_warn_db": 6.0,     # |RMS_L - RMS_R| in dB above → imbalance
    "polarity_corr": -0.90,       # interchannel r below → polarity inversion
    "identical_corr": 0.9999,     # r above AND max|diff| tiny → identical
    "identical_max_diff": 1e-6,
    "delay_max_ms": 20.0,         # cross-correlation search window
    "delay_min_corr": 0.5,        # only report delay when correlation supports it
    # --- categorical severity ---------------------------------------------
    "clip_ratio_warn": 0.001,     # combined clipping ratio above → condition
    "silence_ratio_warn": 0.98,   # silent fraction above → no-information
    "dropout_runs_warn": 3,       # runs above → dropout corruption
    # --- analysis windows --------------------------------------------------
    "fft_size": 4096,             # STFT length (documented in DIAGNOSTICS.md)
    "hop_size": 1024,
    "max_full_analysis_sec": 120.0,  # longer files switch to windowed sampling
    "sample_window_sec": 60.0,       # windowed mode: 3 windows (start/mid/end)
}
