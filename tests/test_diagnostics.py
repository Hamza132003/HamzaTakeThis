"""Condition-vector diagnostics on synthetic fixtures (no files, no network).

Tolerances are justified by the underlying math (e.g. RMS of a unit sine is
1/sqrt(2) exactly; quantile SNR of a constructed two-level signal is the
constructed level difference within frame-quantization error). Tests avoid
restating implementation constants except where the contract IS the constant
(e.g. thresholds_version propagation).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from pipeline.diagnostics import THRESHOLDS_VERSION, analyse  # noqa: E402

SR = 16000


def _get(cv, name):
    for m in cv.measurements:
        if m.name == name:
            return m
    raise KeyError(name)


def _cond(cv, name):
    for c in cv.conditions:
        if c.condition == name:
            return c
    raise KeyError(name)


def _sine(freq, seconds=2.0, amp=0.5, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_rms_peak_crest_of_unit_sine():
    cv = analyse(_sine(440, amp=1.0), SR)
    assert _get(cv, "rms").value == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert _get(cv, "peak_amplitude").value == pytest.approx(1.0, rel=1e-3)
    assert _get(cv, "crest_factor").value == pytest.approx(np.sqrt(2), rel=1e-2)


def test_silence_and_near_silence():
    cv = analyse(np.zeros(SR), SR)
    assert _get(cv, "silent_sample_ratio").value == 1.0
    assert _cond(cv, "probable_silence_no_information").status == "present"
    cv2 = analyse(np.full(SR, 5e-4), SR)          # near-silent, not silent
    assert _get(cv2, "near_silence_ratio").value == 1.0
    assert _get(cv2, "silent_sample_ratio").value == 0.0


def test_dc_offset_detection():
    cv = analyse(_sine(440, amp=0.1) + 0.2, SR)
    assert _get(cv, "dc_offset").value == pytest.approx(0.2, abs=1e-3)
    assert _cond(cv, "probable_dc_offset").status == "present"


def test_positive_and_negative_clipping_and_runs():
    x = _sine(100, amp=1.4)                       # symmetric hard clip
    xc = np.clip(x, -1.0, 1.0)
    cv = analyse(xc, SR)
    pos = _get(cv, "positive_clipping_ratio").value
    neg = _get(cv, "negative_clipping_ratio").value
    assert pos > 0.05 and neg > 0.05
    assert pos == pytest.approx(neg, rel=0.05)    # symmetric by construction
    assert _get(cv, "clipping_run_count").value > 0
    assert _get(cv, "max_clipping_run").value > 0
    assert _cond(cv, "probable_clipping").status == "present"
    # Positive-only clipping = asymmetric saturation: DC-shifted sine whose
    # positive peaks (1.1) clip at full scale while negative peaks (-0.1)
    # stay far below the clip level.
    shifted = np.clip(0.6 * _sine(100, amp=1.0) + 0.5, -1.0, 1.0)
    only_pos = analyse(shifted, SR)
    assert _get(only_pos, "positive_clipping_ratio").value > 0.0
    assert _get(only_pos, "negative_clipping_ratio").value == 0.0


def test_hum_50hz_with_harmonics_detected():
    x = sum(_sine(50 * h, amp=0.3 / h) for h in range(1, 4))
    cv = analyse(x, SR)
    assert _get(cv, "hum_50hz_score").value > _get(cv, "hum_60hz_score").value
    assert _get(cv, "hum_dominant_family").value == 50.0
    assert _cond(cv, "probable_mains_hum").status == "present"


def test_hum_60hz_family():
    x = sum(_sine(60 * h, amp=0.3 / h) for h in range(1, 4))
    cv = analyse(x, SR)
    assert _get(cv, "hum_dominant_family").value == 60.0


def test_narrowband_radio_signal():
    rng = np.random.default_rng(0)                # deterministic seed
    noise = rng.standard_normal(2 * SR)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    spec[(freqs < 300) | (freqs > 3400)] = 0      # 300–3400 Hz radio band
    x = np.fft.irfft(spec, len(noise)) * 0.1
    cv = analyse(x, SR)
    assert _cond(cv, "probable_narrowband_radio").status == "present"
    assert 200 < _get(cv, "occupied_band_low_edge").value < 500
    assert 3000 < _get(cv, "occupied_band_high_edge").value < 4000


def test_severe_lowpass_flag_and_fullband_absence():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(2 * SR)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    spec[freqs > 2000] = 0
    lp = np.fft.irfft(spec, len(noise)) * 0.1
    assert _cond(analyse(lp, SR), "probable_severe_lowpass").status == "present"
    full = rng.standard_normal(2 * SR) * 0.1      # broadband
    assert _cond(analyse(full, SR), "probable_severe_lowpass").status == "absent"


def test_snr_proxy_labeled_proxy_and_ordered():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(4 * SR) * 0.01
    speechish = noise.copy()
    speechish[SR:2 * SR] += _sine(300, seconds=1.0, amp=0.5)[: SR]
    m_clean = _get(analyse(speechish, SR), "snr_quantile_proxy")
    m_noisy = _get(analyse(noise, SR), "snr_quantile_proxy")
    assert m_clean.kind == "proxy" and m_noisy.kind == "proxy"
    assert "true SNR" in m_clean.method or "NOT true SNR" in m_clean.method
    assert m_clean.value > m_noisy.value          # contrast orders correctly


def test_dropouts_and_repeated_zero_gaps():
    x = _sine(440, seconds=3.0, amp=0.4)
    for start_s in (0.5, 1.2, 1.9, 2.5):          # 4 gaps of 60 ms
        a = int(start_s * SR)
        x[a:a + int(0.06 * SR)] = 0.0
    cv = analyse(x, SR)
    assert _get(cv, "dropout_run_count").value == 4
    assert _get(cv, "repeated_zero_runs").value == 4
    assert _get(cv, "max_dropout").value == pytest.approx(0.06, rel=0.1)
    assert _cond(cv, "probable_dropout_corruption").status == "present"


def test_channel_imbalance_and_identical_and_polarity():
    ch = _sine(440, amp=0.5)
    imbalanced = np.stack([ch, ch * 0.05])
    cv = analyse(imbalanced, SR)
    assert _get(cv, "channel_imbalance").value > 20
    assert _cond(cv, "probable_channel_imbalance").status == "present"

    ident = analyse(np.stack([ch, ch]), SR)
    assert _get(ident, "identical_channels_indicator").value == 1.0
    assert _get(ident, "interchannel_correlation").value == pytest.approx(1.0, abs=1e-6)

    inverted = analyse(np.stack([ch, -ch]), SR)
    assert _get(inverted, "polarity_inversion_indicator").value == 1.0


def test_nonfinite_samples_sanitized_and_counted():
    x = _sine(440)
    x[100] = np.nan
    x[200] = np.inf
    cv = analyse(x, SR)
    assert _get(cv, "nonfinite_sample_count").value == 2
    assert cv.warnings and "non-finite" in cv.warnings[0]
    for m in cv.measurements:                     # nothing propagated NaN
        if m.valid and m.value is not None:
            assert np.isfinite(m.value)


def test_input_array_not_modified():
    x = _sine(440)
    x[10] = np.nan
    before = x.copy()
    analyse(x, SR)
    assert np.array_equal(x, before, equal_nan=True)


def test_long_file_switches_to_windowed_mode():
    # 601 s of near-silence at 8 kHz keeps this fast while crossing the limit.
    sr = 8000
    x = np.zeros(601 * sr)
    cv = analyse(x, sr)
    assert cv.analysis_mode == "windowed-sample"
    assert any("windows" in w for w in cv.warnings)


def test_every_condition_carries_support_and_version():
    cv = analyse(_sine(440), SR)
    assert cv.thresholds_version == THRESHOLDS_VERSION
    for c in cv.conditions:
        assert c.supporting_measurements, c.condition
        assert c.thresholds_version == THRESHOLDS_VERSION
        assert c.reason
