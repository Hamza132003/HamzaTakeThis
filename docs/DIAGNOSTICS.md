# DIAGNOSTICS.md — Phase 1 signal diagnostics: formulas, thresholds, limits

Implementation: `pipeline/diagnostics/signal.py`. Constants: `pipeline/diagnostics/thresholds.py`
(`THRESHOLDS_VERSION = "diag-v1"`; any change requires a version bump). All diagnostics are
deterministic and model-free; no LLM assigns any condition. Nothing modifies audio.

## Global analysis parameters

- STFT: Hann window, `fft_size=4096`, `hop_size=1024` → at 16 kHz: 3.9 Hz bin spacing, 256 ms
  window, 64 ms hop. Chosen so 50/60 Hz mains bins are resolvable while frames stay
  speech-scaled.
- Frame energy metrics: 50 ms non-overlapping frames.
- Long files: above `max_full_analysis_sec=600` s, analysis switches to three deterministic
  60 s windows (start/middle/end); `analysis_mode="windowed-sample"` and a warning are recorded.
  Sampled mode can miss localized events between windows (documented false negative).
- Numerical safety: float64 accumulation; ε=1e-12 guards every division; NaN/Inf are counted
  (`nonfinite_sample_count`), reported, and sanitized to 0 before analysis; outputs clamped to
  ±1e12 and non-finite results invalidated with a reason.

## A. Basic waveform properties (units, formula, kind)

| Measurement | Formula | Units | Kind |
|---|---|---|---|
| duration | N / sr | s | measured |
| peak_amplitude | max\|x\| | linear (full scale = 1) | measured |
| rms | sqrt((1/N)·Σx²) | linear | measured |
| dc_offset | (1/N)·Σx | linear | measured |
| crest_factor | peak / RMS | ratio ≥ 1 | measured |
| zero_crossing_rate | mean\|diff(signbit(x))\| | crossings/sample (0..1) | measured |
| silent_sample_ratio | mean(\|x\| < 1e-4) | ratio | measured |
| near_silence_ratio | mean(\|x\| < 1e-3) | ratio | measured |
| dynamic_range_proxy | p95 − p5 of 50 ms frame-RMS dB | dB | proxy |

Interpretation: crest factor ~1.41 for a sine, >4 for speech; low dynamic-range proxy with high
RMS suggests compression/limiting. False positive: DC-offset flag on intentionally asymmetric
signals (e.g., some codec artifacts). Routing-suitable: yes (all cheap and stable).

## B. Clipping (threshold `clip_level=0.999` of full scale)

- ratios: mean(x ≥ +0.999), mean(x ≤ −0.999), union.
- runs: ≥ `clip_run_min_samples=3` consecutive clipped samples; count + longest (s).
- near_clipping: \|x\| ≥ 0.98.
- flat_top_runs (estimated): ≥4 consecutive *identical* samples with \|x\| ≥ 0.98 — saturation
  that an exact-threshold test misses.
- Condition `probable_clipping`: union ratio > `clip_ratio_warn=0.001`; severity = ratio/0.01
  capped at 1.
- Known false negative: clipping that happened upstream at a lower analog level then was
  attenuated (mitigated partially by flat_top_runs). False positive: legitimate full-scale
  square-ish synth content. Routing-suitable: yes.

## C. Dropouts / discontinuities (estimated)

- Candidate: runs of \|x\| < `dropout_level=1e-5` lasting 20–500 ms (shorter = normal
  zero-crossings; longer = ordinary silence, not dropout).
- repeated_zero_runs: same window but exact-0 samples — packet-loss-like.
- discontinuity_count: \|x[n]−x[n−1]\| > 0.5.
- Limitation (documented): cannot distinguish intentional edit silence from transmission
  dropout; concealment algorithms that interpolate rather than zero are invisible here.

## D. Spectral (STFT mean PSD)

- centroid = Σ(f·P)/ΣP (Hz); bandwidth = sqrt(Σ((f−c)²·P)/ΣP); flatness = geomean(P)/mean(P)
  (0..1, 1 = white); rolloff_85 = frequency at 85 % cumulative PSD.
- Band ratios: P(0–300)/ΣP, P(300–3400)/ΣP, P(3400–Nyquist)/ΣP.
- Occupied band edges (estimated): cumulative-PSD quantiles 1 % / 99 %.
- Conditions: `probable_narrowband_radio` = low edge > 200 Hz AND high edge < 4000 Hz
  (canonical telephone band 300–3400 Hz sits inside; full-band content has energy below
  200 Hz). `probable_severe_lowpass` = high edge < 5000 Hz. `probable_severe_highpass` =
  low edge > 300 Hz without the narrowband pattern.
- False positive: a recording that is genuinely only a narrow-band *instrument* (not a channel
  limit). Zero-energy input → edges indeterminate (recorded as such).

## E. Mains hum (estimated)

Per family f₀ ∈ {50, 60} Hz: tonal fraction per frame = Σ P(h·f₀ ± 2 Hz, h=1..5) / ΣP;
score = median over ≤200 frames; `hum_persistence` = fraction of frames with tonal fraction
above half-threshold. Condition `probable_mains_hum` requires score > 0.25 AND persistence
≥ 0.6 — voiced speech harmonics modulate in time, so persistence guards against them
(documented residual false positive: sustained low organ/bass notes near 50/60 Hz;
status becomes `indeterminate` when score is high but persistence low).

## F. Noise / SNR proxies — NEVER "true SNR" (no clean reference exists)

- noise_floor_proxy = q0.10 of 50 ms frame-RMS dB (dBFS).
- stationary_noise_proxy = std of quietest 30 % frames (low = stationary).
- snr_quantile_proxy = q0.90 − q0.10 frame-RMS dB.
- snr_segmental_proxy = mean(top 30 %) − mean(bottom 30 %) frame-RMS dB.
- All carry kind="proxy" and an explicit NOT-true-SNR note in `method`. Failure mode:
  continuous speech with no pauses compresses the quantile gap (underestimates contrast);
  music overestimates. Routing-suitable: yes, as a *relative* signal only.

## G. Channel relationships (first two channels; per-channel stats for all)

- imbalance = \|20·log10(RMS₀/RMS₁)\| dB; condition at > 6 dB.
- interchannel Pearson r = cov(ch0,ch1)/(σ₀σ₁); polarity inversion flag at r < −0.90;
  identical-channel flag at r > 0.9999 AND max\|diff\| < 1e-6 (dual-mono detector).
- mid/side energy ratio = RMS((L+R)/2)/RMS((L−R)/2).
- delay estimate: argmax of cross-correlation over ±20 ms, only computed when \|r\| ≥ 0.5
  ("safely measurable"); otherwise invalid-with-reason.

## H. Categorical conditions

Every condition carries: status present/absent/**indeterminate**, severity 0..1, reason with
the actual numbers, `supporting_measurements` (names of the evidence rows), and
`thresholds_version`. Consumers must treat `indeterminate` as "do not route on this".

## Suitability matrix

Routing-suitable now: clipping, silence, narrowband, lowpass, imbalance, dropout, SNR proxies
(relative). NOT routing decisions by themselves: hum (feeds notch-filter choice later),
DC offset (feeds conservative-DSP branch), delay estimate (informational until validated).
