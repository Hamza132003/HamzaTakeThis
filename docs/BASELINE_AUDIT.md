# BASELINE AUDIT — HamzaTakeThis → AEGIS-X PRIME

Audit date: 2026-07-19
Auditor: Claude Code (planning pass only — no production code changed in this pass)

## 1. Repository state

- **Remote:** `https://github.com/Walayahh/HamzaTakeThis`
- **Branch:** `main`
- **Commit:** `b1f7574` — "Voice Isolator: local voice isolation, transcription, translation and AI analysis pipeline" (single-commit history)
- **Working tree:** DIRTY. Ten tracked files carry uncommitted local modifications made during the current operator session **before** this audit (bug fixes and features: job cancellation incl. mid-stage checks, delete-result endpoint, native browse dialog, AI-summary UI toggle, added media extensions, original-audio playback, crash-surviving `server.log` + faulthandler, `.gitignore` additions). Modified: `.gitignore`, `pipeline/runner.py`, `pipeline/transcription.py`, `pipeline/translation.py`, `pipeline/utils.py`, `webapp/app.py`, `webapp/frontend/src/{App.jsx, Detail.jsx, api.js, styles.css}`.
- **No project LICENSE file is tracked.** (See §5, defect 3.13.)
- **No tests, no CI, no docs/ directory existed before this audit.**

### Files inspected (all 38 tracked files)

`README.md`, `config.yaml`, `requirements.txt`, `setup.ps1`, `dashboard.ps1`, `build_ui.ps1`, `check_env.py`, `main.py`, `generate_deck.py` (skim — pptx generator), `make_usecase1_deck.py` (skim — pptx generator), `.gitignore`,
`pipeline/__init__.py`, `pipeline/audio.py`, `pipeline/separation.py`, `pipeline/noise_analysis.py`, `pipeline/diarization.py`, `pipeline/speakers.py`, `pipeline/transcription.py`, `pipeline/translation.py`, `pipeline/emotion.py`, `pipeline/summary.py`, `pipeline/spectrogram.py` (via runner call-sites), `pipeline/report.py`, `pipeline/models.py`, `pipeline/runner.py`, `pipeline/utils.py`,
`webapp/__init__.py`, `webapp/app.py`, `webapp/frontend/*` (index.html, package.json, vite.config.js, src/App.jsx, src/Detail.jsx, src/Recorder.jsx, src/api.js, src/main.jsx, src/styles.css).

## 2. Environment check (run this date, models already cached)

`check_env.py` → 12/14 components OK.
- `deepfilternet` MISSING (optional; wheel needs Rust toolchain on py3.12 — documented in README).
- `pyannote.audio` reports MISS **only due to an import-order artifact** in `check_env.py` (speechbrain lazy `k2` stub corrupts a `torch._dynamo` module scan). The production pipeline patches this via `pipeline/utils.guard_speechbrain_lazy()`; verified importable through the guard.
- GPU detected: **NVIDIA RTX 5060 Laptop (8 GB, compute capability sm_120 / Blackwell)**.

**Critical environment deviation from the repo's own setup script:** `setup.ps1` installs torch cu124, which supports only ≤ sm_90 and **cannot execute kernels on this GPU** (every CUDA stage fails). The active venv was manually upgraded this session to `torch 2.11.0+cu128 / torchaudio 2.11.0 / torchvision 0.26.0` (sm_120 supported) with `numpy<2` re-pinned for clearvoice. `setup.ps1` and `requirements.txt` do NOT yet reflect this. Full pipeline verified end-to-end on GPU after the upgrade.

## 3. Call graph (verified against code)

```
main.py ──► load_config(config.yaml) ──► pipeline.runner.process_file
webapp/app.py (Flask)
  POST /api/process | /api/record ──► _start_job ──► _worker (thread)
        └─► pipeline.runner.process_file(f, cfg, device, progress, should_cancel)
  GET /api/status /api/events(SSE) /api/recordings /api/recording/<n>
  POST /api/cancel /api/browse-folder    DELETE /api/recording/<n>
  GET /media/<n>/<path>  (serves outputs/<stem>/*)

pipeline.runner.process_file  (11 fixed stages, sequential)
  1  audio.extract_audio          ffmpeg → audio_16k_mono.wav + audio_48k_mono.wav  (mono only)
  2  separation.separate          clearvoice MossFormer2_SE_48K → deepfilternet → speechbrain sepformer
                                  → noisereduce → none (fallback chain); noise = original − voice
  3  noise_analysis.analyse_noise PANNs CNN14 (32 kHz), speech classes filtered out
  4  diarization.diarize          pyannote speaker-diarization-community-1 (needs HF token)
                                  → fallback: librosa energy split, 1 speaker + warning
  5  speakers.build_speaker_tracks  (only if >1 speaker) pairwise overlaps; MossFormer2_SS_16K
                                  2-src separation + ECAPA cosine assignment
  6  transcription.transcribe     faster-whisper large-v3 (int8_float16); decodes original AND
                                  isolated voice; picks ONE whole-file winner; + whisper "translate"
                                  pass → English; anti-hallucination filters
  7  translation.translate_segments  NLLB-200-distilled-1.3B fp16 GPU (600M / CPU fallbacks),
                                  native-source → arb_Arab
  8  emotion.analyse_emotion      superb/wav2vec2-base-superb-er per turn → polarity map
  9  summary.summarize            Qwen3-4B-Instruct-2507 (torchao int8 → bnb 4bit → CPU bf16),
                                  do_sample=True temperature=0.4  (non-deterministic)
  10 spectrogram.render_all + render_speaker_timeline
  11 report.write                 report.json + report.md
Model sharing: pipeline.models.MANAGER (in-process dict cache; one-heavy-model-on-GPU policy;
runner explicitly evicts Whisper before NLLB per file).
```

## 4. Current contracts, config, fallbacks, hardware assumptions, failure modes

- **I/O contract:** input = any file whose suffix ∈ MEDIA_EXT (webapp) or any path (CLI). Output = `outputs/<stem>/` containing wavs, spectrograms/, speakers/, report.json, report.md. `report.json` is an **unversioned** ad-hoc dict (schema documented only by `runner.process_file` assembly code).
- **Config:** single untyped `config.yaml`, per-stage keys (see file). No schema, no validation, no versioning. Webapp overlays per-job keys (`language`, `hf_token`, `separate_overlap`, `ai_summary`, `device`) via `_make_cfg`.
- **Fallback behavior:** every stage catches broad `Exception` and degrades (separation chain → passthrough; diarization → single-speaker energy split **with** a user-visible warning; translation → empty arabic fields; emotion/summary → skipped). Fallbacks are logged but **not** structurally recorded in the report beyond `*_method` strings.
- **Hardware assumptions:** written for a 6 GB RTX A1000 (comments throughout); actual machine is an 8 GB RTX 5060 requiring cu128 builds (see §2). CPU fallback exists for every stage. Windows-first (PowerShell scripts, symlink workarounds, torchcodec bypass in diarization).
- **Failure modes observed live this session:** (a) cu124 torch → CUDA "no kernel image" on all GPU stages, silently degrading to fallback chains — an example of a *silent model failure* the spec's Phase 13 targets; (b) `panns_inference` shells out to `wget` over **plain HTTP** for its label CSV — absent on Windows (manual fetch was required) and an unverified-download supply-chain issue; (c) server process died mid-batch with no persisted log (now mitigated by `server.log` tee, uncommitted); (d) OneDrive-synced repo path caused slow/locking I/O on large caches.

## 5. Defect verification (spec §3) — all thirteen CONFIRMED, two with nuance

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 3.1 | Single-family ASR | **CONFIRMED** | Only faster-whisper in `transcription.py`; "original vs enhanced" both decoded by the same model — correlated hypotheses. |
| 3.2 | Invalid whole-recording branch score | **CONFIRMED** | `transcription.py` `_decode_best`: `score = sum(c["alp"] * len(c["text"]))` — avg_logprob is negative, so score *decreases* with transcript length (character count); a shorter incomplete hypothesis can win. Same expression ranks candidate languages. |
| 3.3 | Whole-branch winner destroys evidence | **CONFIRMED** | `transcribe()` keeps a single `best` source wav for the whole file; the losing branch's segments are discarded entirely. |
| 3.4 | Uncalibrated confidence | **CONFIRMED** | Raw Whisper word probabilities surface directly as "Confidence" in report.md tables and the UI words view; thresholds (`min_word_prob` 0.35 etc.) applied to raw scores. |
| 3.5 | Recording-level language assumption | **CONFIRMED** | One language per clip by design (config comment "one language per clip"; `_decode_best` returns a single lang; UI selector is per-recording). Code-switch segments are forced into the winner language. |
| 3.6 | Hard diarization gating can erase faint speech | **CONFIRMED (partial mitigation exists)** | `gate_with_diarization: true` gates decoding to pyannote-derived `speech_regions` (`_clip_timestamps`, ±0.25 s pad). Diarization output is the *only* VAD. Mitigation: `_decode_task` retries without VAD **only when zero segments survive overall**; region-level misses within an otherwise successful decode remain invisible. Fallback diarization = energy split (librosa `top_db=30`) which drops faint speech below threshold. |
| 3.7 | Pairwise overlap logic wrong for 3+ speakers | **CONFIRMED — guard is dead code** | `speakers.detect_overlaps` builds overlaps **pairwise**, so every overlap record has exactly 2 speakers by construction; the `len(ov["speakers"]) != 2 → skipped_3plus` check can never fire. A 3-speaker region yields 3 contradictory pairwise regions, each running the 2-source separator on 3-voice audio and overwriting buffers in order. No active-speaker-set representation exists. |
| 3.8 | Early mono downmix destroys channel evidence | **CONFIRMED** | `audio.extract_audio` runs ffmpeg with `-ac 1` for both 16 k and 48 k outputs; original channel layout is never preserved or probed. |
| 3.9 | No integrated scientific benchmark | **CONFIRMED** | No evaluation code, datasets, metrics, or reference handling anywhere in the repo. |
| 3.10 | Sequential batch reloads models excessively | **CONFIRMED with nuance** | `models.MANAGER` *does* cache across files within a server process (README's claim is genuine), but: file-at-a-time through all 11 stages; on CUDA the runner evicts Whisper **every file** before NLLB (CTranslate2 objects can't move devices → full reload ≈15 s/file); CLI pays full cold loads per invocation; no persistence, no resumability, no idempotency keys, no content-addressed reuse. A killed batch restarts from zero (observed live this session). |
| 3.11 | Unvalidated emotion + speculative summary | **CONFIRMED** | `emotion.enabled: true` by default; model `superb/wav2vec2-base-superb-er` is an English (IEMOCAP-family) SER model; README markets it "language-agnostic" — unvalidated for Hebrew/Persian degraded channels. Summary uses `do_sample=True, temperature=0.4` (non-deterministic), mixes observation and inference in one narrative, and can attribute location/identity ("possibly") from noise tags. Session change (uncommitted): webapp defaults AI summary OFF; config/CLI default remains ON. |
| 3.12 | Insufficient security & provenance | **CONFIRMED** | No source/model hashing, no revision pinning (`pyannote/speaker-diarization-community-1`, HF `main` floating for all models), no SBOM/lockfile (requirements are `>=` ranges), no audit log, plain-HTTP label download in panns dependency, `trust_remote_code` not used (good), tokens correctly kept out of git (`hf_token.txt` gitignored; UI field). Flask binds 127.0.0.1 only (good) but has no auth and now exposes delete/browse endpoints (session additions) — acceptable for localhost, must be revisited if exposure changes. |
| 3.13 | No repository license | **CONFIRMED** | No LICENSE/COPYING tracked. Release packaging must be blocked until the owner picks one (decision record required — see docs/RISK_REGISTER.md R-1). |

## 6. Session-era additions not in the spec's baseline (uncommitted)

Cancellation (`/api/cancel` + `pipeline.utils.set_cancel_check/check_cancel` polled per Whisper segment and per NLLB batch), result deletion (`DELETE /api/recording/<name>` with path-traversal guard), native folder/file picker (`POST /api/browse-folder`, PowerShell OpenFileDialog subprocess), `ai_summary` per-job flag, `_truthy` FormData coercion fix, extended `MEDIA_EXT` (.mpeg/.mpg/.wma/.aiff/.aif/.amr/.3gp), original-audio playback in report/UI, `server.log` tee + faulthandler. These are functional but untested/uncommitted; Phase 0 must commit them on a branch before any refactor.

## 7. Baseline strengths worth preserving (per spec §0 "preserve what works")

Same-model residual noise derivation; crossfaded chunked enhancement; diarization-failure warnings surfaced to the UI; anti-hallucination segment filters (a sound instinct, to be re-based on calibrated scores); one-heavy-model VRAM policy; Windows resilience patches (`guard_speechbrain_lazy`, torchcodec bypass, UTF-8 console fix, truststore); SSE progress; per-word timing UI with click-to-seek.
