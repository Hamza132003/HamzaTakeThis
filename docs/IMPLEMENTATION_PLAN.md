# IMPLEMENTATION PLAN — AEGIS-X PRIME

Status: DRAFT — awaiting operator approval. No production code changes yet.
Companion docs: BASELINE_AUDIT.md, ARCHITECTURE_DECISIONS.md, RISK_REGISTER.md, ACCEPTANCE_TEST_PLAN.md, MODEL_LICENSE_MATRIX.md, DATA_LICENSE_MATRIX.md.

## 1. Phase dependency graph

```
P0 foundation (branch, commit session work, locks, licenses, CI, fakes)
 └─► P1 ingest+diagnostics ─► P2 branch factory ─► P3 VAD/LID/diarization
                                   │                    │
                                   └────────┬───────────┘
                                            ▼
                          P4 ASR forest (adapters; Whisper first, OmniASR CTC second)
                                            ▼
                          P5 alignment + evidence graph + ROVER/MBR
                                            ▼
                          P6 confidence calibration + abstention  ◄─ needs P10 dev data
                                            ▼
              P7 critical verification      P8 translation forest
                                            ▼
                          P9 reports/UI     P10 evaluation framework (can start after P0, parallel)
                                            ▼
                          P11 ablations ─► P12 DAG/batch ─► P13 security ─► P14 tests/CI hardening
                                            ▼
                          P15 config schema ─► P16 demo
```
Parallelizable early: P10 dataset registry + metrics scaffolding can begin right after P0 and is a prerequisite for honest P4–P6 decisions. P15 config schema should land incrementally from P1 onward.

## 2. File-by-file change map (migration, not rewrite)

Existing modules stay importable until compatibility tests pass (spec §26). New code lands under `pipeline/` subpackages from spec §5; old modules become thin wrappers before deletion.

| Current file | Fate | Target |
|---|---|---|
| `pipeline/audio.py` | wrap → supersede | `ingest/immutable.py` (hashing, content IDs), `ingest/media_probe.py` (ffprobe), `ingest/channel_factory.py` (per-channel/mid/side; mono becomes one branch) |
| `pipeline/separation.py` | refactor in place → registry | `enhancement/registry.py` + `enhancement/clearvoice.py` (chunking code moves intact), `classical.py` (the existing high-pass/LUFS/VAD-gain becomes `conservative_dsp` — **moved out of the mandatory post-path**, becoming a branch), `quality_gate.py`, `observation_addition.py` (new) |
| `pipeline/diarization.py` | wrap → backend | `diarization/pyannote_backend.py` + `vad/energy.py` (current fallback), `vad/ensemble.py`, `vad/silero.py` (Silero already ships with faster-whisper — zero new weights), `vad/rescue.py` |
| `pipeline/speakers.py` | rewrite overlap repr | `diarization/overlap.py` (active-speaker sets — fixes dead 3+ guard), `separation/two_speaker.py`, `separation/speaker_assignment.py` (ECAPA code moves intact, adds assignment margins) |
| `pipeline/transcription.py` | split | `asr/adapters/whisper.py` (decode code), `asr/normalization.py` (junk/ngram filters), **delete** `_decode_best` whole-file score (defect 3.2) with regression test |
| `pipeline/translation.py` | wrap → adapter | `translation/adapters/nllb.py`, `translation/consistency.py` (new number/entity checks) |
| `pipeline/emotion.py` | gate | experimental flag, default OFF (config change; README correction) |
| `pipeline/summary.py` | gate + determinize | `do_sample=False`; OBSERVATIONS/INFERENCES split; timestamp citations; off in forensic profile |
| `pipeline/runner.py` | becomes compat shim | `orchestration/dag.py`, `orchestration/jobs.py`; `process_file` kept as a façade that submits a 1-file DAG |
| `pipeline/models.py` | extend | `orchestration/workers.py` (persistent per-model workers), keep MANAGER API |
| `pipeline/report.py` | version schema | `reporting/`; `report_schema_version`; migration utility for existing report.json |
| `webapp/app.py` | keep; add routes | job DAG status, hypothesis browser, review edits (P9) |
| `config.yaml` | migrate | typed schema + profiles (P15), migration tool + deprecation warnings |
| `check_env.py` | supersede | `python -m aegis doctor` (fix the import-order false-miss) |
| new | — | `pipeline/contracts/*` (Pydantic), `pipeline/security/*`, `pipeline/evaluation/*`, `tests/`, `.github/workflows/ci.yml` |

## 3. Phase plans, DoD, and runtime estimates

Estimates assume the current single dev machine (RTX 5060 8 GB), sequential work; they are engineering-effort estimates, not calendar promises.

### Phase 0 — foundation (est. 1–2 days)
Branch `feature/aegis-x-prime`; **first commit = the currently-dirty session fixes** (they predate this plan and must be isolated cleanly). Add CLAUDE.md, THIRD_PARTY_NOTICES.md, MODEL_LICENSES.yaml, DATA_LICENSES.yaml, MODEL_MANIFEST.yaml (pinned revisions — currently everything floats on `main`), SECURITY.md, PRIVACY.md. Lockfile via `uv pip compile` (keep human-readable requirements.txt). Update `setup.ps1`/requirements for the **cu128 torch requirement discovered on this hardware** (sm_120; cu124 is broken here — see BASELINE_AUDIT §2). CI (lint, mypy on new code, pytest, secret scan, no model downloads). Unit-test fakes for every heavy model. DoD: CI green on a clean clone without network model access; session fixes committed; no floating model revisions referenced by new code.

### Phase 1 — immutable ingest + diagnostics (est. 2–3 days)
SHA-256/512 of source; ffprobe metadata; per-channel lossless PCM extraction; mono/mid/side as *derived* artifacts; duplicate detection by hash. ConditionVector diagnostics with synthetic tests (clipping, hum, silence, band-limit, channel imbalance, dropouts). DoD: contracts `AudioAsset`/`DerivedAudio`/`ConditionVector` versioned + tested; existing pipeline consumes the new ingest through the compat shim with byte-identical mono output.

### Phase 2 — branch factory (est. 3–4 days)
Registry with stable `branch_id` + transformation manifests; branches per spec §8 (original_mono, per-channel, mid/side, conservative_dsp, mossformer2_48k, deepfilternet3 [needs Rust toolchain or prebuilt wheel — currently uninstallable on py3.12/Windows; gate accordingly], radio_conditioned, declip, dereverb, observation_addition λ∈{0.25,0.5,0.75}). Delete defect-3.2 score; segment utility U(h) per spec §8.2 with regression test ("longer correct hypothesis not penalized"). DoD: branch outputs content-addressed; no branch deleted on perceptual grounds; regression test red-on-old-formula, green-on-new.

### Phase 3 — VAD ensemble, segment LID, diarization forest, overlap sets (est. 3–5 days)
High-recall union (pyannote activity + Silero + energy) with per-detector support recorded; conservative intersection tier; ASR rescue scan for uncovered speech-like energy (fixes defect 3.6). Active-speaker-set timeline replaces pairwise overlaps (fixes 3.7); 2-src separation only when exactly two active; `overlap_unresolved` for 3+. Segment/speaker-level LID (Whisper posteriors + text LID; acoustic LID adapter optional) — fixes 3.5. DoD: synthetic 3-speaker fixture yields `overlap_unresolved`, not contradictory tracks; missed-VAD fixture recovered by rescue scan; code-switch fixture keeps two languages.

### Phase 4 — ASR forest (est. 4–6 days + model downloads)
Adapter interface (capabilities, license metadata, revision, failure reason). Tree A Whisper (existing, + chunk-offset diversity, no temperature-as-family). Tree B **Omnilingual ASR CTC 300M** — the second independent family required for the 6 GB/8 GB `fast` profile; startup verification of he/fa support against model metadata, fail-closed. Trees C (7B rescue), D (OWSM-CTC v4), E/F (specialists) land as license/hardware-gated adapters with mocks only, per spec §26 — no >2 GB downloads without separate approval. Tree G keyword/phoneme rescue scaffold. FAST/FULL/RESCUE router skeleton. DoD: two independent families produce hypotheses on the fixture pack on this GPU sequentially; every adapter records revision + failure states; unsupported-language fail-closed test.

### Phase 5 — alignment, evidence graph, ROVER, MBR (est. 4–6 days)
RTL-aware normalization (raw forms always preserved); time+edit-distance alignment; word lattice with null arcs; ROVER with condition-dependent weights + diversity penalty; optional MBR. DoD: unit tests for RTL normalization, null voting, known-fusion fixtures; consensus words link to supporting hypothesis IDs and branch intervals.

### Phase 6 — confidence + abstention (est. 3–5 days, needs P10 dev data)
Feature extraction per spec §12.1; logistic/GBT/isotonic/beta calibrators; ECE/Brier/NLL/risk-coverage; VERIFIED…UNRECOVERABLE tiers; `[INAUDIBLE]` abstention output. DoD: calibration trained on dev only, evaluated on disjoint test; reliability plots emitted by evaluation runner; UI shows calibrated vs raw distinctly.

### Phase 7 — critical verification (est. 2–3 days)
Reviewed lexicons; entity/number/negation extraction + metrics; independent-support rule. DoD: critical-token metrics in benchmark output; no lexicon-forced insertions (test).

### Phase 8 — translation forest (est. 2–4 days)
NLLB adapter (**license gate: CC-BY-NC — see MODEL_LICENSE_MATRIX; blocks commercial deployment profile**); consistency tests (numbers, entities, negation); SeamlessM4T v2 adapter license-gated + mock-tested; uncertainty propagation to target spans. DoD: consistency test suite green; per-path translation metrics reported separately.

### Phase 9 — evidence-linked report + review UI (est. 3–5 days)
Versioned report schema + migration for existing report.json; hypothesis browser; calibrated-confidence heatmap; reviewer edits (author, timestamp, reason, previous value); summary determinized + OBSERVATIONS/INFERENCES; emotion default-off experimental. DoD: click-a-word plays original interval by default; all fallbacks visible; old reports still load.

### Phase 10 — evaluation framework (start early; est. 4–6 days engineering)
Manifest-driven dataset registry; normalization versions; WER/CER, cpWER/tcpWER, DER/JER, chrF++/COMET/BLEU, calibration metrics, hallucinated-words-per-non-speech-minute; paired block bootstrap by source/speaker; baselines 1–9 of spec §16.5. Data acquisition itself is operator-gated (licenses — see DATA_LICENSE_MATRIX). DoD: `aegis evaluate`/`compare` reproduce numbers from a manifest deterministically; no metric hard-coded anywhere.

### Phase 11 — ablations (est. 2–3 days) — automated ablation matrix + error buckets per spec §17.
### Phase 12 — DAG + batch (est. 4–6 days) — SQLite state, content-addressed artifacts, stage-oriented batching (load Whisper once for all queued spans, then OmniASR, …), resumability, OOM recovery, quarantine. Fixes defect 3.10 fully. **Storage note: the repo currently lives under OneDrive; artifact store and SQLite must move to a non-synced local path (risk R-8).**
### Phase 13 — security & provenance (est. 2–4 days) — hashes everywhere, offline-mode network kill-switch test, read-only model registry, append-only audit log, SBOM; HTTPS-only acquisition (the panns HTTP label fetch gets a checksummed HTTPS mirror step).
### Phase 14 — tests/CI hardening (continuous; final sweep 1–2 days).
### Phase 15 — config schema + profiles (incremental; final 1–2 days) — `profile: fast|full|rescue|forensic|research`, `hardware_profile`, `offline_mode`; migration tool for old config.yaml.
### Phase 16 — demo route (est. 2–3 days) — three-case comparison per spec §22.

## 4. Hardware profiles (adjusted to observed hardware)

| Profile | This repo's target | Notes |
|---|---|---|
| 8 GB GPU (RTX 5060 Laptop, sm_120) — **the actual dev machine** | fast + selective full | Requires torch cu128+; whisper int8_float16 ≈3.2 GB, NLLB-1.3B fp16 ≈2.9 GB, OmniASR CTC 300M ≈1 GB est. — sequential only; 7B rescue excluded |
| 6 GB GPU (spec's original A1000 target) | fast | as spec §18.3 |
| 12–24 GB | full | OmniASR 1B benchmarking, Seamless sequential |
| ≥48 GB | rescue incl. 7B | evidence gates still apply |
| CPU | degraded-transparent | existing int8 paths work today |

## 5. Model download budget (new weights beyond current cache)

Already cached locally: Whisper large-v3 (~3 GB), NLLB 1.3B (~5.5 GB), MossFormer2 SE/SS (~2 GB), PANNs (~650 MB), Qwen3-4B (~8 GB), ECAPA, pyannote community-1 (~30 MB). New required for Phase 4 fast-profile: **Omnilingual ASR CTC 300M (≈1–2 GB, needs license verification + operator approval per §26 >2 GB rule if larger)**. Everything else (OWSM 1B, Seamless v2 ~9 GB, 7B rescue ~15 GB, DiariZen) is opt-in and gated.

## 6. Test strategy, rollback, definition of done

- Tests: unit (pure functions, contracts), synthetic integration fixtures (generated locally, incl. Hebrew/Persian Unicode cases), opt-in cached-model smoke tests, one E2E offline acceptance command (spec §20.4). CI never downloads weights.
- Rollback: every phase on the feature branch in small commits; old pipeline callable via compat shim until parity tests pass; revert = `git revert` of the phase's commit range; report-schema migrations are forward-only with version stamps, old readers keep working.
- Global definition of done: spec §25 items 1–20, tracked as a checklist in this file at implementation time.

## 7. Blockers requiring operator action (also in RISK_REGISTER)

1. **Project license decision** (blocks release packaging; spec §3.13).
2. **NLLB CC-BY-NC-4.0 weights** — current translation stage is non-commercial-only as shipped; decide deployment posture or approve alternative MT evaluation.
3. **Commit strategy for the dirty working tree** (session fixes) — proposed: commit as-is on `feature/aegis-x-prime` as the first change.
4. **Omnilingual ASR download + license acceptance** (Phase 4's independent family).
5. **Evaluation data acquisition** (all sources operator/license-gated; nothing usable is in the repo).
6. **HF token** remains user-held for pyannote (already understood).
