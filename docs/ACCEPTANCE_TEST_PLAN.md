# ACCEPTANCE TEST PLAN — AEGIS-X PRIME

Scope: how every phase proves itself, and what "accepted" means for the whole system.
Binding constraints: CI never downloads model weights (spec §6.12); no accuracy claim exists until the
benchmark runner emits it from a run artifact (spec §1, §25.18).

## 1. Test tiers

### T1 — Unit (every commit, CI, no network)
- Contracts: round-trip serialize/validate every Pydantic schema; schema-version bump detection.
- RTL/Unicode normalization: Hebrew final forms, niqqud stripping policy, Persian ye/kaf vs Arabic ye/kaf, ZWNJ handling, mixed-digit (٠١٢/۰۱۲/012) normalization — raw form always retained.
- Branch scoring regression (defect 3.2): construct a long correct hypothesis and a short truncated one with realistic avg_logprobs; assert the old formula prefers the short one (documents the bug) and the new segment utility prefers the correct one.
- ROVER: null/deletion voting bins; tie-breaking; condition-weight softmax; diversity-penalty capping of correlated families.
- MBR selection on a fixed toy N-best with known minimizer.
- Calibration math: ECE/Brier/NLL on hand-computed fixtures; temperature-scaling closed-form checks.
- Overlap: active-speaker-set construction from synthetic turns; 3-speaker region → `overlap_unresolved`; no pairwise duplication of the same acoustic words.
- Hashing/manifests: content-address stability, atomic-write crash simulation, manifest completeness.
- Config migration: legacy config.yaml → new schema with deprecation warnings; unknown keys fail loudly.
- Security: no-secret logging (token patterns redacted), path-traversal rejection on media/delete routes.
- Cancellation: JobCancelled propagates through broad exception handlers of every stage wrapper (regression for the re-raise pattern).

### T2 — Synthetic integration fixtures (CI, generated at test time, no weights)
Deterministic generators produce: clean single speaker (TTS or tone-modulated), pure silence, hum (50/60 Hz + harmonics), stationary noise, clipping runs, 300–3400 Hz band-limit "radio", two-speaker overlap mix, three-speaker overlap mix, stereo with speakers hard-panned to separate channels, dropout/packet-loss gaps, Hebrew/Persian Unicode text fixtures, conflicting fake-ASR hypothesis sets.
All heavy models replaced by deterministic fake adapters implementing the same adapter interface.
Assertions: condition vector detects each degradation; channel factory preserves both stereo channels; VAD-rescue recovers a region deliberately excluded from the primary detector; router escalates on planted disagreement; fusion output carries provenance links for every consensus word; report validates against schema.

### T3 — Cached-model smoke tests (opt-in, `aegis models smoke`, local only)
Per adapter: loads pinned revision, reports capabilities, transcribes a 5 s fixture, handles empty/silence without output, frees/reuses memory within profile watermark, records structured failure without killing the batch. Runs only when weights are already cached; never in CI.

### T4 — End-to-end offline acceptance (`aegis offline-test --fixture-pack tests/fixtures/legal_demo`)
Network access disabled at the process level (socket guard); full pipeline on a small legal fixture pack; emits report JSON/MD/HTML + metrics + manifest + audit log; zero unhandled exceptions; every fallback that fired is present in the report; run is reproducible (same hashes in, same manifest out, modulo timestamps).

### T5 — Scientific benchmark runs (operator-triggered, data-gated)
`aegis evaluate --manifest … --system {raw_whisper | current_pipeline | branch_X | family_Y | rover | full | full+calibration}` and `aegis compare` with paired block bootstrap (block = source file/speaker/session; words are never treated as independent). Metrics per spec §16.6: WER/CER, cpWER/tcpWER/SA-WER, DER/JER, VAD miss/FA, chrF++/COMET/BLEU + entity/number/negation preservation, ECE/adaptive-ECE/Brier/NLL/AURC, hallucinated-words-per-non-speech-minute, insertion rate on silence, RTF/files-per-hour/peak VRAM, failure/fallback rates. All stratified by language and condition.

## 2. Predeclared acceptance gates (spec §16.9 — comparative, not absolute)

1. Statistically significant WER/CER improvement vs best single-model baseline on the locked degraded test set (paired bootstrap, 95% CI excluding zero), separately for Hebrew AND Persian.
2. Clean-audio non-inferiority within a predeclared margin (default 1% absolute WER, to be ratified with the operator before the first locked run).
3. Severe/overlap strata: improvement OR justified abstention (risk-coverage dominance at matched coverage).
4. Calibration: test-set ECE and Brier at or below predeclared thresholds set from pilot dev data (never tuned on test); reliability diagram artifact required.
5. Critical tokens: number/digit accuracy, entity F1, negation accuracy meeting application gates agreed with the operator; rule-of-three reporting for zero-event bounds (zero critical failures in n opportunities → ≤3/n at 95%).
6. Zero silent model failures across the acceptance run; every fallback visible in reports (spec §25.16 — the cu124-torch incident is the canonical counterexample this gate exists to catch).
7. Reproducibility: clean-environment run from pinned artifacts reproduces headline metrics within bootstrap noise.
8. Hallucination: no increase in hallucinated-words-per-non-speech-minute vs the raw-Whisper baseline at system defaults.

## 3. Sample-size planning (spec §16.7–16.8)

Hackathon evidence package (per language): 500–1,000 clips, ≥30 speakers, ≥20,000 reference words, strata {clean, moderate, severe, narrow-band, clipping, overlap, no-speech}; 3–4 controlled degraded variants per source; augmented copies never counted as independent sources. Naive binomial n≈6,147 words for ±1% at p=0.20 is a floor only — final n set from pilot variance + design effect of blocked sampling.

## 4. Phase-gate table

| Phase | Gate to pass before next phase |
|---|---|
| P0 | CI green on clean clone; lockfile installs; session fixes committed; secret scan clean |
| P1 | T1 contract+diagnostic tests; byte-identical mono via compat shim |
| P2 | 3.2-regression red→green; branch manifests content-addressed; T2 degradation detection |
| P3 | T2 rescue/overlap/code-switch fixtures; dead-guard replaced (3.7) |
| P4 | Two independent families on fixture pack (T3); fail-closed language check test |
| P5 | Fusion provenance links complete; RTL suite green |
| P6 | Dev-only training verified; reliability artifacts emitted |
| P7–P9 | Consistency suites; schema migration tests; UI raw-vs-calibrated distinction |
| P10–P11 | Deterministic re-run of a manifest; baseline table 1–9 produced |
| P12 | Kill-and-resume test; no duplicate work on resume; quarantine path test |
| P13 | Offline-mode T4 with sockets disabled; audit log append-only test |
| P14–P16 | Full T1–T4 sweep; demo cases 1–3 reproducible |

## 5. Out of scope for automated acceptance

Human bilingual adequacy review (spec §16.6 translation), annotation quality management, and legal
sign-off are operator-run processes; the framework only records their artifacts.
