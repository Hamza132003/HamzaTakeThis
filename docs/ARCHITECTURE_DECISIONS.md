# ARCHITECTURE DECISIONS — AEGIS-X PRIME

Format: lightweight ADRs. Status values: PROPOSED (needs approval), ACCEPTED, SUPERSEDED.

## ADR-001: Migrate by wrapping, not rewriting — PROPOSED
The existing pipeline is a working prototype with real Windows-hardening value (speechbrain lazy-import guard, torchcodec bypass, chunked enhancement). New spec-§5 subpackages are built alongside; old modules become façades; deletion only after parity tests. Rejected alternative: greenfield rewrite (violates spec §0, loses hardened edge cases).

## ADR-002: Typed contracts with Pydantic v2 — PROPOSED
`pipeline/contracts/*` uses Pydantic models with `schema_version` fields (AudioAsset, DerivedAudio, ConditionVector, SpeechRegion, DiarizationTurn, ASRHypothesis, WordHypothesis, EvidenceNode/Edge, ConsensusWord, TranslationHypothesis, EvidenceManifest, EvaluationRecord). Pydantic is already an installed transitive dep and gives JSON-schema export for the report contract. Rejected: bare dataclasses (no validation), attrs (extra dep, no JSON schema out of the box).

## ADR-003: Evidence store is content-addressed files + SQLite index — PROPOSED
Single-machine pilot: artifacts under `store/<sha256[:2]>/<sha256>/`, SQLite for job/DAG state (WAL mode). PostgreSQL only when multi-worker production exists. **Must live outside the OneDrive-synced tree** (observed sync interference this session). Atomic writes via temp-file + rename.

## ADR-004: Keep Flask + React dashboard; no framework migration — PROPOSED
The UI is a working, pure-CSS Vite/React app. AEGIS additions (hypothesis browser, confidence heatmap, review edits) are new routes/components, not a rework. Localhost-only listener stays; auth added only if exposure ever changes.

## ADR-005: Second ASR family = Omnilingual ASR CTC (300M) — PROPOSED
Chosen for: structurally independent (CTC vs Whisper's attention decoder), explicit he/fa coverage claims (must be verified programmatically at adapter startup, fail-closed), size fits the 8 GB sequential budget. OWSM-CTC v4 is the challenger, gated behind measured validation gain (spec §10.4). Rejected as "second family": more Whisper variants (correlated — spec forbids), wav2vec2 monolingual checkpoints (fragmented he/fa quality, unclear provenance) — these may still become Trees E/F specialists.

## ADR-006: Fusion = ROVER/confusion network first, MBR optional — PROPOSED
ROVER with condition-dependent weights and a diversity penalty is implementable without training data beyond dev-set weight fitting and is inspectable (full voting breakdown persisted). MBR added later as an ablation-gated alternative. LLM never assembles the transcript (spec §11.5).

## ADR-007: Confidence = feature-based binary correctness model + post-hoc calibration — PROPOSED
Gradient-boosted trees and logistic regression benchmarked; isotonic/beta/temperature as post-hoc calibrators; selection on dev Brier/ECE/risk-coverage. Raw scores renamed `raw_model_score` everywhere in schemas (spec §1).

## ADR-008: Cancellation stays cooperative (per-segment/per-batch checks) — ACCEPTED (implemented pre-plan, this session)
`pipeline.utils.set_cancel_check/check_cancel` polled inside Whisper's lazy segment generator and NLLB batch loop; JobCancelled re-raised past broad handlers. Measured cancel latency ~32 s mid-transcribe. Process-kill rejected (loses model cache, corrupts partial artifacts). The DAG (Phase 12) inherits this contract per stage.

## ADR-009: Overlap representation = active-speaker-set timeline — PROPOSED
Replaces pairwise overlap records (whose 3+-speaker guard is provably dead code — see BASELINE_AUDIT §5/3.7). Two-source separation runs only on exact-2 sets with sufficient margin; 3+ → `overlap_unresolved`, mixed evidence preserved.

## ADR-010: VAD = high-recall union + conservative consensus, diarization never sole gate — PROPOSED
pyannote activity + Silero (already bundled with faster-whisper — zero new weights) + energy; per-detector support stored on every SpeechRegion; ASR rescue scan over uncovered speech-like spans. Fixes defect 3.6 without abandoning gating's anti-hallucination benefit.

## ADR-011: Hardware profile is explicit config, detected at doctor-time — PROPOSED
`hardware_profile: {gpu_8gb, gpu_6gb, gpu_24gb, gpu_48gb, cpu}` selected/verified by `aegis doctor`. Note: dev machine is RTX 5060 sm_120 → torch cu128+ is a hard requirement (cu124 kernels absent; caused silent GPU-stage failure — the motivating example for fail-loud model health checks in Phase 13).

## ADR-012: Deterministic summary, experimental emotion — PROPOSED
Summary: `do_sample=False`, OBSERVATIONS vs INFERENCES sections, timestamp citations, excluded from forensic profile by default. Emotion: default OFF, `experimental: true` label, README corrected (current model is English-trained; "language-agnostic" claim removed) until a target-domain validation study exists.

## ADR-013: Model acquisition pinned + hashed; no floating `main` — PROPOSED
MODEL_MANIFEST.yaml records repo ID, exact revision, file hashes, license, gating status. Downloads go through one acquisition module (HTTPS, size + checksum verified) — replaces ad-hoc downloads incl. panns' HTTP wget. `trust_remote_code=True` forbidden unless a pinned revision is reviewed and allowlisted.
