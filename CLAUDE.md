# CLAUDE.md — AEGIS-X PRIME project rules

Read before any change. These rules bind all automated work in this repository.

## Non-negotiable research-integrity rules (from the approved specification §1)

- Information absent from the waveform cannot be reconstructed. Mark such intervals
  `[INAUDIBLE]` / `[UNRECOVERABLE]`. Never invent text.
- Raw model scores are `raw_model_score`, never "accuracy" or "confidence", until calibrated.
- No LLM rewrites the transcript as source of truth. Summaries/normalizations must be traceable.
- Never overwrite, delete, or normalize the original recording in place.
- Never silently discard a conflicting hypothesis; keep N-best + provenance.
- Enhancement branches are selected by held-out recognition evidence, not by sound.
- Code license and weight license are recorded separately; "downloadable" ≠ "open source".
- No private, intercepted, classified, or unauthorized audio — ever, including tests.
- No tokens, keys, personal audio, model weights, outputs, or annotations in Git.
- No `--dangerously-skip-permissions`, destructive Git, or blanket deletion.
- No accuracy claims without a benchmark-runner artifact.
- Do not accept model licenses on the operator's behalf; gated downloads are operator actions.
- Downloads >2 GB of new weights require explicit operator approval first.

## Project decisions currently in force

- **License: DEFERRED by operator (2026-07-19).** Redistribution and release packaging are
  BLOCKED (see RELEASE_BLOCKED.md, docs/LICENSE_DECISION.md). Internal development and local
  hackathon evaluation may continue. Do not add any project license without explicit approval.
- **NLLB stays**, clearly marked CC-BY-NC-4.0 (non-commercial). Never describe it as
  commercially deployable. Reports must record translation model + license profile.
- **Omnilingual ASR download deferred** until Phase 4 operator approval (see MODEL_MANIFEST.yaml
  `pending_operator_approval`).
- **This machine:** RTX 5060 Laptop, sm_120 → torch **cu128+** required; cu124 kernels cannot
  execute here. `setup.ps1` pins the working combination; `python -m aegis doctor` must pass
  before pipeline runs. See docs/HARDWARE_SETUP.md for other machines.
- **OneDrive:** the repo currently lives in a synced folder. Future artifact/DB stores must go
  to `storage.artifact_dir` (non-synced). Do not create SQLite/WAL files inside the repo tree.

## Commands

```powershell
./setup.ps1                        # one-time env setup (GPU default; -Cpu for CPU-only)
./dashboard.ps1                    # build UI (first run) + start server at 127.0.0.1:5000
.\.venv\Scripts\python.exe -m aegis doctor       # environment + GPU-compat + gate checks
.\.venv\Scripts\python.exe -m pytest             # unit tests (no model downloads)
.\.venv\Scripts\python.exe tools\secret_scan.py  # secret scan over tracked files
.\.venv\Scripts\python.exe tools\release_gate.py # release gate (expected BLOCKED for now)
.\.venv\Scripts\python.exe -m ruff check aegis tools tests pipeline/asr
.\.venv\Scripts\python.exe -m mypy aegis tools pipeline/asr
```

## Test expectations

- Unit tests and CI run with **no network model access** and no cached-weight dependency.
- Heavy-import tests carry the `heavy` marker; CI runs `-m "not heavy"`, the full suite runs
  locally where the venv has all runtime deps.
- Every new pure function gets a unit test. Model adapters get deterministic fakes.

## No-claim policy

Allowed phrasing: "research prototype", "offline-capable", "confidence-calibrated after
validation", "best on the named benchmark" (only with a run artifact). Banned: "uncrackable",
"perfect", "guaranteed", "military-grade", "language-agnostic" (for the emotion model), and any
unverified accuracy figure.
