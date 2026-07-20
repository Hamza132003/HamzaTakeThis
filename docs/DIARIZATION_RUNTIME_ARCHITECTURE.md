# DIARIZATION RUNTIME ARCHITECTURE — why diarization runs in its own environment

## 1. The dependency conflict (measured, not assumed)

Two hard requirements in this project are mutually unsatisfiable in one
environment:

| Package | Requirement | Source of truth |
|---|---|---|
| `clearvoice 0.1.2` (enhancement + separation) | `numpy>=1.24.3,<2.0` | `importlib.metadata.requires("clearvoice")` in the main venv |
| `pyannote.audio 4.0.7` (modern diarization) | pulls `pyannote-core 6.0.1` → `numpy>=2.0` | `pip install --dry-run --report` against the main venv |

A non-mutating resolver run (`pip install --dry-run --report`) of
`pyannote.audio==4.0.7` against the main environment resolves to **numpy 2.5.1**
and would additionally install/replace 18 packages, including
`pyannote-core 6.0.1`, `pyannote-database 6.1.1`, `pyannote-metrics 4.1`,
`pyannote-pipeline 4.0.0`, `pyannoteai-sdk 0.4.0`, `torchcodec 0.15.0` and the
OpenTelemetry stack.

Installing that into the main environment would silently break ClearVoice — the
enhancement stage that the whole pipeline depends on.

### Why the previously shipped state was also wrong

The main environment currently holds `pyannote.audio 3.4.0`, which is **broken**
against the pinned `torchaudio 2.11.0+cu128`:

```
AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
```

torchaudio 2.11 removed `torchaudio.AudioMetaData` and the whole
`torchaudio.backend` module; pyannote.audio 3.4 still references them. So the
main environment cannot do diarization either way: pyannote 3 is broken by
torch, and pyannote 4 is blocked by numpy.

### Explicitly rejected "fixes"

Each of these was considered and rejected because it trades a visible failure
for a hidden one:

- upgrading the main environment to numpy 2 (breaks ClearVoice);
- removing ClearVoice (it is the primary enhancement path);
- `pip install --no-deps` for pyannote 4 (produces an environment the resolver
  never validated; failures would surface as arbitrary runtime errors);
- suppressing resolver warnings;
- monkeypatching third-party `site-packages` (e.g. re-adding
  `torchaudio.AudioMetaData`) — unsupported, invisible to `pip check`, and
  silently wrong across upgrades;
- downgrading torch/torchaudio independently (the cu128 build is required for
  the RTX 5060's sm_120 kernels — see docs/HARDWARE_SETUP.md);
- treating a successful `import pyannote.audio` as proof that diarization works.

## 2. Chosen architecture: process isolation, not dependency negotiation

```
 MAIN ENVIRONMENT  (repo .venv, numpy 1.26.4)      DIARIZATION WORKER ENV
 ------------------------------------------       -----------------------------
 ClearVoice / MossFormer2 enhancement + sep        pyannote.audio 4.0.7
 faster-whisper transcription                      pyannote-core 6.0.1
 NLLB translation, emotion, summary                numpy 2.x
 Flask dashboard, orchestration                    torch/torchaudio 2.11.0+cu128
 pipeline/diarization.py  ── subprocess ──────►    workers/diarization_worker
        (typed JSON request/response over files)
```

- The main process **never imports pyannote**. It launches the worker with the
  worker environment's own interpreter, using an argument list (never
  `shell=True`), and exchanges a typed, versioned JSON request/response.
- The worker environment lives **outside the OneDrive repository** at
  `%LOCALAPPDATA%\AegisXPrime\envs\diarization` (override:
  `diarization.worker_python` in `config.yaml`).
- Both environments pin the same `torch/torchaudio 2.11.0+cu128` family so the
  GPU story is identical; they differ only in the numpy generation and the
  diarization stack.
- Because the two never share a process, `pip check` passes in each
  environment independently — the conflict is *eliminated*, not silenced.

## 3. Legacy pyannote 3 in the main environment

`pyannote.audio` is removed from the main runtime dependency set
(`requirements.txt`) so it cannot be mistaken for the active backend. The main
pipeline's diarization stage is an adapter that talks to the worker; if the
worker environment is absent, the stage reports
`method=fallback_single_speaker` with an explicit failure stage — never
"diarization succeeded".

## 4. Audio input policy

pyannote is given a **preloaded in-memory waveform** (tensor + sample rate),
not a file path, so `torchcodec` is not the runtime file decoder on Windows
(its DLLs are a known failure point there). torchcodec is still installed
because pyannote 4 declares it, and its import is verified and reported by
`aegis doctor` — an import failure is surfaced, never hidden.

The worker reads the pipeline's existing **16 kHz mono** asset
(`voice_16k.wav` / the canonical `mono_mix` equivalent), which is already
mono by construction, so no additional downmix rule is applied inside the
worker. The canonical audio asset is opened read-only and never modified.
Memory note: pyannote 4's community-1 pipeline requires the **whole waveform**
in memory for a single inference call; at 16 kHz mono float32 this is ~3.8 MB
per audio-minute (~230 MB for a 60-minute recording). That is an unavoidable
property of the model API and is recorded here rather than worked around.

## 5. VRAM policy (8 GB RTX 5060)

The worker is launched **sequentially**: the main process evicts the preceding
heavy enhancement model and releases CUDA cache through the existing
`pipeline.models.MANAGER` before starting the worker, so ClearVoice and
pyannote are never resident at the same time. The worker process exits after
the stage, which returns its VRAM to the driver deterministically. A persistent
worker may be introduced only once the Phase 12 scheduler owns model residency.
