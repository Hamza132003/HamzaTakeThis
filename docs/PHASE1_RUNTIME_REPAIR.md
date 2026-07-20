# PHASE 1 RUNTIME REPAIR — environment regression, diagnosis, and rebuild

## 1. The regression (recorded before removal)

During the Phase 1 implementation session the repository's `.venv` was replaced by
an environment built from **Python 3.14**, which this project does not support
(`setup.ps1` requires standard CPython 3.11/3.12; torch publishes no 3.14 wheels).

Evidence captured before deletion:

```
# .venv/pyvenv.cfg
home = C:\Python314
include-system-site-packages = false
version = 3.14.0
executable = C:\Python314\python.exe
command = C:\Python314\python.exe -m venv C:\Users\USER\...\HamzaTakeThis\.venv

# .venv/Scripts/python.exe --version
Python 3.14.0
```

### Symptoms observed (in the order they surfaced)

1. `mypy` aborted with `ModuleNotFoundError: No module named 'librt.internal'`.
2. `pydantic` aborted with `ModuleNotFoundError: No module named
   'pydantic_core._pydantic_core'` — a Python 3.12-tagged binary
   (`_pydantic_core.cp312-win_amd64.pyd`) that a 3.14 interpreter cannot load.
3. `numpy` aborted with `OverflowError: cannot convert longdouble infinity to
   integer` — the classic symptom of an ABI/interpreter mismatch.
4. `numpy/core/` contained **cp314** binaries (14 files, 0 cp312) — i.e. pip had
   correctly installed 3.14 wheels into what was now a 3.14 environment.
5. `torch` failed to import entirely (no CUDA wheels exist for 3.14), which also
   left the Flask dashboard non-functional.

### Cause

The `command =` line in `pyvenv.cfg` records that the virtual environment was
recreated with `C:\Python314\python.exe -m venv`. The machine has both
`C:\Python314` (a pre-existing standalone install) and the project's required
`%LOCALAPPDATA%\Programs\Python\Python312`. The rebuild used the wrong base
interpreter. This occurred outside the AEGIS Phase 1 code work; no committed
Phase 1 change selects an interpreter.

### Why the environment was not preserved

The directory contained only installed packages (`Include/`, `Lib/`, `Scripts/`,
`share/`, `pyvenv.cfg`) — no source, configuration, credentials, or data. It was
therefore rebuilt rather than archived, per the operator's instruction.

## 2. Prevention

- `setup.ps1` already validates the base interpreter, but only searched
  `%LOCALAPPDATA%\Programs\Python\Python31{1,2}` — it never *rejected* a wrong
  pre-existing `.venv`. `python -m aegis doctor` now fails loudly when the
  interpreter is outside the supported range, so the state is detectable before
  a pipeline run rather than discovered through cascading import errors.
- The canonical rebuild command is always `./setup.ps1` (GPU) or
  `./setup.ps1 -Cpu`; never `python -m venv` with an arbitrary interpreter.

## 3. Rebuild performed

Base interpreter verified **before** removal:
`%LOCALAPPDATA%\Programs\Python\Python312\python.exe` → `Python 3.12.10`.

Rebuild executed with the repository's approved process (`./setup.ps1`, pinned
cu128 PyTorch family, `numpy<2` re-pin, `check_env.py` + `aegis doctor` gates),
exit code 0. No new model weights were downloaded; only already-cached weights
were used for smoke tests.

Resulting runtime:

| Component | Version |
|---|---|
| Python | 3.12.10 (CPython) |
| pip | 26.1.2 |
| torch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA runtime (torch) | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute capability | sm_120 |
| torch arch list | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| numpy | 1.26.4 |
| pydantic | 2.13.4 |

`aegis doctor` → **RESULT: OK** (binary kernels present, sm_120 runs on sm_120).
`check_env.py` → 12/14 components, GPU/torch compatibility **[OK]**.

## 4. Open defect found by this rebuild: pyannote.audio vs torchaudio 2.11

`import pyannote.audio` fails:

```
AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
```

- Installed: `pyannote.audio 3.4.0` (requirements pin is `pyannote.audio>=3.1`).
- torchaudio 2.11 removed `torchaudio.AudioMetaData` and the whole
  `torchaudio.backend` module; pyannote.audio 3.4 still references them.
- **Introduced by the Phase 0 cu128 repair** (torch/torchaudio 2.6 → 2.11), and
  missed at the time because `check_env.py`'s pyannote line was already a
  documented false-MISS caused by the SpeechBrain lazy-`k2` import-order
  artifact — the real breakage hid behind a known-false negative.
- **Impact:** the diarization stage raises, is caught by its own handler, and
  degrades to single-speaker energy segmentation with the loud user-visible
  warning it was designed to emit. No silent success. On this machine there is
  also no HF token, so diarization would have run degraded regardless.
- **Not repaired in this pass**: the operator's instruction is to report the
  exact failure rather than install packages ad hoc outside the documented
  setup process. Proposed fix for approval: pin `pyannote.audio>=4.0` in
  `requirements.txt` (the config already targets
  `pyannote/speaker-diarization-community-1`, a pyannote 4.x model, and
  `pipeline/diarization.py` already handles the 4.x `token=` API rename), then
  re-run `setup.ps1`. This is a package change, not a model-weight download,
  but it warrants explicit approval because it changes a pinned dependency.
- Follow-up hardening: `check_env.py` should distinguish "import failed for a
  real reason" from the known lazy-import artifact so this class of breakage
  cannot hide again.
