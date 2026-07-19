# HARDWARE_SETUP.md — choosing the right PyTorch build per machine

## Why this exists

The original `setup.ps1` hard-coded the cu124 PyTorch wheel (targeting an RTX A1000).
This machine's GPU — **RTX 5060 Laptop, compute capability sm_120 (Blackwell)** — cannot
execute cu124 kernels (that build supports only sm_50…sm_90). The failure mode was
*silent*: every GPU stage threw "no kernel image is available", the pipeline degraded
through its fallback chains, and output looked superficially successful. This is the
canonical example of the "zero silent model failures" rule.

## The rule for any machine

1. Find your GPU's compute capability:
   `python -c "import torch; print(torch.cuda.get_device_capability(0))"` (or NVIDIA docs).
2. Install a torch build whose supported architectures include it:
   `python -c "import torch; print(torch.cuda.get_arch_list())"`.
   The installed build must list your `sm_XY` (same major, minor ≤ yours) or a
   `compute_XY` PTX entry your driver can JIT.
3. `python -m aegis doctor` performs exactly this comparison and **fails loudly** on
   mismatch. `pipeline.utils.resolve_device` refuses to run "auto"/"cuda" on an
   incompatible build rather than silently falling back — pass `--device cpu` for a
   deliberate CPU run.

## Known-good combinations

| GPU family | Compute capability | Wheel index |
|---|---|---|
| RTX 50-series (Blackwell), e.g. RTX 5060 | sm_120 | `https://download.pytorch.org/whl/cu128` — **validated on this machine: torch 2.11.0+cu128** |
| RTX 30/40-series, A-series (Ampere/Ada) | sm_86 / sm_89 | cu124 or cu128 both work |
| Older (Turing/Pascal) | sm_60–75 | cu118/cu124 per PyTorch support matrix |
| No NVIDIA GPU | — | `./setup.ps1 -Cpu` (CPU wheels; everything runs, slower) |

Do not assume every machine needs this repo's pinned wheel — run `aegis doctor` after any
torch install. Constraint reminder: `clearvoice` requires `numpy<2.0`; setup re-pins it
after torch upgrades (a cu128 upgrade previously pulled numpy 2.x and broke separation).
