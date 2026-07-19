# Voice Isolator / AEGIS-X PRIME - one-time setup (Windows PowerShell)
# Creates an isolated virtual environment and installs everything locally.
#
# GPU default: pinned PyTorch cu128 build (validated on the RTX 5060 Laptop GPU,
# compute capability sm_120). The previous cu124 default CANNOT execute kernels
# on sm_120 hardware and failed silently — see docs/HARDWARE_SETUP.md for how to
# pick the right wheel on a different machine.
#
# CPU-only machines:  ./setup.ps1 -Cpu
#
# IMPORTANT: builds the venv from a standard (GIL) CPython 3.11/3.12.
# A free-threaded (no-GIL) Python 3.13/3.14 will NOT work - torch and most
# ML wheels have no free-threaded builds.

param([switch]$Cpu)

$ErrorActionPreference = "Stop"

# --- Locate a suitable base interpreter -------------------------------------
$candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
$base = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $base) {
    Write-Error "No standard CPython 3.11/3.12 found. Install one from python.org first."
    exit 1
}
Write-Host "== Base interpreter ==" -ForegroundColor Cyan
& $base --version

# --- Fresh virtual environment ----------------------------------------------
if (Test-Path .\.venv) {
    Write-Host "Removing existing .venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .\.venv
}
Write-Host "== Creating virtual environment (.venv) ==" -ForegroundColor Cyan
& $base -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

# --- PyTorch ------------------------------------------------------------------
if ($Cpu) {
    Write-Host "== Installing PyTorch (CPU-only) ==" -ForegroundColor Cyan
    & $py -m pip install "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" `
        --index-url https://download.pytorch.org/whl/cpu
} else {
    # Pinned combination VALIDATED on this machine (RTX 5060 Laptop, sm_120).
    # Other GPUs: check docs/HARDWARE_SETUP.md before assuming this wheel.
    Write-Host "== Installing PyTorch (CUDA cu128, pinned) ==" -ForegroundColor Cyan
    & $py -m pip install "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" `
        --index-url https://download.pytorch.org/whl/cu128
}

# --- Everything else --------------------------------------------------------
Write-Host "== Installing the rest of the pipeline ==" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

# clearvoice requires numpy<2; some torch-adjacent upgrades pull numpy 2.x back
# in. Re-pin last so the constraint always wins (this bit us in production).
& $py -m pip install "numpy<2.0,>=1.24.3"

# --- Verify -----------------------------------------------------------------
Write-Host "== Verifying environment ==" -ForegroundColor Cyan
& $py check_env.py

Write-Host "== Doctor (GPU/torch compatibility gate) ==" -ForegroundColor Cyan
& $py -m aegis doctor
if ($LASTEXITCODE -ne 0) {
    Write-Error "aegis doctor reported a hard failure - do NOT run the pipeline. See output above."
    exit 1
}

Write-Host ""
Write-Host "Setup complete. Run analyses with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe main.py path\to\recording.mp4" -ForegroundColor Green
