# Voice Isolator - one-time setup (Windows PowerShell)
# Creates an isolated virtual environment and installs everything locally.
#
# IMPORTANT: builds the venv from a standard (GIL) CPython 3.11/3.12.
# A free-threaded (no-GIL) Python 3.13/3.14 will NOT work - torch and most
# ML wheels have no free-threaded builds.

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

# --- PyTorch (CUDA build for the RTX A1000) ---------------------------------
Write-Host "== Installing PyTorch (CUDA cu124) ==" -ForegroundColor Cyan
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Everything else --------------------------------------------------------
Write-Host "== Installing the rest of the pipeline ==" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

# --- Verify -----------------------------------------------------------------
Write-Host "== Verifying environment ==" -ForegroundColor Cyan
& $py check_env.py

Write-Host ""
Write-Host "Setup complete. Run analyses with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe main.py path\to\recording.mp4" -ForegroundColor Green
