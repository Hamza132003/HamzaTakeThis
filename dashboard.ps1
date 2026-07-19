# Launch the Voice Isolator dashboard (local, offline).
# Builds the React UI on first run, then starts the server and opens the browser.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\webapp\frontend\dist\index.html")) {
    Write-Host "Building the UI for the first time (needs Node.js)..." -ForegroundColor Cyan
    Push-Location ".\webapp\frontend"
    try { npm install; npm run build } finally { Pop-Location }
}

Write-Host "Starting dashboard at http://127.0.0.1:5000 ..." -ForegroundColor Green
& .\.venv\Scripts\python.exe webapp\app.py
