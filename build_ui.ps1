# Build the React dashboard (needs Node.js). One-time, or after UI changes.
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\webapp\frontend"
try {
    Write-Host "== Installing UI dependencies ==" -ForegroundColor Cyan
    npm install
    Write-Host "== Building UI ==" -ForegroundColor Cyan
    npm run build
    Write-Host "UI built to webapp/frontend/dist" -ForegroundColor Green
} finally {
    Pop-Location
}
