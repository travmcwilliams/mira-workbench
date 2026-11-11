# MIRA local setup (Python 3.11)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found on PATH"
}

python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .

Write-Host ""
Write-Host "Optional: playwright install chromium"
Write-Host "Run app: streamlit run app.py"