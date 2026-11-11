# Create and populate the Python 3.11 virtual environment for PyCaret support.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python 3.11 first."
}

py -3.11 --version
if (-not (Test-Path ".venv311")) {
    py -3.11 -m venv .venv311
}

& .\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools
pip install `
    "streamlit==1.39.*" `
    "pandas==2.1.*" `
    "numpy==1.26.*" `
    "pyarrow==16.*" `
    "openpyxl==3.1.*" `
    "ydata-profiling==4.10.*" `
    "autoviz==0.1.901" `
    "numpy==1.26.4" `
    "scipy==1.11.4" `
    "scikit-learn==1.4.2" `
    "joblib==1.3.2" `
    "xgboost==2.1.*" `
    "lightgbm==4.5.*" `
    "catboost==1.2.*" `
    "statsmodels==0.14.*" `
    "pycaret==3.3.*" `
    "shap==0.46.*" `
    "category-encoders==2.6.*" `
    "feature-engine==1.8.*" `
    "featuretools==1.31.*" `
    "matplotlib==3.7.*" `
    "seaborn==0.13.*" `
    "plotly==5.24.*" `
    "nbformat>=5.9" `
    "pytest==8.3.*"

Write-Host ""
Write-Host "Python 3.11 venv ready at $Root\.venv311"
Write-Host "Activate with: .\.venv311\Scripts\Activate.ps1"
Write-Host "Run GUI:       streamlit run app.py"
Write-Host "Run headless:  python run_headless.py --data data\sample\boston_sample.csv --task regression --target MEDV"