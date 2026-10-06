$ErrorActionPreference = 'Stop'
$uiPython = 'D:\Projects\ML-Assignment-streamlit-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $uiPython)) {
    throw 'Streamlit environment not found. See docs/STREAMLIT_APP.md'
}
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
& $uiPython -B -m streamlit run (Join-Path $PSScriptRoot 'streamlit_app.py') --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
