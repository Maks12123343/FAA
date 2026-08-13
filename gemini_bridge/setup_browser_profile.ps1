param(
    [string]$PythonPath = "$env:USERPROFILE\FAA-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$bridgeDir = $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "Installing Playwright in: $PythonPath"
& $PythonPath -m pip install playwright
if ($LASTEXITCODE -ne 0) { throw "Playwright installation failed." }

Write-Host "Opening the persistent Gemini browser profile..."
& $PythonPath (Join-Path $bridgeDir "setup_browser_profile.py")
if ($LASTEXITCODE -ne 0) { throw "Gemini browser profile setup failed." }
