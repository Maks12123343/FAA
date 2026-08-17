param(
    [string]$RepoDir = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "$env:USERPROFILE\FAA-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$bridgeDir = Join-Path $RepoDir "gemini_bridge"
$envFile = Join-Path $bridgeDir ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Run setup_browser_profile.ps1 once."
}

Set-Location -LiteralPath $bridgeDir
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}
& $PythonPath -c "import gflow_cli" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Google Flow driver is missing. Run gemini_bridge\setup_browser_profile.ps1 once."
}
& $PythonPath (Join-Path $bridgeDir "server.py")
