param(
    [string]$RepoDir = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "$env:USERPROFILE\FAA-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$bridgeDir = Join-Path $RepoDir "gemini_bridge"
$envFile = Join-Path $bridgeDir ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Copy .env.example to .env and fill the local API key and Gemini cookies."
}

Set-Location -LiteralPath $bridgeDir
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}
& $PythonPath server.py
