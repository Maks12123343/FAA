param(
    [string]$RepoDir = "C:\Users\Ukraine\FAA"
)

$ErrorActionPreference = "Stop"
$bridgeDir = Join-Path $RepoDir "gemini_bridge"
$envFile = Join-Path $bridgeDir ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Copy .env.example to .env and fill the local API key and Gemini cookies."
}

Set-Location -LiteralPath $bridgeDir
python server.py
