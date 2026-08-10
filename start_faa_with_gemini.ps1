param(
    [string]$RepoDir = "C:\Users\Ukraine\FAA"
)

$ErrorActionPreference = "Stop"
$bridgeDir = Join-Path $RepoDir "gemini_bridge"
$envFile = Join-Path $bridgeDir ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Copy .env.example to .env and fill the bridge settings first."
}

$python = (Get-Command python -ErrorAction Stop).Source
$bridgeCommand = "Set-Location -LiteralPath '$bridgeDir'; & '$python' server.py"
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoLogo", "-NoExit", "-Command", $bridgeCommand) -WorkingDirectory $bridgeDir
Start-Sleep -Seconds 2

Set-Location -LiteralPath $RepoDir
$env:FAA_DEV = "1"
$env:FAA_CORS_ORIGIN = "*"
$env:FAA_PORT = "5050"
& $python run.py
