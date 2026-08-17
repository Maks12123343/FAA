param(
    [string]$PythonPath = "$env:USERPROFILE\FAA-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$bridgeDir = $PSScriptRoot
$envFile = Join-Path $bridgeDir ".env"
$exampleFile = Join-Path $bridgeDir ".env.example"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $exampleFile -Destination $envFile
    $randomKey = -join ((1..64) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
    (Get-Content -LiteralPath $envFile -Raw).Replace(
        "replace-with-a-long-random-local-key",
        $randomKey
    ) | Set-Content -LiteralPath $envFile -Encoding UTF8
    Write-Host "Created gemini_bridge\.env with a random local API key."
}

function Get-EnvValue([string]$Name, [string]$Default) {
    $line = Get-Content -LiteralPath $envFile |
        Where-Object { $_ -match "^$([regex]::Escape($Name))=" } |
        Select-Object -First 1
    if (-not $line) { return $Default }
    $value = ($line -replace "^[^=]+=", "").Trim().Trim('"').Trim("'")
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return [Environment]::ExpandEnvironmentVariables($value)
}

$profile = Get-EnvValue "FLOW_PROFILE" "faa"
$flowHome = Get-EnvValue "FLOW_HOME" (Join-Path $env:LOCALAPPDATA "FAA\flow_browser")
$env:GFLOW_CLI_HOME = $flowHome
$env:GFLOW_CLI_PROFILE = $profile
$env:GFLOW_CLI_HEADLESS = "false"

Write-Host "Installing the pinned Google Flow browser driver..."
& $PythonPath -m pip install "gflow-cli==0.59.0"
if ($LASTEXITCODE -ne 0) { throw "gflow-cli installation failed." }

Write-Host "Opening a dedicated Chrome profile for Google Flow."
Write-Host "Sign in to Google, open Flow, and follow the instructions in that window."
& $PythonPath -m gflow_cli auth login --profile $profile --browser chrome
if ($LASTEXITCODE -ne 0) { throw "Google Flow sign-in was not completed." }

Write-Host "Verifying the saved Flow session (this does not spend credits)..."
& $PythonPath -m gflow_cli auth status --profile $profile
if ($LASTEXITCODE -ne 0) { throw "Flow session verification failed. Run this setup again." }

Write-Host "Google Flow profile is ready: $flowHome\profile_$profile"
