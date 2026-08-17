param(
    [string]$RepoDir = $PSScriptRoot,
    [string]$PythonPath = "$env:USERPROFILE\FAA-venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$bridgeDir = Join-Path $RepoDir "gemini_bridge"
$envFile = Join-Path $bridgeDir ".env"
$bridgeScript = Join-Path $bridgeDir "start_bridge.ps1"

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Run gemini_bridge\setup_browser_profile.ps1 once."
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}

$oldProcesses = Get-CimInstance Win32_Process | Where-Object {
    if ($_.Name -notmatch '^python(3)?\.exe$') { return $false }
    $isFaa = $_.CommandLine -match '(^|[\\/\s"])run\.py([\s"]|$)|(^|[\\/\s"])app\.py([\s"]|$)'
    $isFlowBridge = $_.CommandLine -match 'gemini_bridge[\\/]server\.py' -or (
        $_.ExecutablePath -eq $PythonPath -and
        $_.CommandLine -match '(^|[\\/\s"])server\.py([\s"]|$)'
    )
    return $isFaa -or $isFlowBridge
}
foreach ($process in $oldProcesses) {
    & taskkill.exe /PID $process.ProcessId /T /F 2>$null | Out-Null
}

$bridgeArguments = @(
    "-NoLogo",
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$bridgeScript`"",
    "-RepoDir", "`"$RepoDir`"",
    "-PythonPath", "`"$PythonPath`""
)
Start-Process -FilePath "powershell.exe" -ArgumentList $bridgeArguments -WorkingDirectory $bridgeDir

$health = $null
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:4981/health" -TimeoutSec 2
        break
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $health) {
    throw "Google Flow bridge did not start. Check the separate bridge PowerShell window."
}
if (-not $health.installed -or -not $health.configured) {
    throw "Flow browser profile is not ready. Run gemini_bridge\setup_browser_profile.ps1."
}

$keyLine = Get-Content -LiteralPath $envFile |
    Where-Object { $_ -match '^LOCAL_API_KEY=' } |
    Select-Object -First 1
$localKey = (($keyLine -replace '^LOCAL_API_KEY=', '').Trim().Trim('"').Trim("'"))
if ([string]::IsNullOrWhiteSpace($localKey)) {
    throw "LOCAL_API_KEY is missing in gemini_bridge\.env."
}
try {
    $authHeaders = @{ Authorization = "Bearer $localKey" }
    $authStatus = Invoke-RestMethod "http://127.0.0.1:4981/auth/status" -Headers $authHeaders -TimeoutSec 120
    if (-not $authStatus.ok) { throw $authStatus.detail }
}
catch {
    throw "Google Flow session verification failed. Run gemini_bridge\setup_browser_profile.ps1. $($_.Exception.Message)"
}

Set-Location -LiteralPath $RepoDir
$env:FAA_DEV = "1"
$env:FAA_CORS_ORIGIN = "*"
$env:FAA_PORT = "5050"
Write-Host "Google Flow bridge ready on http://127.0.0.1:4981"
Write-Host "FAA ready on http://127.0.0.1:5050"
& $PythonPath run.py
