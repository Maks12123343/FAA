param(
    [string]$EnvPath = (Join-Path $PSScriptRoot ".env")
)

$ErrorActionPreference = "Stop"

Write-Host "Paste values locally only. Do not send API keys or cookies in chat."
$localKey = Read-Host "LOCAL_API_KEY"
$psid = Read-Host "Value of __Secure-1PSID"
$psidts = Read-Host "Value of __Secure-1PSIDTS"

if ([string]::IsNullOrWhiteSpace($localKey) -or
    [string]::IsNullOrWhiteSpace($psid) -or
    [string]::IsNullOrWhiteSpace($psidts)) {
    throw "All three values are required."
}

function Clean-Value([string]$value) {
    return $value.Trim().Trim('"').Trim("'")
}

$content = @(
    "LOCAL_API_KEY=$(Clean-Value $localKey)"
    "GEMINI_1PSID=$(Clean-Value $psid)"
    "GEMINI_1PSIDTS=$(Clean-Value $psidts)"
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($EnvPath, $content, $utf8NoBom)
Write-Host "Saved Gemini Bridge settings to $EnvPath"
