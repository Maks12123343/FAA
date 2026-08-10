param(
    [string]$RepoDir = "C:\Users\Ukraine\FAA",
    [string]$Out = "D:\youtube",
    [string]$SshHost = "115.78.134.198",
    [int]$SshPort = 48201,
    [int]$LocalPort = 5050,
    [int]$RemotePort = 5050,
    [double]$IntervalMinutes = 1,
    [string]$StateFile = "",
    [switch]$WatchNewOnly
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoDir

$tunnelCommand = @"
while (`$true) {
    Write-Host "[tunnel] opening SSH tunnel on localhost:$LocalPort ..."
    ssh -N -p $SshPort root@$SshHost -L ${LocalPort}:localhost:${RemotePort} -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
    Write-Host "[tunnel] tunnel stopped. Restarting in 5 seconds..."
    Start-Sleep -Seconds 5
}
"@

Write-Host "[auto] starting tunnel watcher in a separate PowerShell window"
Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    $tunnelCommand
)

Start-Sleep -Seconds 3

$downloadArgs = @(
    "download_ready_from_site.py",
    "--base-url", "http://localhost:$LocalPort",
    "--out", $Out,
    "--watch",
    "--interval-minutes", ([string]$IntervalMinutes),
    "--retries", "5",
    "--download-timeout", "7200"
)

if ($WatchNewOnly) {
    $downloadArgs += "--watch-new-only"
}

if ($StateFile) {
    $downloadArgs += "--state-file"
    $downloadArgs += $StateFile
}

Write-Host "[auto] starting downloader"
Write-Host "[auto] output folder: $Out"
Write-Host "[auto] interval: $IntervalMinutes minute(s)"
if ($WatchNewOnly) {
    Write-Host "[auto] mode: only projects that become ready after startup"
} else {
    Write-Host "[auto] mode: download every ready project that is not already in local state"
}

python @downloadArgs
