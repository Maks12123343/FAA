# Keeps Google Drive alive during long production runs.
#
# Drive's own log says "not restarting and exiting" after a fatal CHECK, so a
# crash leaves the mount gone until someone restarts it by hand. This watches
# the mount and relaunches Drive when it disappears.
#
# Run it in its own PowerShell window before starting production:
#   powershell -ExecutionPolicy Bypass -File drive_watchdog.ps1
# Stop it with Ctrl+C when the batch is done.

param(
    [string]$MountPath = 'E:\',
    [int]$CheckSeconds = 15,
    [int]$SettleSeconds = 90,
    [switch]$Once
)

$ErrorActionPreference = 'Continue'

function Get-DriveExe {
    $root = 'C:\Program Files\Google\Drive File Stream'
    $versions = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d' } |
        Sort-Object { [version]($_.Name) } -Descending
    foreach ($v in $versions) {
        $exe = Join-Path $v.FullName 'GoogleDriveFS.exe'
        if (Test-Path $exe) { return $exe }
    }
    return $null
}

$exe = Get-DriveExe
if (-not $exe) {
    Write-Host "[watchdog] Could not find GoogleDriveFS.exe. Is Drive installed?" -ForegroundColor Red
    exit 1
}

Write-Host "[watchdog] watching $MountPath every ${CheckSeconds}s"
Write-Host "[watchdog] will relaunch: $exe"
Write-Host "[watchdog] Ctrl+C to stop"

$restarts = 0
$wasDown = $false

while ($true) {
    $up = Test-Path -LiteralPath $MountPath

    if ($up) {
        if ($wasDown) {
            Write-Host ("[watchdog] {0}  mount is back after restart #{1}" -f (Get-Date -Format 'HH:mm:ss'), $restarts) -ForegroundColor Green
            $wasDown = $false
        }
    }
    else {
        $wasDown = $true
        $restarts++
        Write-Host ("[watchdog] {0}  {1} is GONE - Drive crashed. Restarting (#{2})..." -f (Get-Date -Format 'HH:mm:ss'), $MountPath, $restarts) -ForegroundColor Yellow

        # Clear out any half-dead processes first, otherwise the new one exits.
        Get-Process -Name 'GoogleDriveFS' -ErrorAction SilentlyContinue | ForEach-Object {
            try { $_.Kill() } catch { }
        }
        Start-Sleep -Seconds 3

        try {
            Start-Process -FilePath $exe -ErrorAction Stop
        }
        catch {
            Write-Host "[watchdog] could not start Drive: $($_.Exception.Message)" -ForegroundColor Red
        }

        # Give the mount time to appear before checking again.
        for ($i = 0; $i -lt $SettleSeconds; $i += 5) {
            Start-Sleep -Seconds 5
            if (Test-Path -LiteralPath $MountPath) {
                Write-Host ("[watchdog] {0}  mount returned after {1}s" -f (Get-Date -Format 'HH:mm:ss'), ($i + 5)) -ForegroundColor Green
                $wasDown = $false
                break
            }
        }
        if ($wasDown) {
            Write-Host "[watchdog] mount still missing after ${SettleSeconds}s - may need manual sign-in" -ForegroundColor Red
        }
        if ($Once) { break }
    }

    Start-Sleep -Seconds $CheckSeconds
}
