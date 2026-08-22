# Collects evidence about why Google Drive File Stream keeps crashing.
# Run on the machine where the mount disappears:
#   powershell -ExecutionPolicy Bypass -File diag_drive_crash.ps1
# Writes a report next to itself and prints a summary.

$ErrorActionPreference = 'Continue'
$report = Join-Path $PSScriptRoot 'drive_crash_report.txt'
$out = New-Object System.Collections.Generic.List[string]

function Add-Line { param([string]$Text) $out.Add($Text); Write-Output $Text }
function Add-Head { param([string]$Text) Add-Line ''; Add-Line ('=' * 70); Add-Line $Text; Add-Line ('=' * 70) }

Add-Head 'GOOGLE DRIVE CRASH DIAGNOSTIC'
Add-Line ("collected: " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Add-Line ("computer : $env:COMPUTERNAME   user: $env:USERNAME")

# ── 1. Drive version and processes ──────────────────────────────────────────
Add-Head '1. DRIVE VERSION AND PROCESSES'
$drv = Get-ItemProperty 'HKLM:\SOFTWARE\Google\DriveFS' -ErrorAction SilentlyContinue
if ($drv) { Add-Line ("driver version : " + $drv.DriverVersion) }
$installed = Get-ChildItem 'C:\Program Files\Google\Drive File Stream' -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d' } | Select-Object -ExpandProperty Name
if ($installed) { Add-Line ("app versions   : " + ($installed -join ', ')) }
$procs = Get-Process -Name '*GoogleDrive*' -ErrorAction SilentlyContinue
if ($procs) {
    foreach ($p in $procs) {
        Add-Line ("running: {0} pid={1} ram={2} MB started={3}" -f $p.Name, $p.Id, [math]::Round($p.WorkingSet64/1MB), $p.StartTime)
    }
} else {
    Add-Line 'running: NONE - Drive is not running right now'
}

# ── 2. Where the cache is and how big Drive lets it get ─────────────────────
Add-Head '2. CACHE LOCATION AND CAPACITY'
$default = Join-Path $env:LOCALAPPDATA 'Google\DriveFS'
$roots = @($default)
foreach ($v in (Get-Volume | Where-Object DriveLetter)) {
    $roots += ("{0}:\DriveFS" -f $v.DriveLetter)
}
foreach ($r in ($roots | Select-Object -Unique)) {
    if (Test-Path $r) {
        $sz = (Get-ChildItem $r -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum
        Add-Line ("found cache dir: {0}  ({1} GB)" -f $r, [math]::Round($sz/1GB,2))
    }
}

# ── 3. Disks: space, type, health ───────────────────────────────────────────
Add-Head '3. DISKS'
foreach ($v in (Get-Volume | Where-Object DriveLetter | Sort-Object DriveLetter)) {
    Add-Line ("{0}: type={1,-9} fs={2,-6} size={3,7} GB free={4,7} GB" -f `
        $v.DriveLetter, $v.DriveType, $v.FileSystemType, [math]::Round($v.Size/1GB,1), [math]::Round($v.SizeRemaining/1GB,1))
}
Add-Line ''
Add-Line 'physical disks:'
foreach ($d in (Get-PhysicalDisk -ErrorAction SilentlyContinue)) {
    Add-Line ("  disk {0}: {1} bus={2} media={3} health={4} size={5} GB" -f `
        $d.DeviceId, $d.FriendlyName, $d.BusType, $d.MediaType, $d.HealthStatus, [math]::Round($d.Size/1GB,1))
}
Add-Line ''
Add-Line 'partition -> disk mapping:'
foreach ($p in (Get-Partition -ErrorAction SilentlyContinue | Where-Object DriveLetter | Sort-Object DriveLetter)) {
    Add-Line ("  {0}: -> physical disk {1}" -f $p.DriveLetter, $p.DiskNumber)
}

# ── 4. Cache capacity decisions straight from Drive's own log ───────────────
Add-Head '4. WHAT DRIVE SAYS ABOUT ITS CACHE (its own log)'
$logDirs = @()
foreach ($r in ($roots | Select-Object -Unique)) {
    if (Test-Path (Join-Path $r 'Logs')) { $logDirs += (Join-Path $r 'Logs') }
}
if (-not $logDirs) {
    Add-Line 'no Logs folder found'
} else {
    foreach ($ld in $logDirs) {
        Add-Line "log dir: $ld"
        $logs = Get-ChildItem $ld -Filter 'drive_fs*.txt' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
        foreach ($lf in ($logs | Select-Object -First 3)) {
            Add-Line ("  -- {0}  ({1} MB, {2})" -f $lf.Name, [math]::Round($lf.Length/1MB,2), $lf.LastWriteTime)
            $lines = Get-Content $lf.FullName -Tail 6000 -ErrorAction SilentlyContinue
            $cap = $lines | Select-String -Pattern 'cache capacity|Content cache used|Free disk space|Evictable' -CaseSensitive:$false |
                   Select-Object -Last 6
            foreach ($c in $cap) { Add-Line ('     ' + $c.Line.Substring(0, [Math]::Min(170, $c.Line.Length))) }
        }
    }
}

# ── 5. Hard errors in Drive's log ──────────────────────────────────────────
Add-Head '5. ERRORS IN DRIVE LOG'
$pattern = 'FATAL|CHECK failed|crash|Cannot allocate|no space|disk full|ENOSPC|corrupt|IO error|Input/output|Access is denied|invalid argument|Errno 22|unmount|shutting down|OOM|out of memory'
$errFound = 0
foreach ($ld in $logDirs) {
    $logs = Get-ChildItem $ld -Filter 'drive_fs*.txt' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    foreach ($lf in ($logs | Select-Object -First 4)) {
        $hits = Get-Content $lf.FullName -ErrorAction SilentlyContinue |
                Select-String -Pattern $pattern -CaseSensitive:$false | Select-Object -Last 25
        if ($hits) {
            Add-Line ("from {0}:" -f $lf.Name)
            foreach ($h in $hits) { Add-Line ('  ' + $h.Line.Substring(0, [Math]::Min(200, $h.Line.Length))); $errFound++ }
        }
    }
}
if ($errFound -eq 0) { Add-Line 'no matching error lines found' }

# ── 6. Crash dumps: when did Drive actually die ─────────────────────────────
Add-Head '6. CRASH DUMPS'
$dumps = @()
foreach ($r in ($roots | Select-Object -Unique)) {
    $cp = Join-Path $r 'Crashpad\reports'
    if (Test-Path $cp) {
        $dumps += Get-ChildItem $cp -Filter '*.dmp' -ErrorAction SilentlyContinue
    }
}
if ($dumps) {
    Add-Line ("total dumps: {0}" -f $dumps.Count)
    foreach ($d in ($dumps | Sort-Object LastWriteTime -Descending | Select-Object -First 10)) {
        Add-Line ("  {0}  {1} KB  {2}" -f $d.Name, [math]::Round($d.Length/1KB), $d.LastWriteTime)
    }
} else {
    Add-Line 'no crash dumps found (good sign, or the cache was wiped)'
}

# ── 7. Windows event log around the crashes ────────────────────────────────
Add-Head '7. WINDOWS EVENT LOG (last 24h, app crashes and disk errors)'
$since = (Get-Date).AddHours(-24)
$app = Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$since; Level=1,2} -ErrorAction SilentlyContinue |
       Where-Object { $_.Message -match 'Drive|google|disk' } | Select-Object -First 12
if ($app) {
    foreach ($e in $app) {
        $msg = ($e.Message -replace '\s+', ' ')
        Add-Line ("  {0} [{1}] {2}" -f $e.TimeCreated, $e.ProviderName, $msg.Substring(0, [Math]::Min(180, $msg.Length)))
    }
} else { Add-Line '  no application errors mentioning Drive or disk' }
Add-Line ''
Add-Line 'disk/storage subsystem errors:'
$sys = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since; Level=1,2,3} -ErrorAction SilentlyContinue |
       Where-Object { $_.ProviderName -match 'disk|Ntfs|storahci|volsnap|volmgr|stornvme' } | Select-Object -First 15
if ($sys) {
    foreach ($e in $sys) {
        $msg = ($e.Message -replace '\s+', ' ')
        Add-Line ("  {0} [{1}] id={2} {3}" -f $e.TimeCreated, $e.ProviderName, $e.Id, $msg.Substring(0, [Math]::Min(160, $msg.Length)))
    }
} else { Add-Line '  none - disks are not reporting hardware errors' }

# ── 8. Antivirus: the classic DriveFS killer ───────────────────────────────
Add-Head '8. ANTIVIRUS AND EXCLUSIONS'
try {
    $av = Get-CimInstance -Namespace 'root\SecurityCenter2' -ClassName AntiVirusProduct -ErrorAction Stop
    foreach ($a in $av) { Add-Line ("  product: {0}" -f $a.displayName) }
} catch { Add-Line '  could not query SecurityCenter2' }
try {
    $pref = Get-MpPreference -ErrorAction Stop
    Add-Line ("  realtime monitoring disabled: {0}" -f $pref.DisableRealtimeMonitoring)
    $ex = @($pref.ExclusionPath)
    if ($ex -and $ex.Count -gt 0 -and $ex[0]) {
        Add-Line '  excluded paths:'
        foreach ($e in $ex) { Add-Line "    $e" }
    } else {
        Add-Line '  excluded paths: NONE'
        Add-Line '  >>> Defender scans every file Drive streams. This is a known cause'
        Add-Line '  >>> of DriveFS aborting large reads. Consider excluding the cache dir.'
    }
} catch { Add-Line '  Defender not available (third-party AV?)' }

# ── 9. Verdict ─────────────────────────────────────────────────────────────
Add-Head '9. SUMMARY'
$cFree = (Get-Volume -DriveLetter C -ErrorAction SilentlyContinue).SizeRemaining
if ($cFree) { Add-Line ("C: free: {0} GB" -f [math]::Round($cFree/1GB,1)) }
Add-Line ("dumps found: {0}   error lines: {1}" -f $dumps.Count, $errFound)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Add-Line ''
    Add-Line 'NOTE: not running as administrator, so antivirus exclusions could not be'
    Add-Line 'read. For the full picture, rerun from an admin PowerShell window.'
}
Add-Line ''
Add-Line "Full report written to: $report"

$out | Set-Content -LiteralPath $report -Encoding utf8
exit 0
