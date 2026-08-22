"""Google Drive File Stream health check for the FAA library index.

Explains and works around the crash where reading the 156 MB index.json makes
the Drive mount disappear (OSError Errno 22, then FileNotFoundError).

Run on the machine that has the Drive mount:
    python diag_drive_health.py
"""

import os
import shutil
import sys
import time

CANDIDATE_MOUNTS = ["G:", "E:"]
NICHE = "russia_ukraine_war"


def _human(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def find_index():
    for mount in CANDIDATE_MOUNTS:
        for middle in (
            os.path.join(mount, os.sep, "My Drive", "workspace", "gdrive", "movies"),
            os.path.join(mount, os.sep, "Мій диск", "workspace", "gdrive", "movies"),
        ):
            p = os.path.join(middle, NICHE, "index.json")
            if os.path.exists(p):
                return p
    return ""


def main():
    print("=" * 70)
    print("GOOGLE DRIVE HEALTH CHECK")
    print("=" * 70)

    # 1. Drive cache location and size cap
    base = os.path.expandvars(r"%LOCALAPPDATA%\Google\DriveFS")
    print(f"DriveFS folder      : {base}")
    print(f"  exists            : {os.path.isdir(base)}")
    if os.path.isdir(base):
        total = 0
        for root, _dirs, files in os.walk(base):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        print(f"  cache size on disk: {_human(total)}")

    # 2. Free space on every relevant volume
    print("\nDisk space:")
    for drive in ["C:" + os.sep] + [m + os.sep for m in CANDIDATE_MOUNTS]:
        try:
            t, u, f = shutil.disk_usage(drive)
            flag = "  <-- LOW" if f < 5e9 else ""
            print(f"  {drive:6s} total={_human(t):>10s} free={_human(f):>10s}{flag}")
        except OSError:
            print(f"  {drive:6s} not available")

    # 3. The index file itself
    index_path = find_index()
    print(f"\nIndex file          : {index_path or 'NOT FOUND'}")
    if not index_path:
        print("Drive mount is offline right now. Restart Drive and rerun.")
        return 1
    size = os.path.getsize(index_path)
    print(f"  size              : {_human(size)}")

    # 4. Is it materialised locally, or will Drive have to download it?
    print("\nReading it in 1 MB blocks (this is what the pipeline does now)...")
    t0 = time.time()
    read = 0
    try:
        with open(index_path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                read += len(chunk)
    except OSError as exc:
        print(f"  FAILED at {_human(read)}: {exc!r}")
        print("\n  The mount dropped mid-read. This is the bug you keep hitting.")
        print("  Fix: keep a local copy of the index (see the command below).")
        return 1
    elapsed = time.time() - t0
    speed = read / max(elapsed, 0.01) / 1e6
    print(f"  OK {_human(read)} in {elapsed:.1f}s ({speed:.0f} MB/s)")
    if speed < 25:
        print("  Slow: Drive is streaming this from the network, not from cache.")
        print("  That download is exactly when the mount tends to crash.")
    else:
        print("  Fast: the file is already cached locally on this machine.")

    # 5. Local copy status
    local_dir = os.path.join(os.path.expanduser("~"), ".faa_index_cache", NICHE)
    local_path = os.path.join(local_dir, "index.json")
    print(f"\nLocal copy          : {local_path}")
    if os.path.exists(local_path):
        lsize = os.path.getsize(local_path)
        same = lsize == size
        print(f"  exists, size {_human(lsize)}, matches Drive: {same}")
        if not same:
            print("  Size differs - refresh it with --copy")
    else:
        print("  missing - create it with:  python diag_drive_health.py --copy")

    if "--copy" in sys.argv:
        print(f"\nCopying index to {local_path} ...")
        os.makedirs(local_dir, exist_ok=True)
        tmp = local_path + ".part"
        t0 = time.time()
        with open(index_path, "rb") as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        os.replace(tmp, local_path)
        print(f"  done in {time.time()-t0:.1f}s -> {_human(os.path.getsize(local_path))}")
        print("  The pipeline will now use this copy and stop touching Drive for it.")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
