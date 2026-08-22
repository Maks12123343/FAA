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

    # 1. Where the Drive cache actually lives, and how big it is
    default_base = os.path.expandvars(r"%LOCALAPPDATA%\Google\DriveFS")
    found = []
    for cand in [default_base] + [
        os.path.join(f"{chr(c)}:", os.sep, "DriveFS") for c in range(ord("A"), ord("Z") + 1)
    ]:
        if os.path.isdir(cand):
            total = 0
            for root, _dirs, files in os.walk(cand):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            found.append((cand, total))
    print(f"Drive cache (default): {default_base}")
    print(f"  exists            : {os.path.isdir(default_base)}")
    for path, size in found:
        where = "system disk" if path.upper().startswith("C:") else "moved off C:"
        print(f"  {path}  {_human(size)}  ({where})")
    if found and all(p.upper().startswith("C:") for p, _ in found):
        print("  NOTE: cache is on the system disk. Move it in Drive settings if C: is tight.")

    # 2. Free space on every fixed volume
    print("\nDisk space:")
    for c in range(ord("A"), ord("Z") + 1):
        drive = f"{chr(c)}:" + os.sep
        try:
            t, u, f = shutil.disk_usage(drive)
        except OSError:
            continue
        flags = []
        if f < 10e9:
            flags.append("LOW")
        if f > 60e9:
            flags.append("roomy - good cache target")
        tail = ("  <-- " + ", ".join(flags)) if flags else ""
        print(f"  {drive:5s} total={_human(t):>10s} free={_human(f):>10s}{tail}")

    # 3. The index file itself
    index_path = find_index()
    print(f"\nIndex file          : {index_path or 'NOT FOUND'}")
    if not index_path:
        print("Drive mount is offline right now. Restart Drive and rerun.")
        return 1
    size = os.path.getsize(index_path)
    print(f"  size              : {_human(size)}")

    # 4. Is it materialised locally, or will Drive have to download it?
    print("\nReading it the way the pipeline does (resumable, 256 KB blocks)...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.war_pipeline import _read_file_resumable

    t0 = time.time()
    try:
        data = _read_file_resumable(index_path)
        read = len(data)
    except OSError as exc:
        print(f"  FAILED: {exc}")
        print("\n  Drive would not hand over the whole file even with resuming.")
        print("  Restart Google Drive, then rerun. If it keeps failing, copy the")
        print("  index over from the other PC instead.")
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
