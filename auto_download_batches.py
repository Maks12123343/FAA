import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_OUT_DIR = r"D:\youtube"
DEFAULT_HOST = "91.150.160.38"
DEFAULT_PORT = "11655"
DEFAULT_USER = "root"
DEFAULT_REMOTE_PROJECTS = "/workspace/FAA/projects"
DEFAULT_REMOTE_LOG = "/tmp/faa.log"
DEFAULT_NICHE = "russia_ukraine_war"
DEFAULT_LANGUAGES = ["pl", "tr", "cs", "ro", "hu", "sv"]

LANGUAGE_FOLDERS = {
    "pl": "польський канал",
    "tr": "турецький канал",
    "cs": "чеський канал",
    "ro": "румунський канал",
    "hu": "угорський канал",
    "sv": "шведський канал",
    "sw": "суахілі канал",
    "de": "німецький канал",
    "fr": "французький канал",
    "es": "іспанський канал",
    "it": "італійський канал",
    "pt": "португальський канал",
    "en": "англійський канал",
}


REMOTE_SCAN_SCRIPT = r"""
import glob
import json
import os
import re

projects_dir = __PROJECTS_DIR__
niche = __NICHE__
languages = set(__LANGUAGES__)
pattern = re.compile(r"^" + re.escape(niche) + r"_(?P<lang>[a-z]{2,5})_(?P<ts>\d+)$")
items = []

for project_dir in glob.glob(os.path.join(projects_dir, niche + "_*")):
    project_id = os.path.basename(project_dir)
    m = pattern.match(project_id)
    if not m:
        continue
    lang = m.group("lang")
    if languages and lang not in languages:
        continue
    mp4 = os.path.join(project_dir, project_id + ".mp4")
    meta = os.path.join(project_dir, "metadata.json")
    script = os.path.join(project_dir, "script.txt")
    if not (os.path.exists(mp4) and os.path.exists(meta) and os.path.exists(script)):
        continue
    try:
        size = os.path.getsize(mp4)
        if size < 1024 * 1024:
            continue
        mtime = max(os.path.getmtime(project_dir), os.path.getmtime(mp4), os.path.getmtime(meta))
    except OSError:
        continue

    title = ""
    try:
        with open(meta, encoding="utf-8") as f:
            title = str(json.load(f).get("title", ""))
    except Exception:
        pass

    items.append({
        "project_id": project_id,
        "project_dir": project_dir,
        "language": lang,
        "timestamp": int(m.group("ts")),
        "mtime": float(mtime),
        "mp4": mp4,
        "metadata": meta,
        "size": int(size),
        "title": title,
    })

items.sort(key=lambda x: (x["mtime"], x["timestamp"]), reverse=True)
print("FAA_JSON_START")
print(json.dumps(items, ensure_ascii=False))
print("FAA_JSON_END")
"""


REMOTE_PROJECT_INFO_SCRIPT = r"""
import json
import os
import re

mp4 = __MP4_PATH__
niche = __NICHE__
languages = set(__LANGUAGES__)
project_dir = os.path.dirname(mp4)
project_id = os.path.basename(project_dir)
expected_mp4 = os.path.join(project_dir, project_id + ".mp4")
meta = os.path.join(project_dir, "metadata.json")
script = os.path.join(project_dir, "script.txt")
pattern = re.compile(r"^" + re.escape(niche) + r"_(?P<lang>[a-z]{2,5})_(?P<ts>\d+)$")
m = pattern.match(project_id)

item = None
if m and os.path.realpath(mp4) == os.path.realpath(expected_mp4):
    lang = m.group("lang")
    if (not languages or lang in languages) and os.path.exists(mp4) and os.path.exists(meta) and os.path.exists(script):
        try:
            size = os.path.getsize(mp4)
            mtime = max(os.path.getmtime(project_dir), os.path.getmtime(mp4), os.path.getmtime(meta))
            if size >= 1024 * 1024:
                title = ""
                try:
                    with open(meta, encoding="utf-8") as f:
                        title = str(json.load(f).get("title", ""))
                except Exception:
                    pass
                item = {
                    "project_id": project_id,
                    "project_dir": project_dir,
                    "language": lang,
                    "timestamp": int(m.group("ts")),
                    "mtime": float(mtime),
                    "mp4": mp4,
                    "metadata": meta,
                    "size": int(size),
                    "title": title,
                }
        except OSError:
            pass

print("FAA_JSON_START")
print(json.dumps(item, ensure_ascii=False))
print("FAA_JSON_END")
"""


def _run(cmd, *, input_text=None, timeout=None):
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )


def _ssh_base(args):
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(args.port),
        f"{args.user}@{args.host}",
    ]


def _scp_base(args):
    return [
        "scp",
        "-P",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]


def _extract_json(stdout: str) -> list:
    m = re.search(r"FAA_JSON_START\s*(.*?)\s*FAA_JSON_END", stdout or "", re.DOTALL)
    if not m:
        raise RuntimeError("Server scan did not return JSON markers")
    return json.loads(m.group(1))


def scan_ready_projects(args) -> list:
    script = (
        REMOTE_SCAN_SCRIPT
        .replace("__PROJECTS_DIR__", repr(args.remote_projects))
        .replace("__NICHE__", repr(args.niche))
        .replace("__LANGUAGES__", repr(args.languages))
    )
    cmd = _ssh_base(args) + ["python3 -"]
    result = _run(cmd, input_text=script, timeout=60)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ssh scan failed").strip())
    return _extract_json(result.stdout)


def fetch_remote_log(args) -> str:
    cmd = _ssh_base(args) + [f"tail -n {int(args.log_tail)} {args.remote_log} 2>/dev/null || true"]
    result = _run(cmd, timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "remote log read failed").strip())
    return result.stdout or ""


def ready_paths_from_log(text: str) -> list:
    paths = []
    seen = set()
    patterns = [
        r"Video ready:\s+(/workspace/FAA/projects/\S+?\.mp4)",
        r"Final polish done in [^:]+:\s+(/workspace/FAA/projects/\S+?\.mp4)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            path = match.group(1).strip().rstrip(".,;")
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def project_info_from_mp4(args, mp4_path: str) -> dict | None:
    script = (
        REMOTE_PROJECT_INFO_SCRIPT
        .replace("__MP4_PATH__", repr(mp4_path))
        .replace("__NICHE__", repr(args.niche))
        .replace("__LANGUAGES__", repr(args.languages))
    )
    cmd = _ssh_base(args) + ["python3 -"]
    result = _run(cmd, input_text=script, timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "project info failed").strip())
    return _extract_json(result.stdout)


def ready_projects_from_log(args) -> list:
    items = []
    seen = set()
    for mp4_path in ready_paths_from_log(fetch_remote_log(args)):
        item = project_info_from_mp4(args, mp4_path)
        if not item:
            continue
        pid = item["project_id"]
        if pid in seen:
            continue
        seen.add(pid)
        items.append(item)
    items.sort(key=lambda x: (x["mtime"], x["timestamp"]))
    return items


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"downloaded": {}, "ignored": {}, "active_batch": None}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("downloaded", {})
        data.setdefault("ignored", {})
        data.setdefault("active_batch", None)
        return data
    except Exception:
        backup = path.with_suffix(path.suffix + ".broken")
        path.replace(backup)
        return {"downloaded": {}, "ignored": {}, "active_batch": None}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def latest_per_language(items: list, languages: list, downloaded: dict, ignored: dict) -> list:
    selected = []
    seen = set()
    for item in items:
        lang = item["language"]
        if lang not in languages or lang in seen:
            continue
        if item["project_id"] in downloaded or item["project_id"] in ignored:
            seen.add(lang)
            continue
        selected.append(item)
        seen.add(lang)
    return selected


def all_new_ready(items: list, languages: list, downloaded: dict, ignored: dict) -> list:
    out = []
    for item in sorted(items, key=lambda x: (x["mtime"], x["timestamp"])):
        if (
            item["language"] in languages
            and item["project_id"] not in downloaded
            and item["project_id"] not in ignored
        ):
            out.append(item)
    return out


def batch_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_active_batch(args, state: dict) -> dict:
    active = state.get("active_batch")
    if active and active.get("folder"):
        return active
    name = batch_name()
    folder = str(Path(args.out_dir) / name)
    active = {
        "name": name,
        "folder": folder,
        "created_at": time.time(),
        "languages": {},
    }
    state["active_batch"] = active
    return active


def render_metadata(meta: dict, remote_meta_path: str, project_id: str) -> str:
    titles = meta.get("titles") or []
    description = meta.get("description") or ""
    tags_raw = meta.get("tags_raw") or ", ".join(meta.get("tags") or [])

    lines = [
        f"Project: {project_id}",
        f"Source metadata: {remote_meta_path}",
    ]
    if meta.get("title"):
        lines.append(f"Main title: {meta['title']}")
    lines.extend(["", "### Optimized Titles:"])
    if titles:
        for idx, title in enumerate(titles, start=1):
            lines.append(f"{idx}. {title}")
    else:
        lines.append("(no titles found)")
    lines.extend(["", "### Optimized Description:", description.strip()])
    lines.extend(["", "### Optimized Tags:", tags_raw.strip()])
    lines.extend(["", f"Tags chars: {len(tags_raw.strip())}"])
    return "\n".join(lines).rstrip() + "\n"


def scp_download(args, remote_path: str, local_path: Path, retries: int) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    part = local_path.with_name(local_path.name + ".part")
    if part.exists():
        part.unlink()

    remote = f"{args.user}@{args.host}:{remote_path}"
    last_error = ""
    for attempt in range(1, retries + 1):
        cmd = _scp_base(args) + [remote, str(part)]
        result = _run(cmd, timeout=args.scp_timeout)
        if result.returncode == 0 and part.exists() and part.stat().st_size > 0:
            part.replace(local_path)
            return
        last_error = (result.stderr or result.stdout or "scp failed").strip()
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass
        time.sleep(min(20, attempt * 5))
    raise RuntimeError(last_error or f"Failed to download {remote_path}")


def download_project(args, state: dict, item: dict) -> bool:
    active = ensure_active_batch(args, state)
    lang = item["language"]
    folder_name = LANGUAGE_FOLDERS.get(lang, f"{lang} канал")
    dest_dir = Path(active["folder"]) / folder_name
    dest_video = dest_dir / "video.mp4"
    dest_meta_json = dest_dir / "metadata.json.tmp"
    dest_meta_txt = dest_dir / "metadata.txt"

    print(f"[download] {item['project_id']} -> {dest_dir}")
    scp_download(args, item["mp4"], dest_video, args.retries)
    scp_download(args, item["metadata"], dest_meta_json, args.retries)

    with dest_meta_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    dest_meta_txt.write_text(
        render_metadata(meta, item["metadata"], item["project_id"]),
        encoding="utf-8",
    )
    try:
        dest_meta_json.unlink()
    except OSError:
        pass

    state["downloaded"][item["project_id"]] = {
        "language": lang,
        "folder": str(dest_dir),
        "downloaded_at": time.time(),
        "size": item["size"],
        "title": item.get("title", ""),
    }
    active.setdefault("languages", {})[lang] = item["project_id"]
    save_state(Path(args.state_file), state)
    print(f"[done] {lang}: video.mp4 + metadata.txt")
    return True


def maybe_close_batch(args, state: dict) -> None:
    active = state.get("active_batch")
    if not active:
        return
    have = set((active.get("languages") or {}).keys())
    need = set(args.languages)
    if need and need.issubset(have):
        print(f"[batch] complete: {active['folder']}")
        state["active_batch"] = None
        save_state(Path(args.state_file), state)


def run_once(args, state: dict, initial: bool) -> int:
    if args.mode == "log":
        queue = all_new_ready(
            ready_projects_from_log(args),
            args.languages,
            state["downloaded"],
            state.get("ignored", {}),
        )
        if args.dry_run:
            print("[dry-run] ready projects from log:")
            for item in queue:
                print(f"[dry-run] {item['language']}: {item['project_id']} ({item['size']} bytes)")
            return 0
        if not queue:
            print("[log] no new Video ready lines")
            return 0
        print("[log] ready to download: " + ", ".join(x["project_id"] for x in queue))
        count = 0
        for item in queue:
            try:
                if download_project(args, state, item):
                    count += 1
            except Exception as e:
                print(f"[error] {item['project_id']}: {e}", file=sys.stderr)
        maybe_close_batch(args, state)
        return count

    items = scan_ready_projects(args)
    if initial and args.latest_per_language:
        queue = latest_per_language(
            items,
            args.languages,
            state["downloaded"],
            state.get("ignored", {}),
        )
        if args.dry_run:
            print("[dry-run] latest ready per language:")
            for item in queue:
                print(f"[dry-run] {item['language']}: {item['project_id']} ({item['size']} bytes)")
            return 0
        selected_ids = {x["project_id"] for x in queue}
        for item in items:
            if item["project_id"] in state["downloaded"] or item["project_id"] in selected_ids:
                continue
            state.setdefault("ignored", {})[item["project_id"]] = {
                "language": item["language"],
                "ignored_at": time.time(),
                "reason": "older than first latest-per-language scan",
            }
        save_state(Path(args.state_file), state)
    else:
        queue = all_new_ready(items, args.languages, state["downloaded"], state.get("ignored", {}))
        if args.dry_run:
            print("[dry-run] new ready projects:")
            for item in queue:
                print(f"[dry-run] {item['language']}: {item['project_id']} ({item['size']} bytes)")
            return 0

    if not queue:
        print("[scan] no new ready projects")
        return 0

    print("[scan] ready to download: " + ", ".join(x["project_id"] for x in queue))
    count = 0
    for item in queue:
        try:
            if download_project(args, state, item):
                count += 1
        except Exception as e:
            print(f"[error] {item['project_id']}: {e}", file=sys.stderr)
    maybe_close_batch(args, state)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically download finished FAA videos and metadata from the Vast.ai server."
    )
    parser.add_argument("--out", "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote-projects", default=DEFAULT_REMOTE_PROJECTS)
    parser.add_argument("--remote-log", default=DEFAULT_REMOTE_LOG)
    parser.add_argument("--niche", default=DEFAULT_NICHE)
    parser.add_argument(
        "--mode",
        choices=["log", "scan"],
        default="log",
        help="log = download exact projects from /tmp/faa.log Video ready lines; scan = scan newest ready folders.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated languages to download. Default: pl,tr,cs,ro,hu,sv",
    )
    parser.add_argument("--poll", type=int, default=180, help="Seconds between scans in watch mode.")
    parser.add_argument("--log-tail", type=int, default=1200, help="Remote log lines to inspect in log mode.")
    parser.add_argument("--once", action="store_true", help="Scan once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded, without downloading.")
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Ignore projects that already exist on the first scan and download only future finished projects.",
    )
    parser.add_argument(
        "--latest-per-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On first scan, download only the newest ready project for each language.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--scp-timeout", type=int, default=7200)
    args = parser.parse_args()

    args.languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    args.state_file = str(out_dir / ".faa_downloaded.json")
    return args


def mark_existing_ignored(args, state: dict) -> None:
    if state.get("new_only_initialized"):
        return
    if args.mode == "log":
        items = ready_projects_from_log(args)
    else:
        items = scan_ready_projects(args)
    if args.dry_run:
        print(f"[dry-run][new-only] would ignore existing ready projects: {len(items)}")
        return
    now = time.time()
    ignored = state.setdefault("ignored", {})
    count = 0
    for item in items:
        pid = item["project_id"]
        if pid in state.get("downloaded", {}) or pid in ignored:
            continue
        ignored[pid] = {
            "language": item["language"],
            "ignored_at": now,
            "reason": "existing project ignored by --new-only",
        }
        count += 1
    state["new_only_initialized"] = True
    save_state(Path(args.state_file), state)
    print(f"[new-only] ignored existing ready projects: {count}")


def main() -> int:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    state = load_state(Path(args.state_file))
    if args.new_only:
        mark_existing_ignored(args, state)

    initial = True
    while True:
        try:
            run_once(args, state, initial=initial)
        except Exception as e:
            print(f"[scan-error] {e}", file=sys.stderr)
        initial = False
        if args.once:
            return 0
        time.sleep(max(30, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
