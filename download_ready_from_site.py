import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_BASE_URL = "http://localhost:5050"
DEFAULT_OUT_DIR = r"D:\youtube"
DEFAULT_LANGUAGES = "pl,tr,cs,ro,hu,sv"

LANGUAGE_FOLDERS = {
    "pl": "польська мова",
    "tr": "турецька мова",
    "cs": "чеська мова",
    "ro": "румунська мова",
    "hu": "угорська мова",
    "sv": "шведська мова",
    "de": "німецька мова",
    "fr": "французька мова",
    "es": "іспанська мова",
    "it": "італійська мова",
    "pt": "португальська мова",
    "uk": "українська мова",
    "ru": "російська мова",
    "en": "англійська мова",
}


def _http_json(url: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_name(value: str, fallback: str = "unknown") -> str:
    value = (value or "").strip() or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:80] or fallback


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"downloaded": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("downloaded", {})
        return data
    except Exception:
        broken = path.with_suffix(path.suffix + ".broken")
        path.replace(broken)
        return {"downloaded": {}}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _metadata_text(item: dict) -> str:
    titles = item.get("all_titles") or []
    tags_raw = item.get("tags_raw") or ", ".join(item.get("tags") or [])
    lines = [
        f"Project ID: {item.get('project_id', '')}",
        f"Language: {item.get('language_name') or item.get('language') or ''}",
        f"Created: {item.get('created_at', '')}",
        "",
        "### Main Title:",
        str(item.get("title") or "").strip(),
        "",
        "### All Title Options:",
    ]
    if titles:
        for idx, title in enumerate(titles, start=1):
            lines.append(f"{idx}. {title}")
    else:
        lines.append("(no title options found)")

    lines.extend([
        "",
        "### Description:",
        str(item.get("description") or "").strip(),
        "",
        "### Tags:",
        tags_raw.strip(),
        "",
        f"Tags chars: {len(tags_raw.strip())}",
    ])

    thumbnail_prompt = str(item.get("thumbnail_prompt") or "").strip()
    if thumbnail_prompt:
        lines.extend(["", "### Thumbnail Prompt:", thumbnail_prompt])

    return "\n".join(lines).rstrip() + "\n"


def _language_folder(item: dict) -> str:
    code = str(item.get("language") or "").strip().lower()
    folder = LANGUAGE_FOLDERS.get(code)
    if folder:
        return _safe_name(folder)
    return _safe_name(item.get("language_name") or code or "unknown")


def _download_file(url: str, dest: Path, retries: int, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "video/mp4,*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                next_report = 50 * 1024 * 1024
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        if total:
                            print(f"    {done // 1024 // 1024}MB / {total // 1024 // 1024}MB")
                        else:
                            print(f"    {done // 1024 // 1024}MB")
                        next_report += 50 * 1024 * 1024
            if part.exists() and part.stat().st_size > 0:
                part.replace(dest)
                return
            last_error = "empty download"
        except Exception as e:
            last_error = str(e)
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            if attempt < retries:
                wait = min(20, attempt * 5)
                print(f"  retry {attempt}/{retries} in {wait}s: {last_error}")
                time.sleep(wait)
    raise RuntimeError(last_error or f"download failed: {url}")


def _ready_url(args) -> str:
    params = {
        "languages": ",".join(args.languages),
        "latest_per_language": "1" if args.latest_per_language else "0",
    }
    return args.base_url.rstrip("/") + "/api/projects/ready?" + urllib.parse.urlencode(params)


def _project_metadata(args, project_id: str) -> dict:
    quoted = urllib.parse.quote(project_id, safe="")
    url = args.base_url.rstrip("/") + f"/api/projects/{quoted}/metadata"
    return _http_json(url, args.timeout)


def _video_url(args, project_id: str) -> str:
    quoted = urllib.parse.quote(project_id, safe="")
    return args.base_url.rstrip("/") + f"/api/download/{quoted}"


def _select_projects(args, state: dict) -> list:
    data = _http_json(_ready_url(args), args.timeout)
    projects = data.get("projects") or []
    out = []
    for item in projects:
        pid = item.get("project_id")
        if not pid:
            continue
        if not args.force and pid in state.get("downloaded", {}):
            continue
        out.append(item)
    return out


def run(args) -> int:
    out_dir = Path(args.out_dir)
    state_file = out_dir / ".faa_site_downloaded.json"
    state = _load_state(state_file)

    projects = _select_projects(args, state)
    if not projects:
        print("No new ready projects.")
        return 0

    print("Ready projects:")
    for item in projects:
        lang = item.get("language_name") or item.get("language") or "unknown"
        size_mb = int((item.get("video_size") or 0) / 1024 / 1024)
        print(f"  {lang}: {item.get('project_id')} ({size_mb}MB)")

    if args.dry_run:
        print("Dry run only. Nothing downloaded.")
        return 0

    batch_dir = out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir.mkdir(parents=True, exist_ok=True)

    for item in projects:
        pid = item["project_id"]
        item = _project_metadata(args, pid)
        lang_folder = _language_folder(item)
        dest_dir = batch_dir / lang_folder
        dest_video = dest_dir / "video.mp4"
        dest_meta = dest_dir / "metadata.txt"
        dest_info = dest_dir / "project.json"

        print(f"[download] {pid} -> {dest_dir}")
        _download_file(_video_url(args, pid), dest_video, args.retries, args.download_timeout)
        dest_meta.write_text(_metadata_text(item), encoding="utf-8")
        dest_info.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

        state.setdefault("downloaded", {})[pid] = {
            "language": item.get("language", ""),
            "language_name": item.get("language_name", ""),
            "folder": str(dest_dir),
            "downloaded_at": time.time(),
            "title": item.get("title", ""),
            "video_size": item.get("video_size", 0),
        }
        _save_state(state_file, state)
        print(f"[done] {lang_folder}")

    print(f"Batch folder: {batch_dir}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ready FAA videos and metadata through the local website tunnel."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument(
        "--latest-per-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download only the newest ready project for each language.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Download even if state says it was already downloaded.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--download-timeout", type=int, default=7200)
    args = parser.parse_args()
    args.languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except urllib.error.URLError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        print("Check that the site is running and the SSH tunnel is open.", file=sys.stderr)
        raise SystemExit(1)
