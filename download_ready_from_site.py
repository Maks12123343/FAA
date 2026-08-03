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
DEFAULT_LANGUAGES = "pl,tr,cs,ro,hu,sv,fi,hr,da,bg"
DEFAULT_INTERVAL_MINUTES = 30

LANGUAGE_FOLDERS = {
    "pl": "польська мова",
    "tr": "турецька мова",
    "cs": "чеська мова",
    "ro": "румунська мова",
    "hu": "угорська мова",
    "sv": "шведська мова",
    "fi": "фінська мова",
    "hr": "хорватська мова",
    "da": "данська мова",
    "bg": "болгарська мова",
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


def _prune_empty_dirs(leaf: Path, stop_at: Path) -> None:
    """
    Прибрати теки, що лишились порожніми після невдалого завантаження.
    _download_file створює теку до початку качання, тому після помилки
    залишається порожня "<день>/<мова>". Йдемо знизу вгору й видаляємо
    лише порожні, ніколи не піднімаючись вище out_dir.
    """
    try:
        leaf = leaf.resolve()
        stop = stop_at.resolve()
    except OSError:
        return
    while leaf != stop and stop in leaf.parents:
        try:
            leaf.rmdir()          # OSError, якщо тека не порожня — тоді стоп
        except OSError:
            return
        leaf = leaf.parent


def _dest_paths(dest_dir: Path) -> tuple:
    """
    Шляхи для video / metadata / project у теці мови.

    Тека мови тепер спільна на весь день, тому друге відео тієї самої мови
    за той самий день не має перезаписати перше — додаємо суфікс _2, _3, ...
    """
    if not (dest_dir / "video.mp4").exists():
        return (
            dest_dir / "video.mp4",
            dest_dir / "metadata.txt",
            dest_dir / "project.json",
        )
    n = 2
    while (dest_dir / f"video_{n}.mp4").exists():
        n += 1
    return (
        dest_dir / f"video_{n}.mp4",
        dest_dir / f"metadata_{n}.txt",
        dest_dir / f"project_{n}.json",
    )


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
        except KeyboardInterrupt:
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            raise
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
        "latest_per_language": "0" if args.project_ids else ("1" if args.latest_per_language else "0"),
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
    requested_ids = set(args.project_ids)
    out = []
    for item in projects:
        pid = item.get("project_id")
        if not pid:
            continue
        if requested_ids and pid not in requested_ids:
            continue
        if not args.force and pid in state.get("downloaded", {}):
            continue
        out.append(item)
    return out


def _state_file(args) -> Path:
    return Path(args.out_dir) / ".faa_site_downloaded.json"


def _mark_existing_ready(args) -> int:
    state_file = _state_file(args)
    state = _load_state(state_file)
    data = _http_json(_ready_url(args), args.timeout)
    projects = data.get("projects") or []
    marked = 0
    for item in projects:
        pid = item.get("project_id")
        if not pid or pid in state.get("downloaded", {}):
            continue
        state.setdefault("downloaded", {})[pid] = {
            "language": item.get("language", ""),
            "language_name": item.get("language_name", ""),
            "folder": "",
            "downloaded_at": time.time(),
            "ignored_existing": True,
            "title": item.get("title", ""),
            "video_size": item.get("video_size", 0),
        }
        marked += 1
    if marked:
        _save_state(state_file, state)
    return marked


def _run_once(args) -> int:
    out_dir = Path(args.out_dir)
    state_file = _state_file(args)
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

    # Одна тека на ДЕНЬ, а не на кожну перевірку.
    # Раніше тут стояв "%Y-%m-%d_%H-%M-%S", через що мови з одного джерела
    # розлітались по десятках тек: сервер робить мови по черзі, а --watch
    # опитує сайт кожні N хвилин, тож кожна мова потрапляла у власну теку.
    # mkdir тут навмисно немає — теку створює _download_file. Якщо завантаження
    # впало, порожню теку прибирає _prune_empty_dirs у except-гілці нижче.
    batch_dir = out_dir / datetime.now().strftime("%Y-%m-%d")

    downloaded = 0
    failed = 0
    for item in projects:
        pid = item["project_id"]
        dest_dir = None
        try:
            item = _project_metadata(args, pid)
            lang_folder = _language_folder(item)
            dest_dir = batch_dir / lang_folder
            dest_video, dest_meta, dest_info = _dest_paths(dest_dir)

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
            downloaded += 1
            print(f"[done] {lang_folder}")
        except KeyboardInterrupt:
            if dest_dir is not None:
                _prune_empty_dirs(dest_dir, out_dir)
            raise
        except Exception as e:
            failed += 1
            print(f"[error] {pid}: {e}", file=sys.stderr)
            if dest_dir is not None:
                _prune_empty_dirs(dest_dir, out_dir)

    if downloaded:
        print(f"Batch folder: {batch_dir}")
    print(f"Downloaded: {downloaded}, failed: {failed}")
    return 0


def _watch(args) -> int:
    interval = max(60, int(args.interval_minutes * 60))
    print(
        f"Watching {args.base_url.rstrip('/')} every {args.interval_minutes:g} minutes. "
        "Press Ctrl+C or close this terminal to stop."
    )
    baseline_done = not args.watch_new_only
    while True:
        print(f"[check] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            if baseline_done:
                downloaded = _run_once(args)
                if downloaded > 0:
                    print("[watch] downloaded new video(s), checking again immediately")
                    continue
            else:
                marked = _mark_existing_ready(args)
                print(f"[watch] ignored {marked} ready project(s) already visible at startup")
                baseline_done = True
        except KeyboardInterrupt:
            print("Stopped.")
            return 0
        except urllib.error.URLError as e:
            print(f"Connection error: {e}", file=sys.stderr)
            print("Check that the site is running and the SSH tunnel is open.", file=sys.stderr)
        except Exception as e:
            print(f"Download check failed: {e}", file=sys.stderr)

        print(f"[sleep] next check in {interval // 60} min")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Stopped.")
            return 0


def run(args) -> int:
    if args.watch:
        return _watch(args)
    _run_once(args)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ready FAA videos and metadata through the local website tunnel."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument(
        "--project-ids",
        default="",
        help="Comma-separated exact project IDs to download; overrides latest-per-language selection.",
    )
    parser.add_argument(
        "--latest-per-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download only the newest ready project for each language.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Download even if state says it was already downloaded.")
    parser.add_argument("--watch", action="store_true", help="Keep checking for new ready videos.")
    parser.add_argument(
        "--watch-new-only",
        action="store_true",
        help="With --watch, skip projects that are already ready when the script starts.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=DEFAULT_INTERVAL_MINUTES,
        help="How often --watch checks the site.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--download-timeout", type=int, default=7200)
    args = parser.parse_args()
    args.languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    args.project_ids = [x.strip() for x in args.project_ids.split(",") if x.strip()]
    if args.watch and args.force:
        parser.error("--force cannot be used with --watch because it would download the same videos repeatedly.")
    if args.project_ids and args.watch:
        parser.error("--project-ids is for a one-time exact batch and cannot be combined with --watch.")
    if args.watch_new_only and not args.watch:
        parser.error("--watch-new-only requires --watch.")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Stopped.")
        raise SystemExit(130)
    except urllib.error.URLError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        print("Check that the site is running and the SSH tunnel is open.", file=sys.stderr)
        raise SystemExit(1)
