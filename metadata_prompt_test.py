import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from backend import channel_scanner
from backend import rewriter


DEFAULT_LANGUAGES = ["pl", "tr", "cs", "ro", "hu", "sv"]
DEFAULT_OUT_DIR = r"D:\youtube\запаска 1"


def _video_id_from_url(url: str) -> str:
    url = (url or "").strip()
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"(?:/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    raise ValueError("Could not extract YouTube video id from URL")


def _yt_dlp_fallback(url: str) -> dict:
    try:
        result = subprocess.run(
            ["yt-dlp", "--no-warnings", "--quiet", "--dump-json", url],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout or "{}")
        return {
            "url": url,
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "tags": data.get("tags", []) or [],
            "channel": data.get("uploader", "") or data.get("channel", ""),
        }
    except Exception as e:
        print(f"[metadata_test] yt-dlp fallback failed: {e}", flush=True)
        return {"url": url, "title": "", "description": "", "tags": []}


def _fetch_source_metadata(url: str) -> dict:
    meta = channel_scanner.get_video_metadata(url)
    if not (meta.get("title") or meta.get("description")):
        fallback = _yt_dlp_fallback(url)
        if fallback.get("title") or fallback.get("description"):
            meta = fallback
    return meta


def _bundle_text(source: dict, result: dict, language: str) -> str:
    lines = []
    lines.append(f"Source URL: {source.get('url', '')}")
    lines.append(f"Source title: {source.get('title', '')}")
    lines.append(f"Target language: {language}")
    lines.append("")
    lines.append("### Optimized Titles:")
    for idx, title in enumerate(result.get("titles", []), start=1):
        lines.append(f"{idx}. {title}")
    lines.append("")
    lines.append("### Optimized Description:")
    lines.append(result.get("description", "").strip())
    lines.append("")
    lines.append("### Optimized Tags:")
    lines.append(result.get("tags_raw", "") or ", ".join(result.get("tags", [])))
    lines.append("")
    lines.append(f"Tags chars: {len(result.get('tags_raw', '') or ', '.join(result.get('tags', [])))}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Fetch YouTube metadata by URL, rewrite it, and save one bundle per language."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--language", default="same as source", help="Single language to rewrite into")
    parser.add_argument(
        "--languages",
        default="all",
        help="Comma-separated target languages, or 'all' for pl,tr,cs,ro,hu,sv.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    args = parser.parse_args()

    video_id = _video_id_from_url(args.url)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{video_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metadata_test] video_id={video_id}")
    print(f"[metadata_test] output={out_dir}")

    print("[metadata_test] fetching source metadata...")
    meta_started = time.monotonic()
    source = _fetch_source_metadata(args.url)
    print(f"[metadata_test] metadata fetched in {time.monotonic() - meta_started:.1f}s")
    if not source.get("title") and not source.get("description"):
        raise RuntimeError("Could not fetch source metadata from URL")

    with (out_dir / "source_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(source, f, ensure_ascii=False, indent=2)

    if args.languages.strip().lower() == "all":
        languages = list(DEFAULT_LANGUAGES)
    else:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if not languages:
        languages = [args.language]
    print(f"[metadata_test] languages parsed: {', '.join(languages)}")

    all_results = []
    for i, language in enumerate(languages, start=1):
        print(f"[metadata_test] rewriting {i}/{len(languages)}: {language}...")
        step_started = time.monotonic()
        result = rewriter._rewrite_metadata(
            language=language,
            source_title=source.get("title", ""),
            source_description=source.get("description", ""),
            source_tags=source.get("tags", []) or [],
        )
        elapsed = time.monotonic() - step_started
        print(f"[metadata_test] {language} took {elapsed:.1f}s")

        bundle_text = _bundle_text(source, result, language)
        bundle_path = out_dir / f"metadata_{i:02d}_{language}.txt"
        bundle_path.write_text(bundle_text, encoding="utf-8")
        print(f"[metadata_test] saved: {bundle_path}")

        result_item = {
            "language": language,
            "source": source,
            "result": result,
            "bundle_file": str(bundle_path),
            "tags_chars": len(result.get("tags_raw", "")),
        }
        all_results.append(result_item)

    summary_path = out_dir / "result.json"
    summary_path.write_text(json.dumps(
        {
            "url": args.url,
            "video_id": video_id,
            "languages": languages,
            "source": source,
            "results": all_results,
        },
        ensure_ascii=False,
        indent=2,
    ), encoding="utf-8")

    print(f"[metadata_test] total took {time.monotonic() - started_at:.1f}s")
    print(f"[metadata_test] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
