"""Run one real thumbnail prompt + Google Flow preflight before production."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

import config
from backend import gemini_image, thumbnail


def _latest_source_thumbnail() -> Path | None:
    root = Path(config.PROJECTS_DIR)
    candidates = list(root.glob("_prepare_*/thumbnail.jpg"))
    candidates.extend(root.glob("_prepare_*/**/thumbnail.jpg"))
    candidates = [path for path in candidates if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Reference thumbnail. Defaults to the newest prepared project thumbnail.")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--title", default="Geopolitical news update")
    args = parser.parse_args()

    settings = config.load_settings()
    if not settings.get("rewrite_thumbnail_enabled", True):
        raise SystemExit("Enable thumbnail prompt rewriting in FAA Settings first.")
    if not settings.get("gemini_image_enabled", False):
        raise SystemExit("Enable Google Flow thumbnail generation in FAA Settings first.")

    source = Path(args.image).expanduser().resolve() if args.image else _latest_source_thumbnail()
    if source is None or not source.is_file():
        raise SystemExit("No prepared thumbnail found. Prepare a source video first or pass --image PATH.")

    output_dir = Path(config.PROJECTS_DIR) / "_thumbnail_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "thumbnail_prompt.txt"
    image_path = output_dir / "thumbnail_generated.png"

    def emit(step: str, message: str) -> None:
        print(f"[{step}] {message}", flush=True)

    print(f"Reference: {source}", flush=True)
    result = thumbnail.analyze_and_rewrite(
        str(source),
        args.language,
        title=args.title,
        emit=emit,
    )
    prompt = (result.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit("Thumbnail API returned an empty generation prompt.")
    prompt_path.write_text(prompt, encoding="utf-8")

    image_path.unlink(missing_ok=True)
    gemini_image.generate_thumbnail(prompt, str(image_path), emit=emit)
    with Image.open(image_path) as generated:
        generated.load()
        if generated.size != (1920, 1080) or generated.format != "PNG":
            raise SystemExit(
                f"Invalid final thumbnail: format={generated.format}, size={generated.size}"
            )

    print("THUMBNAIL_PIPELINE_OK", flush=True)
    print(f"Prompt: {prompt_path}", flush=True)
    print(f"Image:  {image_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
