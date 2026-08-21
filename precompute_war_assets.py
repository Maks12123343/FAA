"""Prepare rewrite, metadata, and thumbnails without starting TTS or montage.

This is intentionally separate from the production pipeline. It uses the same
deterministic project IDs and existing cache files as ``war_pipeline.produce``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import config

from backend import gemini_image
from backend import languages as lang_utils
from backend import thumbnail as thumbnail_api
from backend.rewriter import rewrite_all
from backend.war_pipeline import _download_thumbnail


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached rewrite/metadata/thumbnails without VoiceGen, Whisper, or FFmpeg."
    )
    parser.add_argument("--prepare-id", required=True, help="Numeric ID or full ID, e.g. 1787078422")
    parser.add_argument("--niche", default="russia_ukraine_war")
    parser.add_argument("--languages", nargs="+", required=True, help="Language codes, e.g. cs tr pl")
    parser.add_argument(
        "--projects-dir",
        default="",
        help="Optional shared projects directory; otherwise the repo configuration is used.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show missing work; do not call APIs.")
    parser.add_argument(
        "--two-stage-compact",
        action="store_true",
        help="Deprecated no-op: Japanese/Korean always translate first before the rewrite.",
    )
    return parser.parse_args()


def _numeric_id(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        raise ValueError(f"Cannot find numeric ID in {value!r}")
    return digits


def _project_dir(projects_dir: str, niche: str, language: str, numeric_id: str) -> Path:
    return Path(projects_dir) / f"{niche}_{language}_{numeric_id}"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected an object in {path}")
    return data


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_text_if_missing(path: Path, value: str) -> None:
    if path.exists() or not value.strip():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def _ensure_reference(prepare_dir: Path, source_url: str) -> Path:
    reference = prepare_dir / "thumbnail.jpg"
    if reference.exists() and reference.stat().st_size > 2000:
        return reference
    match = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", source_url or "")
    if not match:
        raise RuntimeError(f"Cannot find YouTube video ID in {source_url!r}")
    if not _download_thumbnail(match.group(1), str(reference)):
        raise RuntimeError(f"Could not download reference thumbnail to {reference}")
    return reference


def _prepare_language(
    *,
    language: str,
    numeric_id: str,
    niche: str,
    projects_dir: str,
    prepare_dir: Path,
    state: dict,
    settings: dict,
    dry_run: bool,
) -> dict:
    project_dir = _project_dir(projects_dir, niche, language, numeric_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    script_path = project_dir / "script.txt"
    metadata_path = project_dir / "metadata.json"
    prompt_paths = [project_dir / "thumbnail_prompt.txt"]
    image_paths = [project_dir / "thumbnail_generated.png"]
    if bool(settings.get("gemini_image_double_preview", False)):
        prompt_paths.append(project_dir / "thumbnail_prompt_2.txt")
        image_paths.append(project_dir / "thumbnail_generated_2.png")

    missing_script = not _read_text(script_path)
    missing_metadata = not metadata_path.exists()
    missing_prompts = [not _read_text(path) for path in prompt_paths]
    missing_images = [not gemini_image.is_valid_thumbnail(str(path)) for path in image_paths]
    work = {
        "script": missing_script,
        "metadata": missing_metadata,
        "prompts": missing_prompts,
        "images": missing_images,
    }
    print(
        f"[{language}] {project_dir.name}: "
        f"script={'run' if missing_script else 'cached'}, "
        f"metadata={'run' if missing_metadata else 'cached'}, "
        f"prompts={sum(missing_prompts)}/{len(missing_prompts)}, "
        f"images={sum(missing_images)}/{len(missing_images)}",
        flush=True,
    )
    if dry_run:
        return work
    if not any((missing_script, missing_metadata, any(missing_prompts), any(missing_images))):
        return work

    result = None
    if missing_script or missing_metadata:
        print(f"[{language}] Preparing rewrite/metadata cache...", flush=True)
        result = rewrite_all(
            transcript=state["transcript"],
            language=language,
            source_title=state.get("source_title", ""),
            source_description=state.get("source_description", ""),
            source_tags=state.get("source_tags", []),
            cache_dir=str(project_dir),
        )
        if missing_script:
            _write_text_if_missing(script_path, result.get("script", ""))
        if missing_metadata:
            _save_json(
                metadata_path,
                {key: value for key, value in result.items() if key != "script"},
            )
        print(f"[{language}] Rewrite/metadata cache ready.", flush=True)

    if not any(missing_prompts) and not any(missing_images):
        return work

    if not bool(settings.get("rewrite_thumbnail_enabled", True)):
        raise RuntimeError("Thumbnail prompt rewrite is disabled in Settings")
    reference = _ensure_reference(prepare_dir, state.get("source_url", ""))
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    for index, prompt_path in enumerate(prompt_paths):
        prompt = _read_text(prompt_path)
        if not prompt:
            print(f"[{language}] Analyzing reference for thumbnail {index + 1}...", flush=True)
            analysis = thumbnail_api.analyze_and_rewrite(
                str(reference),
                language,
                title=metadata.get("title", state.get("source_title", "")),
            )
            prompt = (analysis.get("prompt") or "").strip()
            if not prompt:
                raise RuntimeError(f"Thumbnail prompt {index + 1} was empty")
            _write_text_if_missing(prompt_path, prompt)

        if not bool(settings.get("gemini_image_enabled", False)):
            print(f"[{language}] Flow generation disabled; prompt cache saved.", flush=True)
            continue
        if gemini_image.is_valid_thumbnail(str(image_paths[index])):
            print(f"[{language}] Thumbnail {index + 1} cached.", flush=True)
            continue
        print(f"[{language}] Generating thumbnail {index + 1} through Flow...", flush=True)
        gemini_image.generate_thumbnail(prompt, str(image_paths[index]))
        print(f"[{language}] Thumbnail {index + 1} saved at 1920x1080.", flush=True)
    return work


def main() -> int:
    args = _parse_args()
    if args.two_stage_compact:
        # Japanese/Korean already translate first by default, and the chunk count
        # must stay the one configured in Settings. Kept so existing commands and
        # scripts do not break.
        print(
            "[precompute] --two-stage-compact is now the default for Japanese/Korean; "
            "the chunk count comes from Settings.",
            flush=True,
        )
    numeric_id = _numeric_id(args.prepare_id)
    projects_dir = os.path.abspath(args.projects_dir or config.PROJECTS_DIR)
    prepare_dir = Path(projects_dir) / f"_prepare_war_{numeric_id}"
    state_path = prepare_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"Prepare state not found: {state_path}")
    state = _load_json(state_path)
    if not state.get("transcript"):
        raise SystemExit(f"Prepare state has no transcript: {state_path}")
    settings = config.load_settings()
    languages = []
    for raw in args.languages:
        code = str(raw).strip().lower()
        if code and code not in languages:
            languages.append(code)
    if not languages:
        raise SystemExit("No languages supplied")

    print(f"Projects: {projects_dir}", flush=True)
    print(f"Prepare: {prepare_dir.name} | source: {state.get('source_url', '')}", flush=True)
    print("Mode: cache-only rewrite/metadata/thumbnails; TTS, Whisper, clips, and montage are disabled.", flush=True)
    failures = []
    for language in languages:
        try:
            _prepare_language(
                language=language,
                numeric_id=numeric_id,
                niche=args.niche,
                projects_dir=projects_dir,
                prepare_dir=prepare_dir,
                state=state,
                settings=settings,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures.append((language, exc))
            print(f"[{language}] ERROR: {exc}", flush=True)

    if failures:
        print("\nFinished with failures:", flush=True)
        for language, error in failures:
            print(f"  {language}: {error}", flush=True)
        return 1
    print("\nDONE: rewrite, metadata, and thumbnail cache preparation finished.", flush=True)
    print("No VoiceGen, Whisper, clip selection, or montage was started.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Existing cache files were left untouched.", flush=True)
        raise SystemExit(130)
