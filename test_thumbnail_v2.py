"""Generate isolated thumbnail-v2 candidates without changing FAA prompts."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile
from typing import Any

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backend import api_client, gemini_image  # noqa: E402


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
}


ANALYSIS_SYSTEM = (
    "You are a strict visual analyst for high-performing YouTube news thumbnails. "
    "Inspect the supplied image pixels. Never infer visual facts from the title."
)


REWRITE_SYSTEM = (
    "You are an expert prompt engineer for realistic, highly clickable YouTube "
    "news thumbnails. Inspect the supplied reference image carefully before "
    "writing the final prompt. The reference geometry is binding."
)


ONE_SHOT_PROMPT = r"""
You are reverse-engineering the PROVIDED REFERENCE IMAGE for a high-performing
YouTube breaking-news thumbnail. The image is attached in this same message and
is the only visual source of truth. Inspect its actual pixels carefully before
writing anything. The video title is context only and must never override what
is visible.

Your task is to produce a complete, self-contained English prompt for Google
Flow. Flow will receive only your written prompt and will NOT receive the
reference image, so every important visual fact must be written explicitly.

The result must be a new thumbnail variation that is at least as strong,
dramatic, readable, realistic, and clickable as the supplied competitor
thumbnail. Preserve the competitor's visual logic and dominant click hooks,
but do not copy the image pixel-for-pixel or reproduce it one-to-one.

NON-NEGOTIABLE REFERENCE LOCKS:
- Preserve the exact visible type of location and environment. Do not replace
  an industrial waterfront with a steppe, or a road convoy with a city, unless
  that is what the image visibly shows.
- Preserve the camera viewpoint, height, angle, perspective, crop, horizon,
  information density, and direction of the scene.
- Describe the left, center, right, foreground, middle ground, and background
  explicitly, including the relative size and position of every dominant
  subject.
- Preserve the main click hook's position, footprint, scale, shape, colors,
  brightness, and relationship to nearby objects.
- Treat the location's visual fingerprint as locked: preserve the same
  distribution of major tanks, roads, pipe corridors, quay or shoreline,
  water boundary, vessels, buildings, and large empty areas. Describe these
  landmarks individually instead of summarizing them as a generic facility.
- Preserve the same destruction footprint: which tanks, roofs, platforms,
  vehicles, decks, roads, and structures are burning, damaged, collapsed,
  blackened, or still intact. Do not replace visible structural damage with a
  clean intact facility around a fireball.
- Do not make the main event smaller, weaker, farther away, cleaner, or less
  colorful than in the supplied image.
- Do not move the drone, aircraft, ship, vehicle, building, explosion, smoke,
  circle, oval, arrow, or other annotation unless the change is genuinely tiny
  and affects only a secondary detail.
- Do not invent a different country, military platform, landscape, weather,
  architecture, or story. If a detail is unclear, describe only what is visibly
  supported by the pixels.

IF A LARGE EXPLOSION OR FIRE IS VISIBLE, describe it in concrete detail:
- exact position in the frame and relation to the horizon and nearby objects;
- its apparent size and footprint compared with the frame and nearby vehicles;
- the broad fireball silhouette and number of visible flame lobes;
- bright yellow-white core, orange and red-orange flames, darker cavities,
  sparks, fragments, heat distortion, and ground or structure connection;
- the full black/charcoal smoke volume, density, rolling shape, direction, and
  how it overlaps the background;
- the visible damage below and around the fire: destroyed or burning tanks,
  collapsed platforms, blackened ground, damaged roofs, broken pipework,
  secondary fires, debris, or intact structures that provide scale. Preserve
  the same amount and severity of destruction shown in the reference;
- the bright fireball must remain a dominant visual hook, not a small fire,
  distant flash, firecracker, or thin flame.

IF A DRONE OR AIRCRAFT IS VISIBLE, describe its visible category, approximate
size, exact frame position, silhouette, and relationship to the main event.
Keep it visibly pitched and tilted downward toward the explosion or impact
area, as if it is descending in an attack run: its nose/front and flight axis
must point toward the blast, with a natural slight bank and clear downward
attitude. It must not look level, stationary, flying away, or unrelated to the
strike. Preserve the aircraft category, proportions, position, scale, and
recognizable silhouette while making this attack trajectory unmistakable. If a
yellow circle or oval is visible, preserve exactly one similar highlight around
the same target, in the same general position, with similar thickness, color,
and visibility. Do not add labels, arrows, captions, logos, or text that are
absent from the image.

Write the final prompt with the following detailed sections:
1. Overall visual concept and breaking-news impact.
2. Camera, perspective, crop, horizon, and spatial layout.
3. Exact location and environment-defining elements.
4. Main subject and dominant click hook.
5. Explosion/fire/smoke details, if present.
6. Vehicles, structures, aircraft, drone, water, roads, or other visible
   secondary subjects.
7. Lighting, color palette, atmosphere, and contrast.
8. Realistic compressed news-photograph quality.
9. Minimal controlled variation.
10. Negative prompt.

The final image must be full-frame 1920x1080, 16:9, without borders or black
bars. It must look like a real compressed breaking-news photograph or video
still, not CGI, digital art, a poster, a game screenshot, or a fantasy scene.
Keep the dominant event and main subjects clear at thumbnail size.

The variation section may change only minor smoke curls, tiny debris positions,
small reflections, subtle haze, or other non-essential details. It must not
change the location, camera composition, dominant event, explosion footprint,
drone position, annotation target, object scale, destruction footprint, or
visual hierarchy. The generated scene must remain recognizably the same type
of industrial site with the same water/land relationship and the same major
landmarks, only slightly reinterpreted.

The negative prompt must explicitly prevent weak/small explosions, wrong
locations, changed camera angles, distant panoramas, altered object positions,
missing annotations, reduced destruction, intact replacement structures,
incorrect aircraft types, level or unrelated aircraft attitude, aircraft
flying away from the explosion, CGI, cartoon rendering, random text, logos,
watermarks, and obvious AI artifacts.

VIDEO TITLE (context only):
{source_title}

TARGET LANGUAGE (no text should appear in the image): {language}
VARIANT CUE (secondary details only): {variant_cue}

Return exactly these three sections and nothing else:

### VISUAL AUDIT
Write a concrete factual audit of the actual image: location type, camera and
crop, all dominant subjects, their positions and relative sizes, main hook,
explosion/fire/smoke details if present, exact fireball structure, visible
damage footprint, destroyed versus intact structures, aircraft/drone position
and descending attack trajectory toward the blast, annotation details,
location-defining landmarks, colors, lighting, and safe micro-variations. Do
not invent facts or summarize the scene too generally.

### VARIANT PROMPT
Write one long, complete, standalone English production prompt for Google Flow.
It must describe the actual reference-specific scene in enough detail to
reconstruct its composition without seeing the image. Do not write “use the
reference image” or refer to an unavailable image.

### NEGATIVE PROMPT
Write a strong compact negative prompt protecting the exact location type,
composition, dominant event, scale, colors, aircraft/drone, annotations,
photorealism, destruction footprint, and thumbnail readability.
""".strip()


VARIANT_CUES = {
    "tr": "Keep the reference crop; use a slightly different natural smoke curl and tiny debris changes.",
    "fi": "Keep the reference crop; use subtly cooler haze and a minor reflection change on the water.",
    "hu": "Keep the reference crop; shift only small smoke wisps and secondary debris.",
    "da": "Keep the reference crop; use a tiny natural camera-height change without adding empty space.",
    "bg": "Keep the reference crop; vary only cloud softness and small industrial reflections.",
    "cs": "Keep the reference crop; vary minor debris positions and heat distortion.",
    "ro": "Keep the reference crop; use a subtle daylight and smoke-drift variation.",
}


def _clean_model_text(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^```(?:json|text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^:::writing\{[^\n]*\}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*:::\s*$", "", text)
    return text.strip()


def _parse_json_response(value: str) -> dict[str, Any]:
    text = _clean_model_text(value)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        preview = re.sub(r"\s+", " ", text)[:300]
        raise RuntimeError(
            "Thumbnail analysis did not return JSON"
            + (f"; response preview: {preview!r}" if preview else "; response was empty")
        )
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Thumbnail analysis returned invalid JSON: {exc}") from exc
    hooks = result.get("locked_click_hooks")
    if not isinstance(hooks, list):
        hooks = []
    result["locked_click_hooks"] = [
        hook for hook in hooks
        if isinstance(hook, dict) and str(hook.get("description") or "").strip()
    ]
    return result


def _complete_locked_hooks(analysis: dict[str, Any]) -> dict[str, Any]:
    """Fill a short hook list from facts GPT already returned in the same JSON."""
    hooks = list(analysis.get("locked_click_hooks") or [])
    seen = {str(item.get("description", "")).strip().lower() for item in hooks}

    candidates: list[dict[str, Any]] = []
    dominant = analysis.get("dominant_event")
    if isinstance(dominant, dict) and dominant.get("description"):
        candidates.append({
            "description": str(dominant["description"]),
            "position": str(dominant.get("position") or "as described in the reference"),
            "must_preserve": True,
        })
    for subject in analysis.get("secondary_subjects") or []:
        if isinstance(subject, dict) and subject.get("description"):
            candidates.append({
                "description": str(subject["description"]),
                "position": str(subject.get("position") or "as described in the reference"),
                "must_preserve": True,
            })
    aircraft = analysis.get("aircraft_or_drone")
    if isinstance(aircraft, dict) and aircraft.get("present") and aircraft.get("visible_type"):
        candidates.append({
            "description": f"Visible aircraft or drone: {aircraft['visible_type']}",
            "position": str(aircraft.get("position") or "as described in the reference"),
            "must_preserve": True,
        })
    annotation = analysis.get("annotation")
    if isinstance(annotation, dict) and annotation.get("present") and annotation.get("target"):
        candidates.append({
            "description": f"Editor annotation targeting {annotation['target']}",
            "position": str(annotation.get("position") or "as described in the reference"),
            "must_preserve": True,
        })
    if analysis.get("camera_and_crop"):
        candidates.append({
            "description": f"Reference camera and crop: {analysis['camera_and_crop']}",
            "position": "full-frame composition",
            "must_preserve": True,
        })
    if analysis.get("information_density"):
        candidates.append({
            "description": f"Reference information density: {analysis['information_density']}",
            "position": "throughout the frame",
            "must_preserve": True,
        })
    if analysis.get("scene_summary"):
        candidates.append({
            "description": f"Reference scene: {analysis['scene_summary']}",
            "position": "as shown in the reference",
            "must_preserve": True,
        })

    for candidate in candidates:
        key = candidate["description"].strip().lower()
        if key and key not in seen:
            hooks.append(candidate)
            seen.add(key)
        if len(hooks) >= 3:
            break
    analysis["locked_click_hooks"] = hooks[:5]
    if not analysis["locked_click_hooks"]:
        raise RuntimeError("Thumbnail analysis returned no usable visual facts")
    return analysis


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _download_thumbnail_with_ytdlp(source_url: str, destination: Path) -> Path:
    """Download only the source video's thumbnail through yt-dlp, never the video."""
    with tempfile.TemporaryDirectory(prefix="faa_thumbnail_source_") as temp_dir:
        temp_root = Path(temp_dir) / "source"
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--skip-download",
            "--write-thumbnail",
            "--convert-thumbnails",
            "jpg",
            "--no-write-info-json",
            "--output",
            str(temp_root) + ".%(ext)s",
            source_url,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1200:]
            raise RuntimeError(f"yt-dlp could not download the source thumbnail: {detail}")

        candidates = sorted(
            item for item in Path(temp_dir).glob("source.*")
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not candidates:
            raise RuntimeError("yt-dlp completed but did not save a thumbnail image")

        source_path = candidates[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_path = destination.with_suffix(source_path.suffix.lower())
        shutil.copyfile(source_path, final_path)
        return final_path


def _call_provider(
    system: str,
    messages: list,
    provider_id: str,
    model_override: str,
    label: str,
) -> str:
    if not provider_id:
        text, _ = api_client.call_rewrite_api(
            system,
            messages,
            timeout=180,
            max_retries=2,
            step_label=label,
        )
        return _clean_model_text(text)

    settings = config.load_settings()
    name, url, key, model, effort, max_tokens = api_client._provider_settings(
        settings, provider_id, allow_legacy=False
    )
    selected_model = model_override or model
    use_responses_for_images = (
        provider_id == "custom" and selected_model.startswith("chatgpt-web/")
    )
    text, _ = api_client._call_openai_compatible(
        provider_name=name,
        api_url=url,
        api_key=key,
        model=selected_model,
        system=system,
        messages=messages,
        timeout=180,
        max_retries=2,
        step_label=label,
        reasoning_effort=effort,
        max_tokens_raw=max_tokens,
        use_responses_for_images=use_responses_for_images,
    )
    return _clean_model_text(text)


def _provider_description(provider_id: str, model_override: str) -> tuple[str, str]:
    """Return the configured provider/model without exposing credentials."""
    if not provider_id:
        settings = config.load_settings()
        provider_id = str(settings.get("rewrite_active_provider") or "a6api")
    settings = config.load_settings()
    name, _url, _key, model, _effort, _max_tokens = api_client._provider_settings(
        settings, provider_id, allow_legacy=False
    )
    return name, model_override or model


def _find_output_dir(downloads_root: Path, project_id: str, language: str) -> Path:
    folder_name = LANGUAGE_FOLDERS.get(language, language)
    if downloads_root.is_dir():
        for date_dir in sorted(downloads_root.iterdir(), reverse=True):
            candidate = date_dir / folder_name
            project_file = candidate / "project.json"
            if not project_file.is_file():
                continue
            try:
                data = json.loads(project_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("project_id") == project_id:
                return candidate
    return Path(config.PROJECTS_DIR) / project_id


def _bridge_settings() -> dict[str, Any]:
    settings = config.load_settings()
    env_values = dotenv_values(ROOT / "gemini_bridge" / ".env")
    return {
        "url": str(settings.get("gemini_image_bridge_url") or "http://127.0.0.1:4981").rstrip("/"),
        "api_key": str(settings.get("gemini_image_api_key") or env_values.get("LOCAL_API_KEY") or "").strip(),
        "model": str(settings.get("gemini_image_model") or env_values.get("FLOW_MODEL") or "flow-nano-pro").strip(),
        "timeout": max(600, min(1800, int(settings.get("gemini_image_timeout") or 600))),
    }


def _generate_flow(prompt: str, output_path: Path) -> None:
    cfg = _bridge_settings()
    if not cfg["api_key"]:
        raise RuntimeError("LOCAL_API_KEY is missing from gemini_bridge/.env")
    try:
        health = requests.get(f"{cfg['url']}/health", timeout=15)
        health.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Flow bridge is not available at {cfg['url']}: {exc}") from exc

    response = requests.post(
        f"{cfg['url']}/v1/images/generations",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json={
            "model": cfg["model"],
            "prompt": prompt,
            "size": "1920x1080",
            "n": 1,
            "response_format": "b64_json",
        },
        timeout=cfg["timeout"],
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Flow bridge HTTP {response.status_code}: {(response.text or '')[:800]}")
    try:
        encoded = response.json()["data"][0]["b64_json"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Flow bridge returned an unexpected response") from exc

    png = gemini_image._normalize_png(gemini_image._decode_image(encoded))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{output_path.stem}.", suffix=".part", dir=output_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(png)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    if not gemini_image.is_valid_thumbnail(str(output_path)):
        raise RuntimeError(f"{output_path.name} failed 1920x1080 PNG validation")


def _project_id(prepare_id: str, language: str) -> str:
    digits = "".join(ch for ch in prepare_id if ch.isdigit())
    if not digits:
        raise RuntimeError(f"Prepare ID has no numeric suffix: {prepare_id}")
    return f"russia_ukraine_war_{language}_{digits}"


def _extract_section(text: str, heading: str, next_heading: str | None = None) -> str:
    cleaned = _clean_model_text(text)
    start = cleaned.find(heading)
    if start < 0:
        raise RuntimeError(f"One-shot response is missing {heading}")
    start += len(heading)
    end = len(cleaned)
    if next_heading:
        next_pos = cleaned.find(next_heading, start)
        if next_pos >= 0:
            end = next_pos
    value = cleaned[start:end].strip()
    if not value:
        raise RuntimeError(f"One-shot response has an empty {heading} section")
    return value


def _extract_one_shot_response(text: str) -> tuple[str, str]:
    audit = _extract_section(text, "### VISUAL AUDIT", "### VARIANT PROMPT")
    variant = _extract_section(text, "### VARIANT PROMPT", "### NEGATIVE PROMPT")
    negative = _extract_section(text, "### NEGATIVE PROMPT")
    prompt = f"### VARIANT PROMPT\n{variant}\n\n### NEGATIVE PROMPT\n{negative}"
    return audit, prompt


def generate_one(args: argparse.Namespace, state: dict[str, Any], image_path: Path, language: str) -> Path:
    project_id = _project_id(args.prepare_id, language)
    output_dir = _find_output_dir(args.downloads_root, project_id, language)
    stem = args.output_stem
    output_path = output_dir / f"{stem}.png"
    analysis_path = output_dir / f"{stem}_analysis.json"
    prompt_path = output_dir / f"{stem}_prompt.txt"
    if output_path.exists() and not args.force:
        raise RuntimeError(f"Output already exists (use --force to replace it): {output_path}")

    analysis = None
    final_prompt = ""
    if not args.refresh_analysis and analysis_path.is_file() and prompt_path.is_file():
        try:
            cached_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            cached_prompt = prompt_path.read_text(encoding="utf-8")
            if isinstance(cached_analysis, dict) and "### VARIANT PROMPT" in cached_prompt:
                analysis = cached_analysis
                final_prompt = cached_prompt
                print(
                    f"[thumbnail-v2] {stem} analysis and prompt cached; skipping provider requests.",
                    flush=True,
                )
        except (OSError, ValueError):
            analysis = None

    if analysis is None:
        data_url = _image_data_url(image_path)
        one_shot_text = ONE_SHOT_PROMPT.format(
            source_title=state.get("source_title") or "(unknown)",
            language=language,
            variant_cue=VARIANT_CUES.get(
                language,
                "Keep the reference geometry fixed and vary only minor natural secondary details.",
            ),
        )
        one_shot_messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": one_shot_text},
            ],
        }]
        response_error = None
        audit = ""
        raw_response = ""
        for format_attempt in range(2):
            if format_attempt:
                one_shot_messages = one_shot_messages + [{
                    "role": "user",
                    "content": (
                        "Your previous response did not follow the required three-section format. "
                        "Inspect the same image again and return only VISUAL AUDIT, VARIANT PROMPT, "
                        "and NEGATIVE PROMPT sections. Do not omit any section."
                    ),
                }]
                print(
                    "[thumbnail-v2] One-shot response format invalid; retrying once...",
                    flush=True,
                )
            raw_response = _call_provider(
                REWRITE_SYSTEM,
                one_shot_messages,
                args.analysis_provider_id,
                args.analysis_model or args.rewrite_model,
                "thumbnail_v2_one_shot_prompt",
            )
            try:
                audit, final_prompt = _extract_one_shot_response(raw_response)
                analysis = {
                    "mode": "one_shot_visual_audit_and_prompt",
                    "provider_id": args.analysis_provider_id,
                    "model": args.analysis_model or args.rewrite_model,
                    "reference_image": str(image_path),
                    "visual_audit": audit,
                }
                break
            except RuntimeError as exc:
                response_error = exc
        if analysis is None:
            raise RuntimeError(
                f"{args.analysis_provider_id or 'primary'} one-shot thumbnail prompt failed validation: "
                f"{response_error}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prompt_path.write_text(final_prompt, encoding="utf-8")
    _generate_flow(final_prompt, output_path)
    print(f"[thumbnail-v2] SAVED: {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an isolated thumbnail candidate; FAA production prompts are not modified."
    )
    parser.add_argument("--prepare-id", required=True, help="For example: war_1786969946")
    parser.add_argument("--languages", default="tr", help="Comma-separated language codes; default: tr")
    parser.add_argument(
        "--source-url",
        default="",
        help="YouTube URL; if omitted, the script asks for it interactively",
    )
    parser.add_argument(
        "--analysis-provider-id",
        default="byesu",
        help="Saved provider used for the one-shot visual audit and prompt; default: byesu",
    )
    parser.add_argument(
        "--rewrite-provider-id",
        default="",
        help="Legacy alias; one-shot mode uses --analysis-provider-id",
    )
    parser.add_argument(
        "--provider-id",
        default="",
        help="Deprecated alias for --rewrite-provider-id (kept for existing commands)",
    )
    parser.add_argument("--analysis-model", default="", help="Optional model override for the one-shot provider")
    parser.add_argument("--rewrite-model", default="", help="Legacy model alias used if --analysis-model is blank")
    parser.add_argument(
        "--downloads-root",
        type=Path,
        default=ROOT.parent / "FAA_downloads",
        help="FAA_downloads root; defaults beside the FAA repository",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="Optional image used as the visual reference instead of the prepare thumbnail",
    )
    parser.add_argument(
        "--output-stem",
        default="thumbnail_3",
        help="Output filename stem; default: thumbnail_3",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing candidate with the same stem")
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="Ignore cached analysis/prompt and call the text provider again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.provider_id:
        if args.rewrite_provider_id and args.rewrite_provider_id != args.provider_id:
            raise RuntimeError("Use either --provider-id or --rewrite-provider-id, not conflicting values")
        args.rewrite_provider_id = args.provider_id
    prepare_dir = Path(config.PROJECTS_DIR) / f"_prepare_{args.prepare_id}"
    state_path = prepare_dir / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Prepare state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_url = args.source_url.strip()
    if args.reference_image:
        if source_url:
            raise RuntimeError("Use either --source-url or --reference-image, not both")
        image_path = args.reference_image
        if not image_path.is_file():
            raise RuntimeError(f"Reference thumbnail not found: {image_path}")
        source_url = "(manual reference image)"
    else:
        if not source_url:
            source_url = input("Paste the YouTube source-video URL: ").strip()
        if not source_url:
            raise RuntimeError("A YouTube source URL is required")
        source_path = prepare_dir / f"{args.output_stem}_source"
        print("[thumbnail-v2] Downloading source thumbnail with yt-dlp (video download disabled)...", flush=True)
        image_path = _download_thumbnail_with_ytdlp(source_url, source_path)
        print(f"[thumbnail-v2] Source thumbnail saved: {image_path}", flush=True)

    # The downloaded image, not the old prepare state, is the only visual source of truth.
    state["source_title"] = "(ignore title; use only the supplied reference pixels)"

    languages = [item.strip().lower() for item in args.languages.split(",") if item.strip()]
    if not languages:
        raise RuntimeError("No languages selected")

    print(f"[thumbnail-v2] Reference: {image_path}", flush=True)
    print(f"[thumbnail-v2] Source video: {source_url}", flush=True)
    print(f"[thumbnail-v2] Languages: {', '.join(languages)}", flush=True)
    prompt_name, prompt_model = _provider_description(
        args.analysis_provider_id, args.analysis_model or args.rewrite_model
    )
    print(
        f"[thumbnail-v2] One-shot analysis + prompt route: {prompt_name} ({prompt_model})",
        flush=True,
    )
    print(
        "[thumbnail-v2] Image generation route: Google Flow bridge",
        flush=True,
    )
    for language in languages:
        generate_one(args, state, image_path, language)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[thumbnail-v2] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
