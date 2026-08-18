"""Generate isolated thumbnail-v2 candidates without changing FAA prompts."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
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


ANALYSIS_PROMPT = r"""
Analyze the PROVIDED REFERENCE IMAGE itself. The image is the only source of
truth. The video title is context only and must never override visible evidence.

Your purpose is to identify why this exact thumbnail is immediately clickable
at small size and which visual elements a new variation is forbidden to lose.

Inspect and record:
1. Exact camera viewpoint, crop, horizon, foreground/background relationship,
   and where land, water, sky, roads, docks, ships, buildings, and vehicles sit.
2. The dominant click hook: what it is, where it is, how large and bright it is,
   and why the eye notices it first.
3. The second and third click hooks and their visual weight.
4. Every aircraft or drone: use only the type visible in the image. Distinguish
   fixed-wing UAV, multirotor/FPV drone, helicopter, fighter, and airliner.
5. Every editor-added annotation such as a yellow circle, oval, arrow, outline,
   or glow. Record its color, target, position, thickness, and importance.
6. The color and contrast pattern that remains readable when the image is shown
   at 10 percent size.
7. Empty or low-information regions. State whether a new version may enlarge
   them. Preserve the reference's information density.
8. Three to five LOCKED CLICK HOOKS that every successful variation must retain.

Do not suggest improvements. Do not invent objects. Do not identify a precise
weapon or vehicle model unless visually certain. If uncertain, describe the
visible category accurately and state the uncertainty.

Return ONLY valid JSON with this schema:
{
  "scene_summary": "...",
  "camera_and_crop": "...",
  "dominant_event": {
    "description": "...",
    "position": "...",
    "prominence": "...",
    "color_contrast": "..."
  },
  "secondary_subjects": [
    {"description": "...", "position": "...", "visual_weight": "..."}
  ],
  "aircraft_or_drone": {
    "present": true,
    "visible_type": "...",
    "position": "...",
    "orientation": "...",
    "relative_size": "..."
  },
  "annotation": {
    "present": true,
    "type": "...",
    "color": "...",
    "target": "...",
    "position": "...",
    "importance": "..."
  },
  "information_density": "...",
  "locked_click_hooks": [
    {"description": "...", "position": "...", "must_preserve": true}
  ],
  "safe_micro_variations": ["...", "...", "..."]
}
""".strip()


REWRITE_SYSTEM = (
    "You are an expert prompt engineer for realistic, highly clickable YouTube "
    "news thumbnails. The supplied reference image and locked hooks are binding."
)


REWRITE_PROMPT = r"""
Create one complete standalone English text-to-image prompt for a NEW variation
of the supplied reference thumbnail.

REFERENCE IMAGE RULES:
- The reference image is the source of truth. Ignore any title implication that
  conflicts with visible pixels.
- This is a controlled variation of a proven thumbnail, not a redesign.
- Preserve the same story, camera side, crop density, visual hierarchy, subject
  relationships, dominant event scale, and immediately readable click hooks.
- Preserve EVERY item in locked_click_hooks. Do not replace, remove, shrink,
  hide, or move a locked hook into a less noticeable region.
- Preserve the exact visible aircraft/drone category. Never replace fixed-wing
  with FPV/multirotor, or the reverse.
- If the reference contains a circle, oval, arrow, outline, or other annotation,
  preserve its color, target, approximate position, and visual purpose.
- Do not increase empty sky, water, or low-information space beyond the reference.

CLICKABILITY RULES:
- At 10 percent display size the dominant event must remain instantly obvious.
- Keep strong local contrast and separation between the main hooks: detailed
  bright fire or impact where present, deep textured smoke where present, clear
  subject silhouettes, and a clean vivid annotation where present.
- Realistic overall color does not mean dull. Preserve the reference's strongest
  local orange, black, red, and yellow contrasts when those colors are visible.
- Keep the image dense, legible, dramatic, and credible as a news photograph.
  Avoid generic distant aftermath scenes, tiny subjects, washed-out fire, large
  empty areas, glossy CGI, fantasy destruction, and movie-poster lighting.

VARIATION RULES:
- Make only 2 to 4 safe micro-variations selected from the analysis, such as a
  subtle smoke drift, minor debris placement, small lighting/reflection changes,
  or a very slight camera adjustment that does not weaken the composition.
- Use the VARIANT CUE only for subtle diversity. It must not change locked hooks.
- No readable text, logos, flags, watermarks, timestamps, HUD, borders, or bars.
- Require a full-frame 1920x1080 16:9 image with natural geometry.

REFERENCE ANALYSIS JSON:
{analysis_json}

VIDEO TITLE (context only):
{source_title}

TARGET LANGUAGE (no text should appear in the image): {language}
VARIANT CUE: {variant_cue}

Return exactly these two sections and no wrapper or commentary:

### VARIANT PROMPT
A complete standalone English generation prompt.

### NEGATIVE PROMPT
A compact negative prompt adapted to the reference and analysis.
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
    fd, temp_name = tempfile.mkstemp(prefix="thumbnail_2.", suffix=".part", dir=output_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(png)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    if not gemini_image.is_valid_thumbnail(str(output_path)):
        raise RuntimeError("thumbnail_2.png failed 1920x1080 PNG validation")


def _project_id(prepare_id: str, language: str) -> str:
    digits = "".join(ch for ch in prepare_id if ch.isdigit())
    if not digits:
        raise RuntimeError(f"Prepare ID has no numeric suffix: {prepare_id}")
    return f"russia_ukraine_war_{language}_{digits}"


def generate_one(args: argparse.Namespace, state: dict[str, Any], image_path: Path, language: str) -> Path:
    project_id = _project_id(args.prepare_id, language)
    output_dir = _find_output_dir(args.downloads_root, project_id, language)
    output_path = output_dir / "thumbnail_2.png"
    analysis_path = output_dir / "thumbnail_2_analysis.json"
    prompt_path = output_dir / "thumbnail_2_prompt.txt"
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
                    "[thumbnail-v2] Analysis and prompt cached; skipping Byesu requests.",
                    flush=True,
                )
        except (OSError, ValueError):
            analysis = None

    if analysis is None:
        data_url = _image_data_url(image_path)
        analysis_messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }]
        analysis_error = None
        for format_attempt in range(2):
            messages = analysis_messages
            if format_attempt:
                messages = analysis_messages + [{
                    "role": "user",
                    "content": (
                        "Your previous answer did not satisfy the required schema. "
                        "Analyze the same supplied image again and return ONLY one valid "
                        "JSON object matching the exact schema."
                    ),
                }]
                print(
                    "[thumbnail-v2] Analysis response did not satisfy the JSON schema; retrying once...",
                    flush=True,
                )
            analysis_text = _call_provider(
                ANALYSIS_SYSTEM,
                messages,
                args.analysis_provider_id,
                args.analysis_model,
                "thumbnail_v2_analysis",
            )
            try:
                analysis = _complete_locked_hooks(_parse_json_response(analysis_text))
                break
            except RuntimeError as exc:
                analysis_error = exc
        if analysis is None:
            raise RuntimeError(
                f"{args.analysis_provider_id or 'primary'} thumbnail analysis failed JSON validation: "
                f"{analysis_error}"
            )

        rewrite_text = REWRITE_PROMPT.format(
            analysis_json=json.dumps(analysis, ensure_ascii=False, indent=2),
            source_title=state.get("source_title") or "(unknown)",
            language=language,
            variant_cue=VARIANT_CUES.get(language, "Keep the reference crop and vary only minor natural details."),
        )
        final_prompt = _call_provider(
            REWRITE_SYSTEM,
            [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": rewrite_text},
                ],
            }],
            args.rewrite_provider_id,
            args.rewrite_model,
            "thumbnail_v2_rewrite",
        )
        if "### VARIANT PROMPT" not in final_prompt:
            raise RuntimeError("Thumbnail rewrite did not return a VARIANT PROMPT section")

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
        description="Generate thumbnail_2.png with isolated test prompts; FAA production prompts are not modified."
    )
    parser.add_argument("--prepare-id", required=True, help="For example: war_1786969946")
    parser.add_argument("--languages", default="tr", help="Comma-separated language codes; default: tr")
    parser.add_argument(
        "--analysis-provider-id",
        default="byesu",
        help="Saved provider used only for reference-image analysis; default: byesu",
    )
    parser.add_argument(
        "--rewrite-provider-id",
        default="",
        help="Saved provider used only to rewrite the final image prompt; blank uses primary/fallback routing",
    )
    parser.add_argument(
        "--provider-id",
        default="",
        help="Deprecated alias for --rewrite-provider-id (kept for existing commands)",
    )
    parser.add_argument("--analysis-model", default="", help="Optional Byesu model override for visual analysis")
    parser.add_argument("--rewrite-model", default="", help="Optional model override for prompt rewriting")
    parser.add_argument(
        "--downloads-root",
        type=Path,
        default=ROOT.parent / "FAA_downloads",
        help="FAA_downloads root; defaults beside the FAA repository",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing thumbnail_2.png")
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
    image_path = prepare_dir / "thumbnail.jpg"
    if not state_path.is_file():
        raise RuntimeError(f"Prepare state not found: {state_path}")
    if not image_path.is_file():
        raise RuntimeError(f"Reference thumbnail not found: {image_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    languages = [item.strip().lower() for item in args.languages.split(",") if item.strip()]
    if not languages:
        raise RuntimeError("No languages selected")

    print(f"[thumbnail-v2] Reference: {image_path}", flush=True)
    print(f"[thumbnail-v2] Source: {state.get('source_url', '')}", flush=True)
    print(f"[thumbnail-v2] Languages: {', '.join(languages)}", flush=True)
    analysis_name, analysis_model = _provider_description(
        args.analysis_provider_id, args.analysis_model
    )
    rewrite_name, rewrite_model = _provider_description(
        args.rewrite_provider_id, args.rewrite_model
    )
    print(
        f"[thumbnail-v2] Analysis route: {analysis_name} ({analysis_model})",
        flush=True,
    )
    print(
        f"[thumbnail-v2] Prompt rewrite route: {rewrite_name} ({rewrite_model})",
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
