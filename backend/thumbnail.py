"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os
import time

from backend import api_client
from backend import languages as lang_utils

THUMBNAIL_ATTEMPTS = 3

ANALYSIS_PROMPT = """
Analyze the provided competitor YouTube thumbnail as a visual reference only.
Do not verify facts and do not claim that the scene is real.

Describe the visible thumbnail strategy in compact detail:
1. Main subject/event and what must be the focal point.
2. Camera angle, perspective, crop tightness, and camera distance.
3. Main-hook geometry: approximate position and size as frame percentages.
4. Explosion/fire/smoke scale, placement, shape, density, debris direction, and visual intensity.
5. Drone/aircraft/weapon details if visible: type, shape, size, angle, position, highlight circle/arrow, and flight direction relative to the blast or target.
6. Military base or location details: buildings, hangars, roads, fences, vehicles, soldiers, damage, dust, and debris.
7. Broad environment and regional feel: climate, terrain, vegetation, architecture, road/ground type.
8. Lighting, weather, color palette, realism level, and clickability.
9. Visible text, arrows, circles, labels, outlines, or other thumbnail effects.
10. What must stay similar.
11. What can change to avoid direct copying.

Be strict about scale. If the main hook is large and close, say that clearly.
Do not describe it as a distant wide landscape or survey shot.
Be strict about realism. If it looks like a real news/photo/drone still, say
that clearly and warn against CGI, movie-poster, game-render, or glossy AI style.
""".strip()

REWRITE_SYSTEM = (
    "You write production-ready image-generation prompts for YouTube thumbnails. "
    "You preserve the competitor thumbnail strategy while changing concrete details. "
    "Always write the final image-generation prompt in English."
)


def _emit(emit, msg: str):
    if emit:
        emit("thumbnail", msg)


def _image_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _call_thumbnail_step(system: str, messages: list, label: str, emit=None) -> str:
    last_err = None
    for attempt in range(1, THUMBNAIL_ATTEMPTS + 1):
        try:
            _emit(emit, f"{label} attempt {attempt}/{THUMBNAIL_ATTEMPTS}...")
            text, _ = api_client.call_pioneer(
                system,
                messages,
                timeout=90,
                max_retries=2,
                emit=emit,
                step_label="thumbnail",
                use_rewrite_model=False,
            )
            text = (text or "").strip()
            if text:
                return text
            last_err = "empty response"
            _emit(emit, f"{label} returned empty response")
        except Exception as e:
            last_err = str(e)
            _emit(emit, f"{label} failed attempt {attempt}/{THUMBNAIL_ATTEMPTS}: {e}")
            print(
                f"[thumbnail] {label} failed attempt {attempt}/{THUMBNAIL_ATTEMPTS}: {e}",
                flush=True,
            )
        if attempt < THUMBNAIL_ATTEMPTS:
            time.sleep(5 * attempt)
    raise RuntimeError(f"{label} failed after {THUMBNAIL_ATTEMPTS} attempts: {last_err}")


def analyze_and_rewrite(image_path: str, language: str, title: str = "", emit=None) -> dict:
    """Analyze a competitor thumbnail and return a prompt for a new one.

    Returns {"prompt": str, "analysis": str}. Any API failure is raised to the
    caller; the pipeline catches it and continues without a thumbnail prompt.
    """
    if not image_path or not os.path.exists(image_path):
        return {"prompt": "", "analysis": ""}

    _emit(emit, "Analyzing source thumbnail...")
    data_url = _image_data_url(image_path)

    analysis = _call_thumbnail_step(
        "You are a precise YouTube thumbnail design analyst.",
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
        "Thumbnail analysis",
        emit=emit,
    )

    _emit(emit, "Writing thumbnail generation prompt...")
    language_name = lang_utils.configured_language_name(language)
    rewrite_prompt = f"""
Use the competitor thumbnail analysis as a visual reference only. Do not verify facts and do not make factual claims.

Create ONE English image-generation prompt for a new YouTube thumbnail.

Video title:
{title or '(unknown)'}

Target language for any visible thumbnail text:
{language_name}

Write the final image-generation prompt in English only. The target language applies only to visible thumbnail text, and only if visible text is truly needed.

Competitor analysis:
{analysis}

Universal rules:
- Preserve the reference thumbnail's core visual strategy, not the exact image.
- Preserve the same main visual hook, camera angle, perspective, approximate camera distance, hook position, explosion/fireball scale, drone position logic, broad military/industrial environment, realism level, and YouTube clickability.
- The main hook must stay large, sharp, and readable at small thumbnail size.
- If the reference has a large explosion, describe a similarly large, powerful, bright explosion with thick smoke, debris, visible damage, and nearby buildings or vehicles for scale. Do not shrink the explosion, push it into the distance, or turn it into a small fire.
- If the reference has a drone, keep it airborne and oriented toward the explosion or target area. Do not place the drone on the ground, flying away, or randomly sideways.
- Preserve any important simple highlight element from the reference, such as a circle or arrow around the drone, but change the exact style slightly.
- Keep the same broad environment. If the reference looks like a temperate Russian or Eastern European military-industrial area, keep that regional feel: concrete roads, hangars, military buildings, green or gray terrain, utilitarian base layout. Do not turn it into desert, tropical, American, Middle Eastern, mountain, or cinematic fantasy scenery.
- Change only secondary details: building arrangement, vehicle positions, smoke texture, debris pattern, drone angle slightly, annotation style, color accents, and background layout details.
- The image must look like a realistic detailed news/photo-style thumbnail, not CGI, not a movie poster, not a game render, not anime, not a painting, not glossy AI art.
- Avoid flags, readable text, logos, watermarks, timestamps, coordinates, HUD overlays, UI elements, gore, blood, blurry details, distorted people, or deformed vehicles unless the reference clearly has a specific simple annotation.

Output only the final image-generation prompt. No explanation.
""".strip()
    prompt = _call_thumbnail_step(
        REWRITE_SYSTEM,
        [{"role": "user", "content": rewrite_prompt}],
        "Thumbnail prompt rewrite",
        emit=emit,
    )

    return {
        "prompt": (prompt or "").strip(),
        "analysis": (analysis or "").strip(),
    }
