"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os
import time

from backend import api_client

THUMBNAIL_ATTEMPTS = 3

ANALYSIS_PROMPT = """
Look at this YouTube thumbnail and describe it for creating a similar but unique thumbnail.

Cover these points:
1. Main subject/event: what must be the focal point.
2. Supporting subjects: secondary objects, people, vehicles, annotations, text.
3. Camera and framing: angle, distance, crop tightness, subject scale.
4. Main-hook geometry: approximate position and size as frame percentages, including center point, width/height, and frame area.
5. Composition: where the main subject is placed and how much of the frame it fills.
6. Geographic/environmental context: climate, terrain, vegetation, architecture, road/ground type, and regional feel.
7. Attack direction: if a drone, missile, aircraft, weapon, or vehicle is visible, describe its direction relative to the explosion/target.
8. Lighting and realism: daylight/night, color style, photo/news/cinematic/AI look.
9. Annotation/text style: circles, arrows, outlines, labels, large text, or no text.
10. What must stay similar.
11. What must change to avoid copying.
12. What to avoid.

Be strict about realism level. If it looks like a real news/drone/photo still,
say that clearly and warn against glossy AI poster style.
Be strict about subject scale. Do not describe a thumbnail as a generic wide
landscape when the main hook is large and readable.
Do not recommend moving the main subject/event to a different side of the frame
as an anti-copying change. Only secondary details should move.
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
    rewrite_prompt = f"""
Based on this competitor thumbnail analysis, write ONE final image-generation prompt.

Video title:
{title or '(unknown)'}

Target language for any visible thumbnail text:
{language}

Write the final image-generation prompt in English only. The target language applies only to visible thumbnail text, and only if visible text is truly needed.

Competitor analysis:
{analysis}

Universal rules:
- Preserve the competitor thumbnail's visual strategy, not the exact image.
- The main subject/event must be large, sharp, and immediately readable at small YouTube thumbnail size.
- If the competitor's key hook is an explosion, destroyed object, face, vehicle, weapon, map, drone, building, fire, injury, or other action, keep that hook as the clear focal point.
- The main visual hook must match the competitor's intensity and scale. Do not weaken it, shrink it, push it into the distance, or make it a background detail.
- Lock the main-hook geometry to the competitor thumbnail: keep the same approximate frame position, same quadrant, same visual center, same foreground/midground depth, and same pixel footprint. Uniqueness must come from details, not from moving or shrinking the hook.
- If the analysis gives approximate percentages for the main hook, repeat those percentages in the final prompt and do not contradict them.
- Do not move the main hook to a different side as a variation.
- Preserve logical attack direction. If a drone, missile, aircraft, weapon, or vehicle is shown as the implied cause of the explosion, orient it so it visually points, flies, or aims toward the explosion/target, not away from it or randomly sideways.
- For drone-attack thumbnails, the drone nose/body direction should clearly suggest movement toward the blast or target area.
- Use a thumbnail-optimized crop around the main hook, not a distant surveillance/archive wide shot.
- If the competitor image is aerial, keep it as a medium-height aerial thumbnail crop or close aerial thumbnail crop, not a high-altitude survey/map view.
- When the reference explosion/fireball occupies a large part of the thumbnail, keep a similar pixel footprint in the new image, roughly one-third to one-half of the frame height when appropriate.
- Never ask for a wide-angle, panoramic, high-altitude, full-compound survey, map-like view, distant view, or landscape-first aerial photo.
- The final prompt must start with the main hook and its locked geometry, not with the landscape or general setting.
- Do not include more empty fields, forest, sky, horizon, or background space than the competitor thumbnail.
- If the competitor has a huge explosion, the final prompt must explicitly ask for a huge foreground/midground explosion that is as large and readable as the reference, not a small fire on a distant building.
- If the hook is explosion, fire, smoke, or destruction, keep it similarly powerful and visually dominant: a large bright fireball, thick dark smoke, visible debris or damaged objects, strong contrast, and clear surrounding vehicles/buildings for scale. It should feel intense but still realistic, like the reference image, not fantasy CGI.
- Preserve the competitor's realism level. If it looks like a real photo, news still, drone image, CCTV image, or screenshot, keep that look. Do not turn it into glossy AI art or a cinematic movie poster.
- Preserve the broad layout: similar subject placement, similar scale, similar background amount, similar lighting/time of day, similar annotation logic.
- Preserve the competitor thumbnail's broad geographic and environmental context: same climate, terrain, vegetation, architecture, road/ground type, and regional feel. Do not move the scene into a clearly different biome or country style.
- If the reference looks like a temperate Eastern European/Russian military-industrial zone with green forests, fields, concrete roads, hangars, barracks, and military vehicles, do not turn it into a desert, Middle Eastern base, tropical area, mountains, or American-style compound.
- Change exact secondary details so it is not a copy: smoke texture, debris shape, vehicle arrangement, building details, annotation stroke, or color accent.
- Do not add giant text unless the competitor clearly uses large text. If the competitor thumbnail has no visible words, the new thumbnail should have no visible words.
- Do not add visible words, numbers, dates, timestamps, coordinates, altitude readouts, camera telemetry, watermarks, UI text, or small technical overlays unless the competitor thumbnail clearly has them.
- For drone thumbnails, avoid weapon-sight crosshairs, red targeting reticles, HUD overlays, surveillance readouts, LAT/LON/ALT labels, and date/time stamps. A simple YouTube-style circle or arrow around the drone is allowed only if the competitor uses that type of annotation.
- Avoid making every channel variant look the same, but vary only secondary details. Keep the main-hook geometry, camera scale, and composition locked.

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
