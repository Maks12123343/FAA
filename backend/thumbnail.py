"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os
import time

from backend import api_client
from backend import languages as lang_utils

THUMBNAIL_ATTEMPTS = 3

ANALYSIS_PROMPT = """
You are an expert visual analyst, reverse-prompt engineer, and YouTube thumbnail
image-generation prompt writer.

Analyze the provided reference thumbnail as a visual reference only. Analyze the
actual pixels, not the apparent topic. Do not verify facts and do not claim that
the scene is real. Reconstruct the CORE PHOTOGRAPHIC SCENE in enough detail that
another image model could create a very similar news photograph.

Inspect all of the following:
- aspect ratio, full-frame 16:9 composition, camera height, distance, and angle;
- foreground, middle ground, background, empty sky/terrain, and visual balance;
- exact visual hierarchy and approximate frame positions and sizes of key objects;
- the main subject, vehicles, buildings, roads, bridges, ships, aircraft, drones,
  industrial or military equipment, and their orientation and interaction;
- explosion location, fireball scale, flame shape, smoke density and direction,
  debris, heat distortion, impact point, visible damage, and nearby scale cues;
- terrain, architecture, vegetation, weather, haze, lighting, color palette,
  shadows, reflections, sensor noise, motion blur, and realistic compression;
- visible circles, arrows, labels, outlines, or other thumbnail effects.

If black bars are present, ignore them and reconstruct the underlying scene as a
full 16:9 frame. Preserve the main hook and its scale. Do not turn a close,
dramatic subject into a distant landscape. Identify specific-looking objects
without inventing exact model numbers that are not visually clear. Warn against
CGI, a movie poster, a game render, glossy AI art, or fantasy scenery when the
reference is a realistic news/photo/drone still.

Return exactly these four sections and nothing else:

### MASTER PROMPT
Write a very detailed standalone English image-generation prompt with sections
for CORE SCENE, CAMERA AND COMPOSITION, MAIN SUBJECT, VEHICLES / EQUIPMENT,
EXPLOSION, SMOKE, DAMAGE, DRONE / AIRCRAFT, LOCATION / BACKGROUND, LIGHTING,
and REAL-PHOTO QUALITY. Require 1920x1080 full-frame 16:9, no black bars, no
distortion, and preserve the visual hierarchy of the reference.

### LOCKED ELEMENTS
List the scene identity, main event/target, approximate composition, explosion
importance, major subject/equipment, environment, and visual hierarchy that must
not be substantially changed.

### SAFE VARIABLE ELEMENTS
List only low-impact details that can vary later: subtle camera shift or crop,
minor secondary object positions, realistic paint shades, smoke drift, fireball
internal shape, small debris, vegetation, and similar details. Do not put major
story elements here.

### NEGATIVE PROMPT
Write a scene-specific negative prompt preventing CGI, glossy 3D render, poster
lighting, malformed or duplicated objects, wrong aircraft type, impossible
geometry, floating objects, weak or fantasy explosions, unwanted text, logos,
watermarks, HUD/UI, gore, and black bars.
""".strip()

REWRITE_SYSTEM = (
    "You are an expert image-prompt editor for realistic YouTube news thumbnails. "
    "Create minimal, natural variations of a successful reference concept. "
    "Always write the final image-generation prompt in English."
)


_VARIANT_IDS = {
    "pl": 1, "tr": 2, "cs": 3, "ro": 4, "hu": 5,
    "sv": 6, "fi": 7, "hr": 8, "da": 9, "bg": 10,
}


def _variant_id(language: str) -> int:
    code = (language or "").strip().lower()
    if code in _VARIANT_IDS:
        return _VARIANT_IDS[code]
    # Keep custom languages deterministic without making the number visible.
    return (sum(ord(ch) for ch in code) % 10) + 1


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
            text, _ = api_client.call_rewrite_api(
                system,
                messages,
                timeout=90,
                max_retries=2,
                emit=emit,
                step_label="thumbnail",
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
You will receive the result of a careful reference-image analysis. Convert it into
one complete standalone image-generation prompt for a new YouTube thumbnail.

VIDEO TITLE (context only):
{title or '(unknown)'}

VARIANT_ID: {_variant_id(language)}
TARGET LANGUAGE: {language_name}
PREVIOUS VARIATIONS: none recorded for this production batch.

The output must preserve approximately 85-95 percent of the reference concept.
This is a minimal variation, not a redesign. Preserve every LOCKED ELEMENT:
main event and target type, main subject, broad environment, camera distance,
visual hierarchy, approximate explosion power and importance, correct drone or
aircraft type, and the relationship between the drone and target.

Select approximately 3-7 small, natural changes from SAFE VARIABLE ELEMENTS.
Use VARIANT_ID only as an invisible diversification cue. Do not mention it in
the generated image. Do not mechanically mirror every variant. Horizontal
mirroring is optional and only allowed for some variants; if used, correct the
orientation, smoke drift, lighting, and secondary object positions so it is not a
simple Photoshop flip.

Preserve realistic damage and previously damaged objects. Preserve the same
explosion scale: never turn a large explosion into a small fire or a realistic
detonation into a fantasy/nuclear mushroom cloud. Preserve the correct drone
type: an FPV multirotor must remain an FPV multirotor, and a fixed-wing drone
must remain fixed-wing.

The image must be a realistic imperfect news/photo/drone still with restrained
saturation, natural shadows and reflections, atmospheric haze, mild sensor noise,
slight JPEG compression, and plausible heat distortion. Avoid glossy CGI,
movie-poster lighting, perfect symmetry, anime, painting, game render, gore,
readable text, logos, watermarks, timestamps, HUD/UI, flags, or black bars.
Require 1920x1080 full-frame 16:9, no letterboxing, no stretched geometry, and a
sharp readable main hook at small thumbnail size. Any visible text is forbidden;
the target language is retained only as context for this production.

REFERENCE ANALYSIS:
{analysis}

Return exactly:

### VARIANT PROMPT
A complete standalone English image-generation prompt. Do not say "same as
master" and do not rely on the image model remembering another prompt.

### NEGATIVE PROMPT
The adapted negative prompt.

No explanation or commentary.
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
