"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os
import time

from backend import api_client
from backend import languages as lang_utils

THUMBNAIL_ATTEMPTS = 3

ANALYSIS_PROMPT = """
You are a professional visual reverse-engineering system for YouTube thumbnail generation.

Your job is to analyze the PROVIDED REFERENCE IMAGE and write a highly accurate English image-generation prompt that reconstructs the reference as closely as possible.

Your task is NOT to improve the composition.
Your task is NOT to invent a better scene.
Your task is NOT to reinterpret the story.

**THE REFERENCE IMAGE IS THE SOURCE OF TRUTH.**

The generated MASTER PROMPT must preserve the visible image as literally as possible.

---

# CRITICAL RULE: REFERENCE FIRST

Before describing anything, visually inspect the actual reference.

Do not infer scene layout from the topic.

Do not assume where objects "should" be.

Do not reverse directions.

Do not change camera side.

Do not invent a different landscape.

Do not redesign vehicles, bridges, ships, buildings, roads, or explosions unless the reference itself is visually ambiguous.

When uncertain, use a slightly more generic description rather than inventing a specific detail.

---

# 1. BUILD A VISUAL COORDINATE MAP FIRST

Internally divide the image into:

- upper-left
- upper-center
- upper-right
- center-left
- center
- center-right
- lower-left
- lower-center
- lower-right

Identify the exact approximate location of every major visual element.

Determine:

- where the main subject begins and ends;
- where the road / bridge / ship / railway / coastline actually runs;
- which direction vehicles face;
- which direction the convoy travels;
- where the explosion originates;
- where the smoke rises;
- where the highlighted drone is located;
- which side contains foreground objects;
- which side contains background objects.

**Do not write the final prompt until these spatial relationships are consistent.**

Before finalizing, verify:

1. LEFT and RIGHT are correct.
2. FOREGROUND and BACKGROUND are correct.
3. The road / bridge / ship direction matches the image.
4. The explosion is on the correct object and in the correct part of the frame.
5. The drone is in the correct quadrant.
6. The camera viewpoint matches the image.

---

# 2. PRESERVE THE ORIGINAL COMPOSITION

The MASTER PROMPT must recreate the reference composition approximately 95–100%.

Preserve:

- main camera angle;
- camera height;
- camera side;
- crop;
- perspective;
- focal distance;
- dominant object positions;
- road direction;
- convoy direction;
- ship direction;
- bridge orientation;
- explosion location;
- smoke location;
- drone position;
- foreground/background balance.

Do not intentionally mirror anything.

Do not rotate the scene.

Do not shift the main subject.

Do not change the basic geometry.

This first stage creates the ORIGINAL MASTER VERSION only.

---

# 3. DO NOT OVER-SPECIFY UNCERTAIN DETAILS

Only describe exact vehicle, aircraft, drone, ship, weapon, or infrastructure details when they are visually clear.

For example:

If a truck appears to be a generic Soviet/Russian heavy military cargo truck, describe it that way.

Do NOT invent a precise model unless the reference clearly supports it.

If the drone payload is visible but unclear, describe:

"visible attached payload"

instead of inventing exact dimensions, mounting hardware, or warhead construction.

Avoid unnecessary numerical constraints such as:

- exact object percentages;
- exact focal length;
- exact meter distances;
- exact vehicle counts when partially obscured.

Use visual relationships instead:

"large enough to remain clearly visible in thumbnail scale"

"slightly smaller than"

"occupying the upper-right portion"

"close to the foreground"

---

# 4. MAIN SUBJECT

Identify the actual dominant physical subject.

Examples:

- military convoy;
- train;
- warship;
- bomber;
- oil terminal;
- bridge;
- military base.

Describe its:

- exact approximate frame position;
- orientation;
- scale;
- visible type;
- colors;
- wear;
- damage;
- relationship to the explosion.

Do not substitute a different subject type.

---

# 5. VEHICLES AND EQUIPMENT

Reproduce the visible vehicle/equipment mix as accurately as possible.

Preserve:

- approximate number visible;
- foreground vehicle positions;
- convoy density;
- tank/truck distribution;
- vehicle orientation;
- vehicle colors;
- whether vehicles are intact, burning, overturned, or destroyed.

Do not add large new categories of vehicles that do not exist in the reference.

Do not make all vehicles identical.

Maintain physically realistic:

- wheel placement;
- tracks;
- suspension;
- contact with ground;
- vehicle scale;
- shadows;
- perspective.

---

# 6. EXPLOSION

If an explosion is present, reproduce its visual importance faithfully.

Analyze:

- exact impact location;
- whether it originates from a vehicle, ship, aircraft, building, tank, roadway, bridge, storage tank, etc.;
- width;
- height;
- flame geometry;
- amount of visible white/yellow core;
- orange flame lobes;
- dark internal flame cavities;
- lower fire connection to the target;
- debris;
- dust;
- reflections.

The explosion may preserve the slightly exaggerated YouTube-thumbnail scale visible in the reference.

However:

Do not make it dramatically larger than the reference.
Do not make it smaller.
Do not detach it from the target.
Do not convert it into a nuclear blast.
Do not invent a perfect mushroom cloud.

---

# 7. DAMAGE MUST MATCH THE REFERENCE

Carefully distinguish between:

- main blast damage;
- previously damaged vehicles or structures;
- secondary fires;
- older smoke;
- debris;
- intact objects.

If the reference shows several damaged vehicles before or behind the main explosion, preserve them.

If the main aircraft itself is damaged, describe the aircraft itself as damaged.

Do not mistakenly place all damage beside the subject.

If the reference shows an intact foreground vehicle, keep it intact.

---

# 8. SMOKE

Match:

- density;
- height;
- width;
- direction;
- number of smoke sources;
- color layers;
- interaction with the background.

Describe only the smoke direction actually visible.

Do not automatically use "drifting left" or "drifting right".

Infer it from the reference.

---

# 9. DRONE / AIRCRAFT

Identify the actual visible type.

Possible types include:

- FPV quadcopter;
- fixed-wing one-way attack drone;
- reconnaissance UAV.

Preserve the correct type.

Do not transform:

FPV drone → fixed-wing aircraft
fixed-wing drone → FPV quadcopter
drone → fighter jet
drone → helicopter

Describe:

- frame quadrant;
- approximate orientation;
- visible top / side / underside;
- banking;
- descent direction;
- target relationship.

If the drone is descending toward the target, explicitly describe the downward trajectory.

If a yellow oval surrounds the drone, preserve exactly ONE yellow oval.

Do not invent arrows, labels, text, or additional annotations.

---

# 10. LOCATION

Describe the actual visible environment.

Do not exaggerate environmental adjectives.

For example, if the reference shows dry rolling grassland, use:

"open dry rolling steppe with muted grass and sparse vegetation"

Do NOT automatically change this into:

"barren treeless highland"

unless the reference truly looks that way.

Match:

- terrain;
- vegetation;
- water;
- hills;
- road surface;
- buildings;
- industrial infrastructure;
- weather;
- sky;
- haze.

---

# 11. PHOTOGRAPHIC CHARACTER

The generated image must look like the reference photographically.

If the reference looks like a compressed online news photograph, preserve:

- slightly imperfect sharpness;
- mild JPEG compression;
- natural sensor noise;
- restrained saturation;
- atmospheric haze;
- moderate detail falloff;
- realistic motion blur;
- natural shadows;
- physically believable smoke and fire;
- slightly clipped explosion highlights;
- imperfect background clarity.

Do not automatically request:

- flawless 4K;
- cinematic lighting;
- extreme HDR;
- dramatic color grading.

The image should feel like a real captured photograph, not a render.

---

# 12. THUMBNAIL PRIORITY

Preserve the same visual hierarchy as the reference.

For example:

1. explosion;
2. convoy / aircraft / ship / facility;
3. highlighted drone;
4. background environment.

Do not make the drone or background more dominant than the main event unless that is true in the reference.

---

# 13. RESOLUTION

Unless the user explicitly specifies another format:

1920 × 1080
16:9
full-frame
no black bars
no letterboxing
no borders

If the source image itself contains black bars, ignore the black bars and reconstruct the underlying image as full 16:9.

---

# OUTPUT

Return exactly:

### MASTER PROMPT

Write one complete, standalone English image-generation prompt reproducing the reference as faithfully as possible.

Do NOT discuss your analysis.

Do NOT mention that you divided the image into coordinates.

Do NOT mention uncertainty.

The MASTER PROMPT must be detailed, but avoid unnecessary invented detail.

Then return:

### LOCKED ELEMENTS

List the major elements that later variants must preserve.

Include:

- camera side and direction;
- main subject;
- main subject position;
- road/bridge/ship orientation;
- explosion approximate position and scale;
- smoke approximate position;
- drone type and approximate quadrant;
- environment;
- major damaged/intact objects;
- overall visual hierarchy.

Then return:

### SAFE VARIABLE ELEMENTS

List only minor details that may later be changed slightly.

Examples:

- 2–3 secondary vehicle positions;
- small vehicle color shifts;
- exact smoke folds;
- exact flame lobes;
- small debris positions;
- drone bank angle;
- minor vegetation;
- minor building colors;
- minor background equipment.

Do NOT put major scene geometry in SAFE VARIABLE ELEMENTS.

Then return:

### NEGATIVE PROMPT

Create a reference-specific negative prompt preventing:

- wrong scene direction;
- wrong drone type;
- malformed vehicles;
- duplicated objects;
- impossible geometry;
- detached explosion;
- incorrect environment;
- CGI appearance;
- unwanted text;
- logos;
- watermarks;
- black bars;
- gore.

Return only these four sections.
""".strip()

REWRITE_SYSTEM = (
    "You are an expert image-prompt editor for realistic YouTube news thumbnails. "
    "Create minimal, natural variations of a successful reference concept. "
    "Always write the final image-generation prompt in English."
)


ONE_SHOT_PROMPT = r"""
You are reverse-engineering the PROVIDED REFERENCE IMAGE for a high-performing
YouTube breaking-news thumbnail. Inspect the actual attached pixels carefully;
they are the only visual source of truth. The title is context only.

Create one complete, self-contained English prompt for Google Flow. Flow will
receive only your written prompt, not the image, so every important visual fact
must be described explicitly. The new thumbnail must be at least as powerful,
realistic, readable, and clickable as the reference, while remaining a new
variation rather than a one-to-one copy.

LOCK THE REFERENCE:
- Preserve the exact visible location type, camera height and angle, crop,
  horizon, perspective, land/water relationship, and visual hierarchy.
- Describe the left, center, right, foreground, middle ground, and background.
- Preserve the location fingerprint: major tanks, roads, pipe corridors,
  docks, shoreline, water, vessels, buildings, and large empty areas.
- Preserve the same destruction footprint: which structures, tanks, platforms,
  vehicles, decks, roads, and roofs are burning, damaged, collapsed, blackened,
  or still intact. Do not replace visible destruction with a clean facility.
- Do not invent a different landscape, country, architecture, aircraft type,
  or story. Do not make the main event smaller, weaker, farther away, or less
  colorful than the reference.

If an explosion is visible, describe its exact position and footprint, its
relation to nearby objects, the broad multi-lobed fireball, bright yellow-white
core, orange and red-orange flames, dark cavities, sparks, debris, heat haze,
ground/structure connection, smoke volume, smoke direction, and the visible
damage below it. The explosion must remain a dominant, powerful, realistic
click hook, never a small fire, distant flash, or firecracker.

If a drone or aircraft is visible, preserve its category, size, position,
silhouette, and relationship to the blast. Make it clearly pitched and tilted
downward toward the explosion, as if descending in an attack run: its nose or
front and flight axis point toward the blast, with a natural slight bank. It
must not look level, stationary, flying away, or unrelated to the strike.
Preserve its recognizable proportions and position. If a yellow circle or oval
is visible, preserve exactly one similar highlight around the same target, in
the same general position, with similar thickness, color, and visibility.

Use full-frame 1920x1080, 16:9, no borders or black bars. Make it look like a
real compressed breaking-news aerial photograph or video still: natural light,
realistic geometry, restrained saturation, believable smoke/fire, shadows,
reflections, haze, mild sensor noise, and slight JPEG compression. No readable
text, logos, captions, watermarks, HUD, CGI, cartoon style, poster design, or
fantasy elements.

Allow only tiny changes to secondary smoke curls, minor debris, reflections,
subtle haze, or cloud texture. Do not change the location, camera composition,
dominant event, explosion scale, destruction footprint, drone position, drone
relationship to the blast, annotation target, or visual hierarchy.

The final negative prompt must prevent weak/small explosions, wrong locations,
changed camera angles, distant panoramas, reduced destruction, intact
replacement structures, level or unrelated aircraft, aircraft flying away,
wrong aircraft types, missing annotations, random text, logos, watermarks,
CGI, cartoon rendering, and obvious AI artifacts.

VIDEO TITLE (context only):
{title}

TARGET LANGUAGE (no text should appear in the image): {language}

Return exactly these three sections and nothing else:

### VISUAL AUDIT
Give a concrete factual audit of the actual image, including location
fingerprint, camera/crop, dominant subjects and positions, explosion structure
and scale, visible damage, drone attack trajectory, annotation, colors,
lighting, and safe micro-variations. Do not invent facts.

### VARIANT PROMPT
Write one long, detailed, standalone English production prompt for Google Flow.
Do not say “use the reference image” or rely on any unavailable image.

### NEGATIVE PROMPT
Write a compact negative prompt protecting the scene, composition, scale,
destruction, drone trajectory, annotation, and photorealistic thumbnail quality.
""".strip()


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


def _thumbnail_section(text: str, heading: str, next_heading: str | None = None) -> str:
    value = str(text or "").strip()
    start = value.find(heading)
    if start < 0:
        raise RuntimeError(f"Thumbnail one-shot response is missing {heading}")
    start += len(heading)
    end = len(value)
    if next_heading:
        next_pos = value.find(next_heading, start)
        if next_pos >= 0:
            end = next_pos
    section = value[start:end].strip()
    if not section:
        raise RuntimeError(f"Thumbnail one-shot response has empty {heading}")
    return section


def analyze_and_rewrite(image_path: str, language: str, title: str = "", emit=None) -> dict:
    """Use one multimodal rewrite request to analyze and rewrite a thumbnail."""
    if not image_path or not os.path.exists(image_path):
        return {"prompt": "", "analysis": ""}

    _emit(emit, "Analyzing and rewriting source thumbnail in one request...")
    data_url = _image_data_url(image_path)
    language_name = lang_utils.configured_language_name(language)
    one_shot_prompt = ONE_SHOT_PROMPT.format(
        title=title or "(unknown)",
        language=language_name,
    )
    raw = _call_thumbnail_step(
        REWRITE_SYSTEM,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": one_shot_prompt},
            ],
        }],
        "Thumbnail one-shot analysis and prompt",
        emit=emit,
    )
    analysis = _thumbnail_section(raw, "### VISUAL AUDIT", "### VARIANT PROMPT")
    variant = _thumbnail_section(raw, "### VARIANT PROMPT", "### NEGATIVE PROMPT")
    negative = _thumbnail_section(raw, "### NEGATIVE PROMPT")
    prompt = f"### VARIANT PROMPT\n{variant}\n\n### NEGATIVE PROMPT\n{negative}"

    return {
        "prompt": (prompt or "").strip(),
        "analysis": (analysis or "").strip(),
    }
