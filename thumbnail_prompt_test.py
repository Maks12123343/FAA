import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import api_client


DEFAULT_LANGUAGES = ["pl", "tr", "cs", "ro", "hu", "sw"]

ANALYSIS_SYSTEM = (
    "You are a precise YouTube thumbnail analyst. You describe what is visible "
    "and what makes the thumbnail clickable, without inventing unseen details."
)

ANALYSIS_PROMPT = """
Analyze the attached competitor YouTube thumbnail.

Do NOT write an image-generation prompt yet. First describe the visual strategy
so another model can create a similar but not identical thumbnail.

Return a compact structured analysis with these sections:
1. Main subject/event: what must be the focal point.
2. Supporting subject(s): secondary objects, people, vehicles, annotations, text.
3. Camera and framing: angle, distance, crop tightness, subject scale.
4. Main-hook geometry: approximate position and size as frame percentages, including center point, width/height, and which frame area it occupies.
5. Composition: where the main subject is placed and how much of the frame it fills.
6. Geographic/environmental context: climate, terrain, vegetation, architecture, road/ground type, and regional feel.
7. Attack direction: if a drone, missile, aircraft, weapon, or vehicle is visible, describe its apparent direction relative to the explosion/target.
8. Lighting and realism: daylight/night, color style, photo/news/cinematic/AI look.
9. Annotation/text style: circles, arrows, outlines, labels, large text, or no text.
10. What must stay similar.
11. What must change to avoid copying.
12. What to avoid.

Be strict about realism level. If it looks like a real news/drone/photo still,
say that clearly and warn against glossy AI poster style.
Be strict about subject scale. Estimate whether the main hook fills a large,
medium, or small part of the thumbnail. Do not describe an image as a generic
wide landscape/survey shot when the main hook is large and readable.
Do not recommend moving the main subject/event to a different side of the frame
as an anti-copying change. Only secondary details should move.
""".strip()

REWRITE_SYSTEM = (
    "You write production-ready image-generation prompts for YouTube thumbnails. "
    "You preserve the competitor thumbnail strategy while changing concrete details. "
    "Always write the final image-generation prompt in English."
)

REWRITE_TEMPLATE = """
Based on this competitor thumbnail analysis, write ONE final image-generation prompt.

Target language for any visible thumbnail text: {language}
Variant: {variant_id} of {variant_count}

Write the final image-generation prompt in English only. Do not translate the
prompt into the target language. The target language applies only to visible
thumbnail text, and only if visible text is truly needed.

Competitor analysis:
{analysis}

Universal rules:
- Preserve the competitor thumbnail's visual strategy, not the exact image.
- The main subject/event must be large, sharp, and immediately readable at small YouTube thumbnail size.
- If the competitor's key hook is an explosion, destroyed object, face, vehicle, weapon, map, drone, building, fire, injury, or other action, keep that hook as the clear focal point.
- The main visual hook must match the competitor's intensity and scale. Do not weaken it, shrink it, push it into the distance, or make it a background detail.
- Lock the main-hook geometry to the competitor thumbnail: keep the same approximate frame position, same quadrant, same visual center, same foreground/midground depth, and same pixel footprint. This is allowed to be very close to the reference; uniqueness must come from details, not from moving or shrinking the hook.
- If the analysis gives approximate percentages for the main hook, repeat those percentages in the final prompt and do not contradict them.
- Do not move the main hook to the left, right, top, or bottom as a variation. Do not place the explosion on a different side of the base if the reference keeps it central.
- Preserve logical attack direction. If a drone, missile, aircraft, weapon, or vehicle is shown as the implied cause of the explosion, orient it so it visually points, flies, or aims toward the explosion/target, not away from it or randomly sideways.
- For drone-attack thumbnails, the drone nose/body direction should clearly suggest movement toward the blast or target area. If needed, use a subtle motion angle or launch/flight direction, but do not add HUD graphics or telemetry.
- Match the competitor's subject scale. If the competitor shows the main hook filling a large part of the frame, the new prompt must ask for a similarly large and readable hook.
- Use a thumbnail-optimized crop around the main hook, not a distant surveillance/archive wide shot. The main hook should feel slightly closer and larger than a neutral documentary frame while still staying realistic.
- If the competitor image is aerial, keep it as a medium-height thumbnail crop, not a high-altitude survey/map view. Buildings, vehicles, damage, fire, and the main hook must remain clearly readable, not tiny details.
- When the reference explosion/fireball occupies a large part of the thumbnail, keep a similar pixel footprint in the new image, roughly one-third to one-half of the frame height when appropriate.
- Never make the final prompt ask for a wide-angle, panoramic, high-altitude, full-compound survey, map-like view, distant view, or a landscape-first aerial photo. Use "medium-height aerial thumbnail crop" or "close aerial thumbnail crop" instead.
- For aerial references, the final prompt should explicitly use the wording "medium-height aerial thumbnail crop" or "close aerial thumbnail crop".
- The final prompt must start with the main hook and its locked geometry, not with the landscape or general setting.
- Do not include more empty fields, forest, sky, horizon, or background space than the competitor thumbnail. The military base/action area should fill most of the frame, and the main hook should stay near the visual center.
- If the competitor has a huge explosion, the final prompt must explicitly ask for a huge foreground/midground explosion that is as large and readable as the reference, not a small fire on a distant building.
- If the competitor's hook is an explosion, fire, smoke, or destruction, keep it similarly powerful and visually dominant: a large bright fireball, thick dark smoke, visible debris or damaged objects, strong contrast, and clear surrounding vehicles/buildings for scale. It should feel intense but still realistic, like the reference image, not fantasy or overblown CGI.
- Do not make the camera much farther away than the competitor image.
- Even if the competitor has a broad aerial feel, match its exact thumbnail scale and do not make the scene wider, higher, farther away, or more landscape-heavy.
- Preserve the competitor's realism level. If it looks like a real photo, news still, drone image, CCTV image, or screenshot, keep that look. Do not turn it into glossy AI art or a cinematic movie poster.
- Preserve the broad layout: similar subject placement, similar scale, similar background amount, similar lighting/time of day, similar annotation logic.
- Preserve the competitor thumbnail's broad geographic and environmental context: same climate, terrain, vegetation, architecture, road/ground type, and regional feel. You may change the exact layout and details, but do not move the scene into a clearly different biome or country style.
- If the reference looks like a temperate Eastern European/Russian military-industrial zone with green forests, fields, concrete roads, hangars, barracks, and military vehicles, do not turn it into a desert, Middle Eastern base, tropical area, mountains, or American-style compound.
- Change exact details so it is not a copy: different secondary object positions, background layout details, smoke/fire texture, debris shape, vehicle arrangement, annotation styling, and color accents. Do not change the main hook's approximate position or size.
- If the competitor uses simple annotations, use a similar simple annotation style but not identical. Avoid neon HUDs unless the competitor clearly uses them.
- Do not add giant text unless the competitor clearly uses large text. If the competitor thumbnail has no visible words, the new thumbnail should have no visible words. If text is used, keep it short and in the target language.
- Do not add any visible words, numbers, dates, timestamps, coordinates, altitude readouts, camera telemetry, watermarks, UI text, or small technical overlays unless the competitor thumbnail clearly has them.
- For drone thumbnails, avoid weapon-sight crosshairs, red targeting reticles, HUD overlays, surveillance readouts, LAT/LON/ALT labels, and date/time stamps. A simple YouTube-style circle or arrow around the drone is allowed if the competitor uses that type of annotation.
- Avoid making every channel variant look the same, but vary only secondary details: smoke texture, debris shape, vehicle arrangement, building details, annotation stroke, or color accent. Keep the main-hook geometry, camera scale, and composition locked.

Output only the final image-generation prompt. No explanation.
""".strip()

BANNED_DISTANCE_TERMS = [
    "wide-angle",
    "wide angle",
    "wide aerial",
    "wide shot",
    "medium-wide",
    "medium wide",
    "panoramic",
    "high-altitude",
    "high altitude",
    "full-compound",
    "full compound",
    "entire compound",
    "map-like",
    "landscape-first",
    "distant aerial",
    "distant view",
    "survey photograph",
    "survey view",
    "establishing shot",
    "bird's-eye view",
    "birds-eye view",
]


def _video_id_from_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urllib.parse.urlparse(url)

    if parsed.netloc.lower().endswith("youtu.be"):
        vid = parsed.path.strip("/").split("/")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or ""):
            return vid

    query = urllib.parse.parse_qs(parsed.query)
    if query.get("v"):
        vid = query["v"][0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or ""):
            return vid

    m = re.search(r"(?:/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)

    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)

    raise ValueError("Could not extract YouTube video id from URL")


def _download_url(url: str, path: str) -> bool:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError:
        return False
    if len(data) < 5_000:
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def _download_youtube_thumbnail(video_id: str, out_dir: str) -> str:
    candidates = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]
    for idx, url in enumerate(candidates):
        path = os.path.join(out_dir, f"source_thumbnail_{idx + 1}.jpg")
        if _download_url(url, path):
            final_path = os.path.join(out_dir, "source_thumbnail.jpg")
            os.replace(path, final_path)
            return final_path
    raise RuntimeError(f"Could not download YouTube thumbnail for video id {video_id}")


def _image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_info(path: str) -> str:
    size = os.path.getsize(path)
    try:
        from PIL import Image

        with Image.open(path) as img:
            return f"{img.width}x{img.height}, {size} bytes"
    except Exception:
        return f"{size} bytes"


def _call_pioneer(system: str, messages: list, timeout: int = 180, use_rewrite_model: bool = False) -> str:
    text, _ = api_client.call_pioneer(
        system,
        messages,
        timeout=timeout,
        max_retries=3,
        step_label="thumbnail_test",
        use_rewrite_model=use_rewrite_model,
    )
    return (text or "").strip()


def _prompt_distance_issue(prompt: str) -> str:
    text = (prompt or "").lower()
    for term in BANNED_DISTANCE_TERMS:
        if term in text:
            return term
    return ""


def analyze_thumbnail(image_path: str) -> str:
    data_url = _image_data_url(image_path)
    return _call_pioneer(
        ANALYSIS_SYSTEM,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
        timeout=180,
        use_rewrite_model=False,
    )


def rewrite_prompt(analysis: str, language: str, variant_id: int, variant_count: int,
                   use_rewrite_model: bool = False) -> str:
    base_msg = REWRITE_TEMPLATE.format(
        language=language,
        variant_id=variant_id,
        variant_count=variant_count,
        analysis=analysis,
    )
    feedback = ""
    last_prompt = ""
    for attempt in range(1, 4):
        user_msg = base_msg + feedback
        prompt = _call_pioneer(
            REWRITE_SYSTEM,
            [{"role": "user", "content": user_msg}],
            timeout=180,
            use_rewrite_model=use_rewrite_model,
        )
        last_prompt = prompt
        issue = _prompt_distance_issue(prompt)
        if not issue:
            return prompt
        print(f"[thumbnail_test] rejected prompt attempt {attempt}/3: distance term '{issue}'")
        feedback = (
            "\n\nPrevious prompt was rejected because it still asked for a too-distant view "
            f"or used the forbidden distance wording: {issue!r}. Rewrite it as a medium-height "
            "or close aerial thumbnail crop. Keep the action/base large, readable, and near "
            "the visual center. Do not use any forbidden distance wording."
        )
    return last_prompt


def main() -> int:
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Download a YouTube thumbnail, analyze it, and generate rewritten thumbnail prompts."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--language", default="same as competitor", help="Target language for thumbnail text")
    parser.add_argument(
        "--languages",
        default="",
        help="Comma-separated target languages, or 'all' for pl,tr,cs,ro,hu,sw.",
    )
    parser.add_argument("--variants", type=int, default=3, help="How many prompt variants to generate")
    parser.add_argument("--out", default="", help="Output directory. Default: thumbnail_tests/<video_id>_<time>")
    parser.add_argument(
        "--rewrite-model",
        action="store_true",
        help="Use the dedicated rewrite model for the text prompt rewrite step.",
    )
    args = parser.parse_args()

    video_id = _video_id_from_url(args.url)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.getcwd(), "thumbnail_tests", f"{video_id}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[thumbnail_test] video_id={video_id}")
    print(f"[thumbnail_test] output={out_dir}")

    image_path = _download_youtube_thumbnail(video_id, out_dir)
    print(f"[thumbnail_test] thumbnail saved: {image_path}")
    print(f"[thumbnail_test] thumbnail info: {_image_info(image_path)}")

    print("[thumbnail_test] analyzing competitor thumbnail...")
    step_started = time.monotonic()
    analysis = analyze_thumbnail(image_path)
    print(f"[thumbnail_test] analysis took {time.monotonic() - step_started:.1f}s")
    analysis_path = os.path.join(out_dir, "analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis + "\n")
    print(f"[thumbnail_test] analysis saved: {analysis_path}")

    if args.languages.strip().lower() == "all":
        languages = list(DEFAULT_LANGUAGES)
    else:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if not languages:
        languages = [args.language]
    print(f"[thumbnail_test] languages parsed: {', '.join(languages)}")

    prompts = []
    variant_count = max(1, args.variants)
    total_prompts = len(languages) if args.languages else variant_count

    if args.languages:
        for i, language in enumerate(languages, start=1):
            print(f"[thumbnail_test] writing prompt for language {i}/{len(languages)}: {language}...")
            step_started = time.monotonic()
            prompt = rewrite_prompt(
                analysis,
                language=language,
                variant_id=i,
                variant_count=len(languages),
                use_rewrite_model=args.rewrite_model,
            )
            print(f"[thumbnail_test] prompt {language} took {time.monotonic() - step_started:.1f}s")
            item = {"language": language, "variant": i, "prompt": prompt}
            prompts.append(item)
            safe_lang = re.sub(r"[^A-Za-z0-9_-]+", "_", language)[:24] or f"lang_{i:02d}"
            prompt_path = os.path.join(out_dir, f"prompt_{i:02d}_{safe_lang}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt + "\n")
            print(f"[thumbnail_test] prompt saved: {prompt_path}")
    else:
        for i in range(1, variant_count + 1):
            print(f"[thumbnail_test] writing prompt variant {i}/{variant_count}...")
            step_started = time.monotonic()
            prompt = rewrite_prompt(
                analysis,
                language=args.language,
                variant_id=i,
                variant_count=variant_count,
                use_rewrite_model=args.rewrite_model,
            )
            print(f"[thumbnail_test] prompt variant {i} took {time.monotonic() - step_started:.1f}s")
            item = {"language": args.language, "variant": i, "prompt": prompt}
            prompts.append(item)
            prompt_path = os.path.join(out_dir, f"prompt_{i:02d}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt + "\n")
            print(f"[thumbnail_test] prompt saved: {prompt_path}")

    summary = {
        "url": args.url,
        "video_id": video_id,
        "language": args.language,
        "languages": languages,
        "thumbnail": image_path,
        "analysis": analysis,
        "prompts": prompts,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== PROMPTS ===")
    for i, item in enumerate(prompts, start=1):
        print(f"\n--- prompt_{i:02d} [{item['language']}] ---\n{item['prompt']}")
    print(f"[thumbnail_test] total took {time.monotonic() - started_at:.1f}s")
    print(f"\n[thumbnail_test] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
