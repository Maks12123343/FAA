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


DEFAULT_LANGUAGES = ["pl", "tr", "cs", "ro", "hu", "sv"]

ANALYSIS_SYSTEM = (
    "You are a precise YouTube thumbnail analyst. You describe what is visible "
    "and what makes the thumbnail clickable, without inventing unseen details."
)

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

REWRITE_TEMPLATE = """
Use the competitor thumbnail analysis as a visual reference only. Do not verify facts and do not make factual claims.

Create ONE English image-generation prompt for a new YouTube thumbnail.

Target language for any visible thumbnail text: {language}
Variant: {variant_id} of {variant_count}

Write the final image-generation prompt in English only. Do not translate the
prompt into the target language. The target language applies only to visible
thumbnail text, and only if visible text is truly needed.

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
        help="Comma-separated target languages, or 'all' for pl,tr,cs,ro,hu,sv.",
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
