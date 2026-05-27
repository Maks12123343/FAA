import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE


def _gemini():
    from google import genai
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    return client, settings.get("gemini_model", "gemini-2.5-flash")


def _duration(path: str) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _frame_bytes(video_path: str, ratio: float) -> bytes:
    dur = _duration(video_path)
    ts  = max(0.01, dur * max(0.0, min(1.0, ratio)))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", video_path,
         "-vframes", "1", "-vf", "scale=640:-2", "-q:v", "4", tmp.name],
        capture_output=True, timeout=30,
    )
    data = b""
    if os.path.exists(tmp.name):
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
    return data


def _analysis_path(clip_path: str) -> str:
    return clip_path + ".analysis.json"


def _cache_valid(clip_path: str, cache_path: str) -> bool:
    """Return True if the cached analysis matches the current file (size + mtime)."""
    try:
        st = os.stat(clip_path)
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return (
            cached.get("_size") == st.st_size and
            cached.get("_mtime") == round(st.st_mtime, 2)
        )
    except Exception:
        return False


def analyze_clip(clip_path: str, clip_text: str = "") -> dict:
    """
    Analyze a single clip with Gemini: extract 3 frames → get description + tags.
    Result is cached to .analysis.json next to the clip file.
    Cache is invalidated automatically if the file size or mtime changes.
    """
    ap = _analysis_path(clip_path)
    if os.path.exists(ap) and _cache_valid(clip_path, ap):
        with open(ap, encoding="utf-8") as f:
            return json.load(f)

    from google.genai import types

    parts = []
    for ratio in [0.0, 0.5, 1.0]:
        fb = _frame_bytes(clip_path, ratio)
        if fb:
            parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    if not parts:
        analysis = {"description": "unknown", "tags": [], "text": clip_text}
        with open(ap, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        return analysis

    prompt = (
        "These are 3 frames (start / middle / end) from a short video clip.\n"
    )
    if clip_text:
        prompt += f"Audio transcript of this clip: \"{clip_text}\"\n"
    prompt += (
        "Answer ALL of the following:\n"
        "1. Describe in 1-2 sentences what is visually shown.\n"
        "2. List 10-15 tags (words/short phrases) for topics this footage could illustrate.\n"
        "3. Pick ONE category that best describes the main visual content:\n"
        "   city, factory, nature, technology, money, shipping, map, document,\n"
        "   protest, military, people, talking_head, generic\n"
        "4. Check for quality issues:\n"
        "   - is_blurry: true if most of the frame is out of focus or motion-blurred\n"
        "   - is_static: true if all 3 frames look nearly identical (no motion at all)\n"
        "5. Check for overlays and measure their size precisely:\n"
        "   - subtitles/captions at bottom:\n"
        "       action='crop_bottom', crop_percent=<% of frame height the subtitle bar occupies, 5-30>\n"
        "   - watermark/logo in ONE corner only:\n"
        "       action='crop_corner', crop_percent=<% of frame width/height the logo occupies from that edge, 3-20>\n"
        "   - watermark centered, on a person/object, or covering >25% of frame → action='reject', crop_percent=0\n"
        "   - is_blurry=true or is_static=true → action='reject', crop_percent=0\n"
        "   - clean clip → action='use', crop_percent=0\n"
        "crop_percent is the EXACT measured size — be precise, not just default 10 or 15.\n"
        "JSON only, no markdown:\n"
        "{\"description\": \"...\", \"tags\": [\"tag1\", ...], \"category\": \"generic\", "
        "\"is_blurry\": false, \"is_static\": false, "
        "\"action\": \"use\", \"crop_percent\": 0, \"watermark_position\": \"none\"}"
    )
    parts.append(prompt)

    import time as _time
    client, model = _gemini()
    analysis = None
    last_err = None
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=model, contents=parts)
            text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            analysis = json.loads(m.group() if m else text)
            break
        except Exception as e:
            last_err = e
            err_lower = str(e).lower()
            is_rate_limit = (
                "429" in str(e) or
                "quota" in err_lower or
                "resource_exhausted" in err_lower
            )
            is_auth_error = (
                "401" in str(e) or "403" in str(e) or
                "permission" in err_lower or
                "unauthenticated" in err_lower or
                "unauthorized" in err_lower or
                "credentials" in err_lower or
                "invalid_argument" in err_lower
            )
            if is_auth_error:
                # Auth/config errors won't be fixed by retry — raise immediately
                raise RuntimeError(
                    f"[clip_matcher] Gemini auth/config error (will not retry): {e}"
                ) from e
            elif is_rate_limit:
                wait = 10 * (attempt + 1)
                print(
                    f"[clip_matcher] Gemini rate limit (attempt {attempt+1}/3), "
                    f"retrying in {wait}s...",
                    flush=True,
                )
                _time.sleep(wait)
            else:
                break  # Unknown error — don't retry
    if analysis is None:
        print(f"[clip_matcher] analyze_clip failed after retries: {last_err}", flush=True)
        analysis = {"description": "unknown", "tags": []}

    # Ensure all fields present with safe defaults
    analysis.setdefault("action", "use")
    analysis.setdefault("crop_percent", 0)
    analysis.setdefault("watermark_position", "none")
    analysis.setdefault("category", "generic")
    analysis.setdefault("is_blurry", False)
    analysis.setdefault("is_static", False)

    # Enforce reject for blurry/static even if Gemini forgot
    if analysis.get("is_blurry") or analysis.get("is_static"):
        analysis["action"] = "reject"

    analysis["text"] = clip_text
    # Store file fingerprint so cache is invalidated if the file changes
    try:
        st = os.stat(clip_path)
        analysis["_size"]  = st.st_size
        analysis["_mtime"] = round(st.st_mtime, 2)
    except Exception:
        pass
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    return analysis



# ── Batch clip analysis (8 clips per Gemini call, 3 parallel workers) ─────────

_CLIP_BATCH_SIZE = 8


def _analyze_batch_chunk(items: list, client, model: str) -> list:
    """
    Send up to _CLIP_BATCH_SIZE clips to Gemini in ONE call.
    items: list of {"clip_path": str, "clip_text": str, "frames": list[bytes]}
    Returns list of analysis dicts in same order.
    """
    from google.genai import types
    import re as _re

    contents = [f"Analyze {len(items)} video clips. For each clip, 3 frames are shown (start/mid/end).\n"]

    for idx, item in enumerate(items):
        label = f"CLIP {idx + 1}"
        if item.get("clip_text"):
            label += f' (audio: "{item["clip_text"][:100]}")'
        contents.append(label + ":")
        for fb in item["frames"]:
            contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    contents.append(
        f"\nFor EACH of the {len(items)} clips answer ALL of the following:\n"
        "1. Describe in 1-2 sentences what is visually shown.\n"
        "2. List 10-15 tags (words/short phrases) for topics this footage could illustrate.\n"
        "3. Pick ONE category: city, factory, nature, technology, money, shipping, map, document, "
        "protest, military, people, talking_head, generic\n"
        "4. is_blurry: true if most of the frame is out of focus or motion-blurred\n"
        "   is_static: true if all 3 frames look nearly identical (no motion at all)\n"
        "5. Overlays:\n"
        "   - subtitles at bottom → action=crop_bottom, crop_percent=exact % of frame height\n"
        "   - corner logo → action=crop_corner, crop_percent=exact % of frame width/height\n"
        "   - centered watermark or >25% → action=reject, crop_percent=0\n"
        "   - blurry/static → action=reject, crop_percent=0\n"
        "   - clean → action=use, crop_percent=0\n"
        f"\nReply ONLY with a JSON array of exactly {len(items)} objects:\n"
        '[{"description": "...", "tags": [...], "category": "generic", '
        '"is_blurry": false, "is_static": false, "action": "use", "crop_percent": 0, '
        '"watermark_position": "none"}, ...]'
    )

    r = client.models.generate_content(model=model, contents=contents)
    text = _re.sub(r"^```(?:json)?\s*", "", r.text.strip())
    text = _re.sub(r"\s*```$", "", text)
    m = _re.search(r"\[.*\]", text, _re.DOTALL)
    raw = json.loads(m.group() if m else text)

    results = []
    for i, item in enumerate(items):
        analysis = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {"description": "unknown", "tags": []}
        analysis.setdefault("action", "use")
        analysis.setdefault("crop_percent", 0)
        analysis.setdefault("watermark_position", "none")
        analysis.setdefault("category", "generic")
        analysis.setdefault("is_blurry", False)
        analysis.setdefault("is_static", False)
        if analysis.get("is_blurry") or analysis.get("is_static"):
            analysis["action"] = "reject"
        analysis["text"] = item.get("clip_text", "")
        try:
            st = os.stat(item["clip_path"])
            analysis["_size"]  = st.st_size
            analysis["_mtime"] = round(st.st_mtime, 2)
        except Exception:
            pass
        results.append(analysis)
    return results


def analyze_all_clips(clips_index: list, emit=None) -> list:
    """
    Analyze all clips in the pool via Gemini: extract 3 frames → get description + tags.
    Results cached to .analysis.json. Only new/changed clips are analyzed.
    Uses batch processing: 8 clips per Gemini call, 3 parallel workers.
    Falls back to single-clip analysis if a batch fails.
    """
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cached     = []
    to_analyze = []

    for clip_info in clips_index:
        clip_path = clip_info.get("file", "")
        if not os.path.exists(clip_path):
            continue
        ap = _analysis_path(clip_path)
        if os.path.exists(ap) and _cache_valid(clip_path, ap):
            try:
                with open(ap, encoding="utf-8") as f:
                    data = json.load(f)
                data["file"] = clip_path
                cached.append(data)
            except Exception:
                to_analyze.append(clip_info)
        else:
            to_analyze.append(clip_info)

    if not to_analyze:
        return cached

    print(
        f"[clip_matcher] Analyzing {len(to_analyze)} new clips "
        f"({len(cached)} cached) in batch mode...",
        flush=True,
    )

    # Extract frames for all clips in parallel first
    def _extract_frames_for_clip(clip_info):
        clip_path = clip_info.get("file", "")
        frames = []
        for ratio in [0.0, 0.5, 1.0]:
            fb = _frame_bytes(clip_path, ratio)
            if fb:
                frames.append(fb)
        return {
            "clip_path": clip_path,
            "clip_text": clip_info.get("text", ""),
            "clip_id":   clip_info.get("id", ""),
            "frames":    frames,
        }

    print(f"[clip_matcher] Extracting frames for {len(to_analyze)} clips...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        items_with_frames = list(pool.map(_extract_frames_for_clip, to_analyze))
    items_with_frames = [it for it in items_with_frames if it["frames"]]

    batches = [
        items_with_frames[i:i + _CLIP_BATCH_SIZE]
        for i in range(0, len(items_with_frames), _CLIP_BATCH_SIZE)
    ]
    print(f"[clip_matcher] {len(items_with_frames)} clips → {len(batches)} batches of {_CLIP_BATCH_SIZE}", flush=True)

    done_count  = [0]
    lock        = threading.Lock()
    results_new = {}
    client, model = _gemini()

    def _process_batch(batch):
        for attempt in range(3):
            try:
                analyses = _analyze_batch_chunk(batch, client, model)
                for item, analysis in zip(batch, analyses):
                    analysis["id"]   = item["clip_id"]
                    analysis["file"] = item["clip_path"]
                    ap = _analysis_path(item["clip_path"])
                    with open(ap, "w", encoding="utf-8") as f:
                        json.dump(analysis, f, ensure_ascii=False, indent=2)
                    with lock:
                        done_count[0] += 1
                        n = done_count[0]
                        results_new[item["clip_id"]] = analysis
                    if emit:
                        emit("clips", f"Analyzing clip {n}/{len(items_with_frames)}...")
                print(f"[clip_matcher] {done_count[0]}/{len(items_with_frames)} analyzed", flush=True)
                return
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower()
                if is_rate and attempt < 2:
                    wait = 15 * (attempt + 1)
                    print(f"[clip_matcher] Rate limit, retry in {wait}s...", flush=True)
                    _time.sleep(wait)
                else:
                    print(f"[clip_matcher] Batch error (attempt {attempt+1}): {e} — falling back to single", flush=True)
                    # Fallback: analyze one by one
                    for item in batch:
                        try:
                            analysis = analyze_clip(item["clip_path"], item["clip_text"])
                            analysis["id"]   = item["clip_id"]
                            analysis["file"] = item["clip_path"]
                            with lock:
                                done_count[0] += 1
                                results_new[item["clip_id"]] = analysis
                        except Exception as e2:
                            print(f"[clip_matcher] Fallback failed {item['clip_path']}: {e2}", flush=True)
                    return

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[clip_matcher] Worker error: {e}", flush=True)

    print(f"[clip_matcher] Done: {len(results_new)} analyzed, {len(cached)} from cache", flush=True)
    return cached + list(results_new.values())
def _score_clip(clip_analysis: dict, section_text: str) -> float:
    """Score a clip against section text using tag/description overlap (no AI)."""
    section_lower = section_text.lower()
    s = 0.0

    for tag in clip_analysis.get("tags", []):
        tag_lower = tag.lower()
        if tag_lower in section_lower:
            s += 3.0
        else:
            for word in tag_lower.split():
                if len(word) > 3 and word in section_lower:
                    s += 1.0

    for word in clip_analysis.get("description", "").lower().split():
        if len(word) > 4 and word in section_lower:
            s += 0.5

    clip_text = clip_analysis.get("text", "").lower()
    if clip_text:
        for word in section_lower.split():
            if len(word) > 4 and word in clip_text:
                s += 0.3

    return s


def _validation_cache_path(clip_path: str, section_text: str) -> str:
    """Disk cache path for validate_clip_for_section results."""
    import hashlib
    h = hashlib.md5(section_text[:300].encode()).hexdigest()[:12]
    return clip_path + f".val_{h}.json"


def validate_clip_for_section(clip_path: str, section_text: str) -> float:
    """
    Ask Gemini to score how well a clip matches a script section.
    Returns float 0.0-1.0. Result cached to disk — same clip+section never calls Gemini twice
    even across different languages or produce() runs.
    """
    cache_path = _validation_cache_path(clip_path, section_text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return float(json.load(f).get("score", 0.0))
        except Exception:
            pass

    from google.genai import types

    parts = []
    for ratio in [0.0, 0.5, 1.0]:
        fb = _frame_bytes(clip_path, ratio)
        if fb:
            parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    if not parts:
        return 0.0

    prompt = (
        f'These are 3 frames from a video clip.\n'
        f'Script section this clip should illustrate: "{section_text[:300]}"\n'
        f'Rate how well this clip visually matches the script section.\n'
        f'JSON only: {{"score": 0.0}} where 0.0=does not match, 1.0=perfect match.'
    )
    parts.append(prompt)

    client, model = _gemini()
    try:
        r = client.models.generate_content(model=model, contents=parts)
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group() if m else text)
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.0

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"score": score}, f)
    except Exception:
        pass

   