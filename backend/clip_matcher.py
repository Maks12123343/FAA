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

# In-memory metadata registry populated by analyze_all_clips().
# Important for movie_library_mode: movie clips are already pre-analyzed in
# index.json, but we do not necessarily write a <clip>.analysis.json file next
# to the mp4 (often on a slow/read-only gdrive mount). batch_validate_candidates()
# receives only clip paths, so text-only validation reads descriptions/tags here.
_CLIP_META_BY_PATH: dict = {}


def _is_gemini_auth_error(err) -> bool:
    text = str(err).lower()
    return any(x in text for x in (
        "401", "403", "permission", "credentials",
        "unauthenticated", "unauthorized",
    ))


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
        # If clip already has pre-analyzed data (e.g. from movie library index.json),
        # use it directly — no Gemini call needed
        if clip_info.get("description") and clip_info.get("tags") is not None:
            meta = {
                "file":               clip_path,
                "id":                 clip_info.get("id", ""),
                "description":        clip_info["description"],
                "tags":               clip_info.get("tags", []),
                "category":           clip_info.get("category", "generic"),
                "is_blurry":          clip_info.get("is_blurry", False),
                "is_static":          clip_info.get("is_static", False),
                "action":             clip_info.get("action", "use"),
                "crop_percent":       clip_info.get("crop_percent", 0),
                "watermark_position": clip_info.get("watermark_position", "none"),
            }
            _CLIP_META_BY_PATH[clip_path] = meta
            cached.append(meta)
            continue
        ap = _analysis_path(clip_path)
        if os.path.exists(ap) and _cache_valid(clip_path, ap):
            try:
                with open(ap, encoding="utf-8") as f:
                    data = json.load(f)
                data["file"] = clip_path
                _CLIP_META_BY_PATH[clip_path] = data
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
                    _CLIP_META_BY_PATH[item["clip_path"]] = analysis
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
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[clip_matcher] Gemini auth/config error: {e}") from e
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


def match_clips_multi(section_texts: list, clips_index: list, top_n: int = 10, emit=None) -> list:
    """
    For each section text, find top-N best matching clips by tag/description overlap.
    clips_index: list of clip dicts with at least a "file" key (enriched by analyze_all_clips).
    Returns list of lists of FILE PATHS (strings), one inner list per section.
    """
    results = []
    for i, section_text in enumerate(section_texts):
        scored = []
        for clip in clips_index:
            score = _score_clip(clip, section_text)
            if score > 0:
                scored.append((score, clip))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Always return file paths (strings), never raw dicts
        top = []
        for _, c in scored[:top_n]:
            path = c.get("file", "") if isinstance(c, dict) else c
            if path:
                top.append(path)
        results.append(top)
        if emit:
            emit("match", f"Matched section {i+1}/{len(section_texts)}")
    return results


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
    except Exception as e:
        err = str(e).lower()
        if _is_gemini_auth_error(e):
            raise RuntimeError(f"[clip_matcher] Gemini auth/config error: {e}") from e
        score = 0.0

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"score": score}, f)
    except Exception:
        pass

    return score


# ── Text-only batch validation for movie library clips ───────────────────────

def _validate_movie_clips_text_batch(items: list, client, model: str) -> list:
    """
    Text-only Gemini validation for movie library clips (no frame extraction).
    items: [{"clip_path": str, "section_text": str, "description": str, "tags": list}, ...]
    Returns list of scores (float 0.0-1.0) in same order.
    """
    import re as _re

    if not items:
        return []

    prompt_parts = [f"Score how well each of {len(items)} movie clips matches its script segment.\n"]
    for idx, item in enumerate(items):
        tags_str = ", ".join(item.get("tags", [])[:10]) or "none"
        prompt_parts.append(
            f"CLIP {idx + 1}:\n"
            f"  Description: {item['description'][:300]}\n"
            f"  Tags: {tags_str}\n"
            f"  Script segment: \"{item['section_text'][:200]}\"\n"
        )
    prompt_parts.append(
        f"\nFor each clip, rate how well it visually illustrates its script segment.\n"
        f"Consider: mood, action, characters, setting, visual theme.\n"
        f"Reply ONLY with a JSON array of exactly {len(items)} objects:\n"
        f'[{{"score": 0.0}}, {{"score": 0.8}}, ...] where 0.0=no match, 1.0=perfect match.'
    )

    prompt = "\n".join(prompt_parts)
    try:
        response = client.models.generate_content(model=model, contents=[prompt])
        text = _re.sub(r"^```(?:json)?\s*", "", response.text.strip())
        text = _re.sub(r"\s*```$", "", text)
        m = _re.search(r"\[.*\]", text, _re.DOTALL)
        raw = json.loads(m.group() if m else text)
        scores = [float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0 for r in raw]
        # Gemini should return exactly len(items), but be defensive: pad/truncate so
        # every candidate always receives a deterministic score and zip() cannot drop it.
        if len(scores) < len(items):
            scores.extend([0.0] * (len(items) - len(scores)))
        return scores[:len(items)]
    except Exception as e:
        # Do NOT silently return/cache zero scores here. If the LLM/API is down or
        # returns invalid JSON, failing the validation is safer than poisoning the
        # per-clip cache and project validated_candidates.json with fake 0.0 scores.
        raise RuntimeError(f"[clip_matcher] text batch validation error: {e}") from e


def _pioneer_keys() -> list:
    settings = config.load_settings()
    keys = settings.get("pioneer_api_keys", [])
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    return [k for k in keys if k]


def _validate_movie_clips_text_pioneer_batch(items: list, api_key: str) -> list:
    """
    Text-only Pioneer.ai validation for movie library clips.
    Same contract as _validate_movie_clips_text_batch(), but uses the
    OpenAI-compatible Pioneer endpoint. Tries the given api_key first,
    then rotates through all available keys if it fails.
    """
    import re as _re
    import time as _time
    import urllib.request

    if not items:
        return []

    settings = config.load_settings()
    api_url  = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")
    model    = settings.get("pioneer_validation_model", settings.get("pioneer_model", "gemini-3.5-flash"))
    timeout  = int(settings.get("pioneer_timeout", 180) or 180)
    retries  = int(settings.get("pioneer_retries", 2) or 2)

    # Optimized prompt: if all items share same section_text, include it once
    unique_sections = set(item["section_text"][:200] for item in items)
    if len(unique_sections) == 1:
        section_text = items[0]["section_text"][:200]
        prompt_parts = [
            f'Script narration: "{section_text}"\n\n'
            f"Score how well each of {len(items)} movie clips matches this narration.\n"
        ]
        for idx, item in enumerate(items):
            tags_str = ", ".join(item.get("tags", [])[:10]) or "none"
            prompt_parts.append(
                f"CLIP {idx + 1}: {item['description'][:150]} [{tags_str}]\n"
            )
    else:
        prompt_parts = [f"Score how well each of {len(items)} movie clips matches its script segment.\n"]
        for idx, item in enumerate(items):
            tags_str = ", ".join(item.get("tags", [])[:10]) or "none"
            prompt_parts.append(
                f"CLIP {idx + 1}:\n"
                f"  Description: {item['description'][:150]}\n"
                f"  Tags: {tags_str}\n"
                f"  Script: \"{item['section_text'][:200]}\"\n"
            )
    prompt_parts.append(
        f"\nRate how well each clip visually illustrates the narration.\n"
        f"Consider: mood, action, characters, setting.\n"
        f"IMPORTANT: Score 0.0 for ANY of these:\n"
        f"- Credits/end credits (author names, directed by, produced by, cast list)\n"
        f"- Title cards, text-only screens, intertitles\n"
        f"- Black/blank screens, studio logos\n"
        f"- Static frames with only text and no action\n"
        f"Only score > 0 for clips showing actual visual ACTION (characters, scenes, environments).\n"
        f"Reply ONLY with a JSON array of {len(items)} objects:\n"
        f'[{{"score": 0.0}}, {{"score": 0.8}}, ...] where 0.0=no match, 1.0=perfect match.'
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "\n".join(prompt_parts)}],
        "stream": False,
    }).encode("utf-8")

    # Build key list: given key first, then all others as fallback
    all_keys = _pioneer_keys()
    keys_to_try = [api_key] + [k for k in all_keys if k != api_key]

    last_error = None
    for key in keys_to_try:
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(
                    api_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                text = _re.sub(r"^```(?:json)?\s*", "", text.strip())
                text = _re.sub(r"\s*```$", "", text)
                m = _re.search(r"\[.*\]", text, _re.DOTALL)
                raw = json.loads(m.group() if m else text)
                scores = []
                for r in raw:
                    if isinstance(r, dict):
                        scores.append(float(r.get("score", 0.0)))
                    elif isinstance(r, (int, float)):
                        scores.append(float(r))
                    else:
                        scores.append(0.0)
                if len(scores) < len(items):
                    scores.extend([0.0] * (len(items) - len(scores)))
                return scores[:len(items)]
            except Exception as e:
                last_error = e
                print(
                    f"[clip_matcher] Pioneer validation error (key ...{key[-12:]}, "
                    f"attempt {attempt+1}/{retries+1}): {e}",
                    flush=True,
                )
                if attempt < retries:
                    _time.sleep(10 * (attempt + 1))
                    continue
                break  # try next key

    raise RuntimeError(f"[clip_matcher] pioneer text batch validation error (all keys failed): {last_error}") from last_error


def _txt_cache_path(clip_path: str, section_text: str, cache_scope: str = "shared") -> str:
    import hashlib
    safe_scope = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cache_scope or "shared")[:40]
    h = hashlib.md5(("movie_txt_v3\n" + safe_scope + "\n" + section_text[:300]).encode()).hexdigest()[:12]
    return clip_path + f".val_txt_v3_{safe_scope}_{h}.json"


def _read_txt_cache(clip_path: str, section_text: str, cache_scope: str = "shared"):
    cp = _txt_cache_path(clip_path, section_text, cache_scope)
    if os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                return float(json.load(f).get("score", 0.0))
        except Exception:
            pass
    return None


def _write_txt_cache(clip_path: str, section_text: str, score: float, cache_scope: str = "shared"):
    cp = _txt_cache_path(clip_path, section_text, cache_scope)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump({"score": score}, f)
    except Exception:
        pass


def _get_clip_meta(clip_path: str) -> dict:
    if clip_path in _CLIP_META_BY_PATH:
        return _CLIP_META_BY_PATH[clip_path]
    ap = clip_path + ".analysis.json"
    if os.path.exists(ap):
        try:
            with open(ap, encoding="utf-8") as f:
                data = json.load(f)
                _CLIP_META_BY_PATH[clip_path] = data
                return data
        except Exception:
            pass
    return {}


# ── Batch validation (8 pairs per Gemini call, 3 parallel workers) ────────────

_VALIDATION_BATCH_SIZE = 8


def _validate_batch_chunk(items: list, client, model: str) -> list:
    """
    items: [{"clip_path": str, "section_text": str, "frames": list[bytes]}, ...]
    Returns list of scores (float 0.0-1.0) in same order.
    """
    from google.genai import types
    import re as _re

    contents = [f"Score how well each of {len(items)} video clips matches its script section.\n"]
    for idx, item in enumerate(items):
        contents.append(f'CLIP {idx + 1} — Script: "{item["section_text"][:200]}":')
        for fb in item["frames"]:
            contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    contents.append(
        f"\nFor each clip, rate how well it visually matches its script section.\n"
        f"Reply ONLY with a JSON array of exactly {len(items)} objects:\n"
        f'[{{"score": 0.0}}, {{"score": 0.8}}, ...] where 0.0=no match, 1.0=perfect match.'
    )

    response = client.models.generate_content(model=model, contents=contents)
    text = _re.sub(r"^```(?:json)?\s*", "", response.text.strip())
    text = _re.sub(r"\s*```$",          "", text)
    m    = _re.search(r"\[.*\]", text, _re.DOTALL)
    try:
        raw = json.loads(m.group() if m else text)
        return [float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0 for item in raw]
    except Exception as e:
        raise RuntimeError(f"[clip_matcher] batch validation parse error: {e}") from e


def _build_validation_result(
    section_candidates: list,
    section_texts: list,
    scores: dict,
) -> list:
    result = []
    for sec_idx, (clips, section_text) in enumerate(zip(section_candidates, section_texts)):
        scored = [
            (clip_path, scores.get((sec_idx, clip_path), 0.0))
            for clip_path in clips
            if os.path.exists(clip_path)
        ]
        scored.sort(key=lambda x: -x[1])
        result.append(scored)
    return result


def batch_validate_candidates(
    section_candidates: list,
    section_texts: list,
    settings: dict,
    emit=None,
    movie_library_mode: bool = False,
) -> list:
    """
    Batch validate competitor clip candidates per section with Gemini.
    section_candidates: list[list[str]] — clip paths per section (from match_clips_multi)
    section_texts: list[str] — original transcript section texts
    Returns: list[list[(clip_path, score)]] sorted by score desc per section.
    Disk-cached per (clip, section_text) — first language pays, others are free.

    movie_library_mode=True: skip frame extraction (clips are on slow gdrive mount)
    and validate with Gemini text-only using movie index description/tags.
    Validates top-3 first; if none score >= 0.75, validates candidates 4-5 too.
    """
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Movie library: text-only validation (no frame extraction).
    # Runs sections in parallel across independent backends:
    #   - 1 Vertex Gemini worker
    #   - up to 3 Pioneer.ai Gemini-proxy workers (one API key per worker)
    # Each section stays on exactly one worker/backend, so scores cannot be mixed.
    # Per section: validate top-3 first; if none ≥ 0.75, validate clips 4-5 too; take best.
    if movie_library_mode:
        MOVIE_THRESHOLD = 0.75
        vertex_client, vertex_model = _gemini()
        backends = [{"kind": "vertex", "name": "Vertex", "client": vertex_client, "model": vertex_model}]
        for i, key in enumerate(_pioneer_keys()[:3], start=1):
            backends.append({"kind": "pioneer", "name": f"Pioneer-{i}", "api_key": key})

        sections = list(enumerate(zip(section_candidates, section_texts)))
        assigned = [[] for _ in backends]
        for n, section_arg in enumerate(sections):
            assigned[n % len(backends)].append(section_arg)

        results_by_sec = {}
        totals = {"validated": 0, "cached": 0}
        totals_lock = threading.Lock()

        def _items_for(clips_to_score: list, section_text: str) -> list:
            items = []
            for c in clips_to_score:
                meta = _get_clip_meta(c)
                items.append({
                    "clip_path": c,
                    "section_text": section_text,
                    "description": meta.get("description", os.path.basename(c)),
                    "tags": meta.get("tags", []),
                })
            return items

        def _score_items(items: list, backend: dict) -> list:
            if backend["kind"] == "pioneer":
                try:
                    return _validate_movie_clips_text_pioneer_batch(items, backend["api_key"])
                except Exception as e:
                    print(
                        f"[clip_matcher] {backend['name']} failed ({e}); falling back to Vertex text validation for this batch",
                        flush=True,
                    )
                    try:
                        return _validate_movie_clips_text_batch(items, vertex_client, vertex_model)
                    except Exception as fallback_error:
                        print(
                            f"[clip_matcher] Vertex fallback also failed for {backend['name']} batch: {fallback_error}; skipping batch",
                            flush=True,
                        )
                        return []
            try:
                return _validate_movie_clips_text_batch(items, backend["client"], backend["model"])
            except Exception as e:
                print(
                    f"[clip_matcher] {backend['name']} text validation failed ({e}); skipping batch",
                    flush=True,
                )
                return []

        def _process_section(sec_idx: int, clips: list, section_text: str, backend: dict) -> tuple:
            existing = [c for c in clips if os.path.exists(c)]
            if not existing:
                return sec_idx, [], 0, 0

            all_scores = {}
            local_validated = 0
            local_cached = 0

            cache_scope = backend["name"]

            # Check cache + collect uncached for top-3
            to_validate_a = []
            for clip in existing[:3]:
                cached = _read_txt_cache(clip, section_text, cache_scope)
                if cached is not None:
                    all_scores[clip] = cached
                    local_cached += 1
                else:
                    to_validate_a.append(clip)

            # Validate top-3 uncached in one batch call
            if to_validate_a:
                scores_a = _score_items(_items_for(to_validate_a, section_text), backend)
                for clip, score in zip(to_validate_a, scores_a):
                    score = round(min(max(float(score), 0.0), 1.0), 4)
                    all_scores[clip] = score
                    _write_txt_cache(clip, section_text, score, cache_scope)
                    local_validated += 1

            best_so_far = max((all_scores.get(c, 0.0) for c in existing[:3]), default=0.0)

            # If no clip ≥ threshold in top-3, validate clips 4-5
            if best_so_far < MOVIE_THRESHOLD and len(existing) > 3:
                to_validate_b = []
                for clip in existing[3:5]:
                    cached = _read_txt_cache(clip, section_text, cache_scope)
                    if cached is not None:
                        all_scores[clip] = cached
                        local_cached += 1
                    else:
                        to_validate_b.append(clip)

                if to_validate_b:
                    scores_b = _score_items(_items_for(to_validate_b, section_text), backend)
                    for clip, score in zip(to_validate_b, scores_b):
                        score = round(min(max(float(score), 0.0), 1.0), 4)
                        all_scores[clip] = score
                        _write_txt_cache(clip, section_text, score, cache_scope)
                        local_validated += 1

            # Sort all scored clips by score desc
            scored = [(c, all_scores[c]) for c in existing[:5] if c in all_scores]
            scored.sort(key=lambda x: -x[1])
            return sec_idx, scored, local_validated, local_cached

        def _process_backend(backend: dict, backend_sections: list):
            print(f"[clip_matcher] Movie text worker {backend['name']}: {len(backend_sections)} sections", flush=True)
            for sec_idx, (clips, section_text) in backend_sections:
                try:
                    sec_idx, scored, v_count, c_count = _process_section(sec_idx, clips, section_text, backend)
                except Exception as e:
                    print(
                        f"[clip_matcher] Movie text section {sec_idx + 1} failed on {backend['name']}: {e}; skipping section",
                        flush=True,
                    )
                    scored, v_count, c_count = [], 0, 0
                with totals_lock:
                    results_by_sec[sec_idx] = scored
                    totals["validated"] += v_count
                    totals["cached"] += c_count
                    done = len(results_by_sec)
                if emit and done % 8 == 0:
                    emit("media", f"Movie library: text-validated sections {done}/{len(sections)}...")

        with ThreadPoolExecutor(max_workers=len(backends)) as pool:
            futures = [
                pool.submit(_process_backend, backend, backend_sections)
                for backend, backend_sections in zip(backends, assigned)
                if backend_sections
            ]
            for f in as_completed(futures):
                f.result()

        result = [results_by_sec.get(i, []) for i in range(len(sections))]
        backend_names = ", ".join(b["name"] for b in backends)
        print(
            f"[clip_matcher] Movie text validation ({backend_names}): "
            f"{totals['validated']} validated, {totals['cached']} cached",
            flush=True,
        )
        if emit:
            emit("media", f"Movie library: text-validated {totals['validated']} clips ({totals['cached']} cached) via {backend_names}")
        return result

    cached_scores: dict = {}
    to_validate:   list = []

    for sec_idx, (clips, section_text) in enumerate(zip(section_candidates, section_texts)):
        for clip_path in clips:
            if not os.path.exists(clip_path):
                continue
            cache_path = _validation_cache_path(clip_path, section_text)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, encoding="utf-8") as f:
                        cached_scores[(sec_idx, clip_path)] = float(json.load(f).get("score", 0.0))
                    continue
                except Exception:
                    pass
            to_validate.append((sec_idx, clip_path, section_text))

    print(f"[clip_matcher] Validation: {len(cached_scores)} cached, {len(to_validate)} to validate", flush=True)
    if emit:
        emit("media", f"Validating {len(to_validate)} pairs ({len(cached_scores)} cached)...")

    if not to_validate:
        return _build_validation_result(section_candidates, section_texts, cached_scores)

    def _extract(args):
        sec_idx, clip_path, section_text = args
        frames = [_frame_bytes(clip_path, r) for r in [0.0, 0.5, 1.0]]
        return {
            "sec_idx":      sec_idx,
            "clip_path":    clip_path,
            "section_text": section_text,
            "frames":       [f for f in frames if f],
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        items = list(pool.map(_extract, to_validate))
    items = [it for it in items if it["frames"]]

    batches       = [items[i:i + _VALIDATION_BATCH_SIZE] for i in range(0, len(items), _VALIDATION_BATCH_SIZE)]
    client, model = _gemini()
    lock          = threading.Lock()
    new_scores:   dict = {}
    done          = [0]
    worker_errors = []

    def _process_batch(batch):
        for attempt in range(3):
            try:
                scores = _validate_batch_chunk(batch, client, model)
                for item, score in zip(batch, scores):
                    cache_path = _validation_cache_path(item["clip_path"], item["section_text"])
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump({"score": score}, f)
                    except Exception:
                        pass
                    with lock:
                        new_scores[(item["sec_idx"], item["clip_path"])] = score
                        done[0] += 1
                        if emit and done[0] % 16 == 0:
                            emit("media", f"Validated {done[0]}/{len(items)}...")
                return
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower()
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[clip_matcher] Gemini auth/config error: {e}") from e
                if is_rate and attempt < 2:
                    _time.sleep(15 * (attempt + 1))
                else:
                    for item in batch:
                        s = validate_clip_for_section(item["clip_path"], item["section_text"])
                        with lock:
                            new_scores[(item["sec_idx"], item["clip_path"])] = s or 0.0
                            done[0] += 1
                    return

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[clip_matcher] Validation worker error: {e}", flush=True)
                worker_errors.append(e)

    if worker_errors:
        raise RuntimeError(f"[clip_matcher] Validation failed: {worker_errors[0]}")

    all_scores = {**cached_scores, **new_scores}
    print(f"[clip_matcher] Validation complete: {len(all_scores)} pairs scored", flush=True)
    return _build_validation_result(section_candidates, section_texts, all_scores)
