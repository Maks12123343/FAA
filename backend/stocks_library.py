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


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _extract_frame(video_path: str, position: float, out_path: str, timeout: int = 900) -> bool:
    duration = _get_duration(video_path)
    if duration <= 0:
        return False
    t = duration * position
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{t:.3f}", "-i", video_path,
         "-vframes", "1", "-vf", "scale=640:-2", "-q:v", "4", out_path],
        capture_output=True, timeout=timeout,
    )
    return os.path.exists(out_path) and os.path.getsize(out_path) > 500


def _analyze_with_gemini(video_path: str, category: str, frame_timeout: int = 120) -> dict:
    from google import genai
    from google.genai import types

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()

    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("gemini_model", "gemini-2.5-flash")

    positions = settings.get("clip_frames_positions", config.DEFAULT_SETTINGS["clip_frames_positions"])

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, pos in enumerate(positions):
            fp = os.path.join(tmp, f"f{i}.jpg")
            if _extract_frame(video_path, pos, fp, timeout=frame_timeout):
                with open(fp, "rb") as fh:
                    parts.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))

        if not parts:
            return {"description": "unknown", "tags": [], "category": category}

        prompt = (
            f'Frames from a stock video clip (category: "{category}").\n'
            f'Describe in 1-2 sentences what is visually shown. '
            f'Then list 8-12 tags (words/short phrases) for what topics this footage could illustrate.\n'
            f'JSON only: {{"description": "...", "tags": ["tag1", ...]}}'
        )
        parts.append(prompt)

        r = client.models.generate_content(model=model, contents=parts)
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            result = json.loads(m.group() if m else text)
            result["category"] = category
            return result
        except Exception:
            return {"description": r.text[:200], "tags": [], "category": category}


def _analysis_path(video_path: str) -> str:
    return video_path + ".analysis.json"


def _trim_inplace(video_path: str) -> None:
    """
    Trim video to stock_max_duration seconds in-place.
    Uses a temp file to avoid reading and writing the same file simultaneously.
    """
    import shutil
    settings = config.load_settings()
    max_dur  = float(settings.get("stock_max_duration", 6))

    dur = _get_duration(video_path)
    if dur <= max_dur:
        return

    tmp_path = video_path + ".tmp.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-i", video_path,
         "-t", str(max_dur),
         *config.get_video_encoder_args("ultrafast"), "-an",
         tmp_path],
        capture_output=True, timeout=120,
    )
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
        os.replace(tmp_path, video_path)
    elif os.path.exists(tmp_path):
        os.remove(tmp_path)


def _local_copy_for_analysis(video_path: str) -> str:
    """
    .mov files on Google Drive make ffprobe hang (lazy mount).
    Copy to /tmp as .mp4 first, return local path.
    Returns original path for .mp4 files already on local disk.
    """
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".mp4":
        return video_path  # already fine

    fname = os.path.splitext(os.path.basename(video_path))[0] + "_local.mp4"
    local_path = os.path.join(tempfile.gettempdir(), fname)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    print(f"[stocks] Converting {os.path.basename(video_path)} → local mp4...", flush=True)
    r = subprocess.run(
        [FFMPEG, "-y", "-i", video_path,
         *config.get_video_encoder_args("ultrafast"), "-an",
         "-t", "30",  # cap at 30s just in case
         local_path],
        capture_output=True, timeout=120,
    )
    if r.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    return video_path  # fallback to original if conversion failed


def analyze_stock_clip(video_path: str, category: str, frame_timeout: int = 900) -> dict:
    ap = _analysis_path(video_path)
    if os.path.exists(ap):
        with open(ap, encoding="utf-8") as f:
            return json.load(f)

    # .mov on Google Drive: copy locally first to avoid ffprobe hang
    work_path = _local_copy_for_analysis(video_path)

    # Trim in-place (works on local copy or original .mp4)
    _trim_inplace(work_path)

    print(f"[stocks] Analyzing: {os.path.basename(video_path)}", flush=True)
    analysis = _analyze_with_gemini(work_path, category, frame_timeout=frame_timeout)
    analysis["path"] = video_path  # always store original Drive path

    with open(ap, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # Clean up temp file
    if work_path != video_path and os.path.exists(work_path):
        try:
            os.unlink(work_path)
        except Exception:
            pass

    return analysis


def _collect_all_videos() -> list:
    stocks_dir = config.get_stocks_dir()
    all_videos = []
    for category in config.STOCK_CATEGORIES:
        cat_dir = os.path.join(stocks_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                all_videos.append((os.path.join(cat_dir, fname), category))
    return all_videos


def scan_and_analyze(emit=None) -> dict:
    """
    Scan all stock category folders, analyze any new clips.
    Retries failed clips with increasing timeout: 120 → 300 → 600 → 900s.
    Keeps retrying until all are done or all retries exhausted.
    """
    TIMEOUT_SCHEDULE = [900, 900, 900, 900]

    new_count      = 0
    existing_count = 0

    all_videos = _collect_all_videos()
    total = len(all_videos)

    pending = []
    for video_path, category in all_videos:
        if os.path.exists(_analysis_path(video_path)):
            existing_count += 1
        else:
            pending.append((video_path, category))

    attempt = 0
    while pending and attempt < len(TIMEOUT_SCHEDULE):
        timeout = TIMEOUT_SCHEDULE[attempt]
        if attempt > 0:
            msg = f"Retry {attempt}/{len(TIMEOUT_SCHEDULE)-1}: {len(pending)} clips left — timeout={timeout}s"
            print(f"[stocks] {msg}", flush=True)
            if emit:
                emit("stocks", msg)

        still_failed = []
        for video_path, category in pending:
            fname = os.path.basename(video_path)
            done_so_far = existing_count + new_count
            msg = f"[{done_so_far+1}/{total}] {fname} (timeout={timeout}s)"
            print(f"[stocks] {msg}", flush=True)
            if emit:
                emit("stocks", msg)
            try:
                analyze_stock_clip(video_path, category, frame_timeout=timeout)
                new_count += 1
            except Exception as e:
                still_failed.append((video_path, category))
                print(f"[stocks] FAIL {fname} attempt={attempt+1}: {e}", flush=True)
                if emit:
                    emit("stocks", f"FAIL {fname}: {e}")

        pending = still_failed
        attempt += 1

    error_count = len(pending)
    if pending:
        names = ", ".join(os.path.basename(p) for p, _ in pending[:5])
        more = f" (+{len(pending)-5} more)" if len(pending) > 5 else ""
        msg = f"Done. Could not analyze {error_count} clips after all retries: {names}{more}"
    else:
        msg = f"All done. Analyzed {new_count} new, {existing_count} already cached."
    print(f"[stocks] {msg}", flush=True)
    if emit:
        emit("stocks", msg)

    return {"analyzed_new": new_count, "already_done": existing_count, "errors": error_count}


def get_all_clips() -> list:
    """Return all analyzed stock clips as list of analysis dicts (with 'path' key)."""
    clips = []
    stocks_dir = config.get_stocks_dir()
    for category in config.STOCK_CATEGORIES:
        cat_dir = os.path.join(stocks_dir, category)
        if not os.path.exists(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            video_path = os.path.join(cat_dir, fname)
            ap = _analysis_path(video_path)
            if os.path.exists(ap):
                try:
                    with open(ap, encoding="utf-8") as f:
                        data = json.load(f)
                    data["path"] = video_path
                    clips.append(data)
                except Exception:
                    pass
    return clips


def pick_stock_clips(section_text: str, n: int = 3) -> list:
    """Pick N stock clips best matching section_text via tag/description overlap.
    Falls back to Pexels API if not enough local clips match."""
    import random
    all_clips = get_all_clips()

    section_lower = section_text.lower()

    def score(clip):
        s = 0
        for tag in clip.get("tags", []):
            if tag.lower() in section_lower:
                s += 2
        for word in clip.get("description", "").lower().split():
            if len(word) > 4 and word in section_lower:
                s += 1
        return s

    result = []
    seen = set()

    if all_clips:
        scored = sorted(all_clips, key=score, reverse=True)
        for clip in scored:
            p = clip["path"]
            if p not in seen and os.path.exists(p) and score(clip) > 0:
                seen.add(p)
                result.append(p)
                if len(result) >= n:
                    break

    # Fallback to Pexels if not enough matching clips
    if len(result) < n:
        pexels_clips = _pexels_fallback(section_text, n - len(result))
        result.extend(pexels_clips)

    # Still not enough — pad with random local clips
    if len(result) < n and all_clips:
        remaining = [c["path"] for c in all_clips if c["path"] not in seen and os.path.exists(c["path"])]
        random.shuffle(remaining)
        result.extend(remaining[:n - len(result)])

    return result


def _pexels_score_file(f: dict) -> tuple:
    w = f.get("width") or 0
    h = f.get("height") or 1
    ratio_penalty = abs(w / h - 16 / 9)
    size_penalty = abs(w - 1920)
    return (ratio_penalty, size_penalty)


def _pexels_download(url: str, dest: str) -> bool:
    """Atomic download with content-type check and 300MB limit."""
    import requests as _requests
    import threading
    tmp = f"{dest}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        with _requests.get(url, stream=True, timeout=60,
                           headers={"User-Agent": "ai-video-agent/1.0"}) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if not any(x in ctype for x in ("video", "image", "octet-stream")):
                print(f"[stocks] unexpected content-type '{ctype}': {url}", flush=True)
                return False
            total = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > 300 * 1024 * 1024:
                        print(f"[stocks] file >300MB, aborting: {url}", flush=True)
                        return False
                    f.write(chunk)
        if os.path.getsize(tmp) < 1000:
            return False
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"[stocks] download failed {url}: {e}", flush=True)
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _pexels_faststart(path: str) -> None:
    """Move moov atom to file start to validate and optimize the file."""
    tmp = path + ".fs.mp4"
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", path, "-c", "copy", "-movflags", "faststart", tmp],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and os.path.getsize(tmp) > 1000:
            os.replace(tmp, path)
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _pexels_fallback(section_text: str, n: int) -> list:
    """Search Pexels for stock clips, download, analyze with Gemini, save to general folder."""
    import requests as _requests

    settings = config.load_settings()
    api_keys = settings.get("pexels_api_keys", [])
    if not api_keys:
        single = settings.get("pexels_api_key", "")
        api_keys = [single] if single else []
    if not api_keys:
        return []

    query = " ".join(section_text.split()[:5])
    stocks_dir = config.get_stocks_dir()
    general_dir = os.path.join(stocks_dir, "general")
    os.makedirs(general_dir, exist_ok=True)

    videos = []
    for api_key in api_keys:
        try:
            r = _requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": api_key},
                params={"query": query, "per_page": n * 2, "orientation": "landscape"},
                timeout=15,
            )
            if r.status_code == 200:
                videos = r.json().get("videos", [])
                break
            elif r.status_code == 429:
                print(f"[stocks] Pexels key rate-limited, trying next...", flush=True)
                continue
            else:
                print(f"[stocks] Pexels API error: {r.status_code}", flush=True)
                continue
        except Exception as e:
            print(f"[stocks] Pexels request failed: {e}", flush=True)
            continue

    if not videos:
        return []

    result = []
    for video in videos:
        if len(result) >= n:
            break

        duration = video.get("duration") or 0
        if duration < 3 or duration > 45:
            continue

        vid_id = video.get("id", "")
        out_path = os.path.join(general_dir, f"pexels_{vid_id}.mp4")

        if os.path.exists(out_path):
            result.append(out_path)
            continue

        # Pick best landscape HD file closest to 16:9
        video_files = video.get("video_files", [])
        valid_files = []
        for vf in video_files:
            w = vf.get("width") or 0
            h = vf.get("height") or 0
            ft = (vf.get("file_type") or "").lower()
            if w >= 1280 and h >= 720 and w > h and "mp4" in ft:
                valid_files.append(vf)

        if not valid_files:
            continue

        best = min(valid_files, key=_pexels_score_file)
        dl_url = best.get("link")
        if not dl_url:
            continue

        print(f"[stocks] Pexels downloading: {vid_id}", flush=True)
        if not _pexels_download(dl_url, out_path):
            continue

        # Validate and optimize file structure
        _pexels_faststart(out_path)

        # Trim to max duration
        _trim_inplace(out_path)

        # Analyze with Gemini (saves .analysis.json)
        try:
            analyze_stock_clip(out_path, "general")
        except Exception as e:
            print(f"[stocks] Pexels analysis failed: {e}", flush=True)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
            result.append(out_path)

    if result:
        print(f"[stocks] Pexels fallback: got {len(result)} clips for '{query}'", flush=True)
    return result


STOCK_SCORE_THRESHOLD = 0.85


def _is_gemini_auth_error(err) -> bool:
    text = str(err).lower()
    return any(x in text for x in (
        "401", "403", "permission", "credentials",
        "unauthenticated", "unauthorized", "invalid_argument",
    ))


def validate_stock_for_section(clip_path: str, section_text: str) -> float:
    """
    Validate a stock clip against a script section using Gemini.
    Returns score 0.0–1.0. Result cached to disk.
    Retries up to 3 times on rate-limit errors (429).
    """
    import hashlib
    import time as _time
    from google import genai
    from google.genai import types

    h = hashlib.md5(section_text[:300].encode()).hexdigest()[:12]
    cache_path = clip_path + f".stockval_{h}.json"

    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return float(json.load(f).get("score", 0.0))
        except Exception:
            pass

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("gemini_model", "gemini-2.5-flash")

    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, pos in enumerate([0.0, 0.5, 1.0]):
            fp = os.path.join(tmp, f"f{i}.jpg")
            if _extract_frame(clip_path, pos, fp):
                with open(fp, "rb") as fh:
                    parts.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))

    if not parts:
        return 0.0

    prompt = (
        f'These are 3 frames from a stock video clip.\n'
        f'Script section this clip should illustrate: "{section_text[:300]}"\n'
        f'Rate how well this clip visually matches the script section.\n'
        f'JSON only: {{"score": 0.0}} where 0.0=no match, 1.0=perfect match.'
    )
    parts.append(prompt)

    score = 0.0
    got_response = False
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=model, contents=parts)
            text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group() if m else text)
            score = float(data.get("score", 0.0))
            got_response = True
            break
        except Exception as e:
            if _is_gemini_auth_error(e):
                raise RuntimeError(f"[stocks] Gemini auth/config error: {e}") from e
            if "429" in str(e) or "quota" in str(e).lower() or "resource" in str(e).lower():
                wait = 5 * (attempt + 1)
                print(f"[stocks] Rate limit, waiting {wait}s (attempt {attempt+1}/3)...", flush=True)
                _time.sleep(wait)
            else:
                break

    if got_response:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"score": score}, f)
        except Exception:
            pass

    return score


def get_stats() -> dict:
    stats = {}
    total = 0
    stocks_dir = config.get_stocks_dir()
    for category in config.STOCK_CATEGORIES:
        cat_dir = os.path.join(stocks_dir, category)
        if not os.path.exists(cat_dir):
            stats[category] = {"total": 0, "analyzed": 0}
            continue
        videos = [f for f in os.listdir(cat_dir)
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        analyzed = sum(1 for f in videos
                       if os.path.exists(os.path.join(cat_dir, f + ".analysis.json")))
        stats[category] = {"total": len(videos), "analyzed": analyzed}
        total += len(videos)
    stats["_total"] = total
    return stats


# ── Batch stock validation (8 clips per Gemini call, 3 parallel workers) ──────

_STOCK_BATCH_SIZE = 8


def _extract_frame_bytes_list(clip_path: str, positions=None) -> list:
    """Extract JPEG frame bytes from a clip. Returns list of bytes objects."""
    if positions is None:
        positions = [0.0, 0.5, 1.0]
    dur = _get_duration(clip_path)
    if dur <= 0:
        return []
    result = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, pos in enumerate(positions):
            ts = max(0.01, dur * max(0.0, min(1.0, pos)))
            fp = os.path.join(tmp, f"f{i}.jpg")
            r = subprocess.run(
                [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", clip_path,
                 "-vframes", "1", "-vf", "scale=640:-2", "-q:v", "4", fp],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0 and os.path.exists(fp) and os.path.getsize(fp) > 500:
                with open(fp, "rb") as f:
                    result.append(f.read())
    return result


def _write_stockval_cache(cache_path: str, score: float) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"score": round(score, 4)}, f)
    except Exception:
        pass


def _official_stock_batch_chunk(items: list, settings: dict) -> list:
    """
    Send up to _STOCK_BATCH_SIZE stock clips to Vertex AI in ONE Gemini call.
    items: list of {"clip_path": str, "section_text": str, "frames": list[bytes]}
    Returns list of float scores in same order as items.
    """
    from google import genai
    from google.genai import types
    import re as _re

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("gemini_model", "gemini-2.5-flash")

    contents = [f"Evaluate {len(items)} stock video clips for relevance to their script sections.\n"]
    for idx, item in enumerate(items):
        contents.append(f'Clip {idx + 1} — script: "{item["section_text"][:200]}"')
        for fb in item["frames"]:
            contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    contents.append(
        f'\nReply ONLY with a JSON array of exactly {len(items)} objects in order:\n'
        '[{"score": 0.0}, ...]  score: 0.0=no match, 1.0=perfect match. Nothing else.'
    )

    r = client.models.generate_content(model=model, contents=contents)
    text = _re.sub(r"^```(?:json)?\s*", "", r.text.strip())
    text = _re.sub(r"\s*```$", "", text)
    m = _re.search(r"\[.*\]", text, _re.DOTALL)
    raw = json.loads(m.group() if m else text)

    scores = []
    for i in range(len(items)):
        entry = raw[i] if i < len(raw) else {"score": 0.0}
        s = float(entry.get("score", 0.0)) if isinstance(entry, dict) else float(entry)
        scores.append(max(0.0, min(1.0, s)))
    return scores


def prewarm_stock_validation(whisper_segments: list, settings: dict, emit=None) -> None:
    """
    Pre-compute all stock validation scores before the assembly loop runs.
    Uses batch Gemini calls (8 clips per call) with 3 parallel workers.
    Writes results to .stockval_<hash>.json cache files.
    validate_stock_for_section() reads from these automatically.
    """
    import hashlib
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    uncached = []
    seen = set()

    for seg in whisper_segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        candidates = pick_stock_clips(seg_text, n=5)
        for clip_path in candidates:
            h = hashlib.md5(seg_text[:300].encode()).hexdigest()[:12]
            cache_path = clip_path + f".stockval_{h}.json"
            key = (clip_path, seg_text[:300])
            if key in seen:
                continue
            seen.add(key)
            if os.path.exists(cache_path):
                continue
            uncached.append({
                "clip_path":    clip_path,
                "section_text": seg_text,
                "cache_path":   cache_path,
            })

    total_pairs = len(seen)
    if not uncached:
        print(f"[stocks] Stock validation: all {total_pairs} pairs cached", flush=True)
        return

    print(
        f"[stocks] Pre-warming stock validation: {len(uncached)} uncached "
        f"({total_pairs - len(uncached)} already cached)...",
        flush=True,
    )
    if emit:
        emit("media", f"Pre-validating {len(uncached)} stock clips in batch mode...")

    # Extract frames in parallel
    def _extract(item):
        frames = _extract_frame_bytes_list(item["clip_path"])
        return {**item, "frames": frames}

    with ThreadPoolExecutor(max_workers=4) as pool:
        items_with_frames = list(pool.map(_extract, uncached))
    items_with_frames = [it for it in items_with_frames if it.get("frames")]

    batches = [
        items_with_frames[i:i + _STOCK_BATCH_SIZE]
        for i in range(0, len(items_with_frames), _STOCK_BATCH_SIZE)
    ]

    done = [0]
    lock = threading.Lock()
    worker_errors = []

    def _process_batch(batch):
        for attempt in range(3):
            try:
                scores = _official_stock_batch_chunk(batch, settings)
                for item, score in zip(batch, scores):
                    _write_stockval_cache(item["cache_path"], score)
                with lock:
                    done[0] += len(batch)
                    n = done[0]
                print(f"[stocks] Validated {n}/{len(items_with_frames)}", flush=True)
                return
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower()
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[stocks] Gemini auth/config error: {e}") from e
                if is_rate and attempt < 2:
                    wait = 15 * (attempt + 1)
                    print(f"[stocks] Rate limit, retry in {wait}s...", flush=True)
                    _time.sleep(wait)
                else:
                    print(f"[stocks] Batch error (attempt {attempt + 1}): {e}", flush=True)
                    break

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[stocks] Worker error: {e}", flush=True)
                worker_errors.append(e)

    if worker_errors:
        raise RuntimeError(f"[stocks] Stock validation failed: {worker_errors[0]}")

    print(f"[stocks] Stock validation pre-warm done.", flush=True)
    if emit:
        emit("media", "Stock validation complete.")
