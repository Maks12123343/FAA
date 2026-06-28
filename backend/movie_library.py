"""
Movie Library — індексація фільмів та підбір кліпів для cartoon-psychology ніші.
"""

import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE

SCENE_THRESHOLD    = 0.35
CLIP_MIN           = 2.0
CLIP_MAX           = 5.0
BATCH_SIZE         = 1      # ОДИН кліп за один запит — щоб модель не плутала кадри між кліпами

# Round-robin counters for API key rotation across parallel workers.
# Each call to _next_*_key() returns the next key in sequence, so concurrent
# batches spread load evenly instead of hammering key #0.
_gc_key_counter   = 0
_gc_key_lock      = threading.Lock()
_pio_key_counter  = 0
_pio_key_lock     = threading.Lock()


def _next_gigacoder_key(keys: list) -> tuple:
    """Pick a starting key index round-robin so parallel workers spread load."""
    global _gc_key_counter
    if not keys:
        return 0, []
    with _gc_key_lock:
        start = _gc_key_counter % len(keys)
        _gc_key_counter += 1
    # Build a rotated key list: starting key first, then the rest as fallbacks
    rotated = keys[start:] + keys[:start]
    return start, rotated


def _next_pioneer_key(keys: list) -> tuple:
    global _pio_key_counter
    if not keys:
        return 0, []
    with _pio_key_lock:
        start = _pio_key_counter % len(keys)
        _pio_key_counter += 1
    rotated = keys[start:] + keys[:start]
    return start, rotated


VALIDATION_THRESHOLD = 0.85  # Gemini validation score (як в FAA)


def _is_gemini_auth_error(err) -> bool:
    text = str(err).lower()
    return any(x in text for x in (
        "401", "403", "permission", "credentials",
        "unauthenticated", "unauthorized", "invalid_argument",
    ))


# ── Шляхи ─────────────────────────────────────────────────────────────────────

def _movies_dir() -> str:
    return config.get_movies_dir()

def _movie_dir(movie_name: str) -> str:
    return os.path.join(_movies_dir(), movie_name)

def _clips_dir(movie_name: str) -> str:
    return os.path.join(_movie_dir(movie_name), "clips")

def _index_path(movie_name: str) -> str:
    return os.path.join(_movie_dir(movie_name), "index.json")


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def _get_duration(path: str) -> float:
    # Windows + eventlet + capture_output deadlocks. Use DEVNULL pipes — the
    # JSON output is parsed via stdout, but we still avoid the deadlock by
    # NOT using capture_output (which spawns reader threads that don't play
    # well with monkey-patched threading).
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=120,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _cut_clip(src: str, out: str, start: float, duration: float):
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}",
         "-c", "copy", "-an", out],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
    )


def _detect_scene_timestamps(video_path: str, total_dur: float, emit=None) -> list:
    """
    Run FFmpeg scene detection while streaming progress from stderr.
    FFmpeg outputs lines like 'time=00:23:45.67' as it processes — we parse
    those to show real-time progress through the file. No timeout: process
    runs as long as it needs to.
    """
    proc = subprocess.Popen(
        [FFMPEG, "-threads", "0", "-i", video_path,
         "-vf", f"select=gt(scene\\,{SCENE_THRESHOLD}),showinfo",
         "-vsync", "vfr", "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    timestamps = [0.0]
    last_emit  = 0.0
    progress_re = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")
    pts_re      = re.compile(r"pts_time:(\d+\.?\d*)")
    start_time  = time.time()

    if emit:
        emit("movie", f"Scene detection started (file is {total_dur/60:.1f} min)")

    try:
        for line in proc.stderr:
            # Collect scene change timestamps from showinfo output
            if "showinfo" in line:
                m = pts_re.search(line)
                if m:
                    t = float(m.group(1))
                    if t > 0.1:
                        timestamps.append(t)

            # Stream progress from FFmpeg's time= reports
            m = progress_re.search(line)
            if m:
                hh, mm, ss = m.groups()
                cur = int(hh) * 3600 + int(mm) * 60 + float(ss)
                if cur - last_emit >= 30 or cur >= total_dur - 1:
                    last_emit = cur
                    pct      = min(100, int(cur / max(1, total_dur) * 100))
                    elapsed  = time.time() - start_time
                    msg = (
                        f"Scene detection: {cur/60:.1f}/{total_dur/60:.1f} min "
                        f"({pct}%) — found {len(timestamps)-1} scenes, "
                        f"elapsed {elapsed/60:.1f} min"
                    )
                    print(f"[movie_library] {msg}", flush=True)
                    if emit:
                        emit("movie", msg)

        rc = proc.wait()
        if rc != 0:
            print(f"[movie_library] FFmpeg scene detect exited with code {rc}", flush=True)

    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"Scene detection failed: {e}") from e

    timestamps.append(total_dur)
    result = sorted(set(timestamps))
    done_msg = f"Scene detection done: {len(result)-2} scene cuts in {(time.time()-start_time)/60:.1f} min"
    print(f"[movie_library] {done_msg}", flush=True)
    if emit:
        emit("movie", done_msg)
    return result


def _cut_by_scenes(src_path: str, out_dir: str, movie_id: str,
                   total_dur: float, emit=None) -> list:
    """
    Cut the source video into clips based on detected scene changes.
    """
    scene_times = _detect_scene_timestamps(src_path, total_dur, emit=emit)

    clips      = []
    idx        = 0
    total_segs = len(scene_times) - 1
    last_emit  = time.time()
    cut_start  = time.time()
    print(f"[movie_library] Starting clip cutting: {total_segs} segments to process", flush=True)

    for i in range(total_segs):
        scene_start = scene_times[i]
        scene_end   = scene_times[i + 1]
        scene_dur   = scene_end - scene_start

        if scene_dur < CLIP_MIN:
            continue

        if scene_dur <= CLIP_MAX:
            out = os.path.join(out_dir, f"{movie_id}_{idx:04d}.mp4")
            _cut_clip(src_path, out, scene_start, scene_dur)
            if os.path.exists(out) and os.path.getsize(out) > 5000:
                clips.append({
                    "id":    f"{movie_id}_{idx:04d}",
                    "file":  out,
                    "start": round(scene_start, 2),
                    "end":   round(scene_end, 2),
                })
                idx += 1
        else:
            t = scene_start
            while t + CLIP_MIN <= scene_end:
                chunk = min(CLIP_MAX, scene_end - t)
                if chunk < CLIP_MIN:
                    break
                out = os.path.join(out_dir, f"{movie_id}_{idx:04d}.mp4")
                _cut_clip(src_path, out, t, chunk)
                if os.path.exists(out) and os.path.getsize(out) > 5000:
                    clips.append({
                        "id":    f"{movie_id}_{idx:04d}",
                        "file":  out,
                        "start": round(t, 2),
                        "end":   round(t + chunk, 2),
                    })
                    idx += 1
                t += chunk

        # Emit progress at most once every 2 seconds (and on every 25th segment)
        now = time.time()
        if now - last_emit >= 2.0 or (i + 1) % 25 == 0:
            last_emit = now
            pct       = int((i + 1) / max(1, total_segs) * 100)
            elapsed   = now - cut_start
            msg       = f"Cutting: {idx} clips so far ({i+1}/{total_segs} segments, {pct}%, elapsed {elapsed:.0f}s)..."
            print(f"[movie_library] {msg}", flush=True)
            if emit:
                emit("movie", msg)

    final_msg = f"Cutting done: {idx} clips in {(time.time()-cut_start)/60:.1f} min"
    print(f"[movie_library] {final_msg}", flush=True)
    if emit:
        emit("movie", final_msg)

    return clips


# ── Gemini helpers ─────────────────────────────────────────────────────────────

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


def _frame_bytes(clip_path: str, ratio: float) -> bytes:
    """
    Extract a single JPEG frame from a clip at relative position `ratio` (0-1).
    Caches the frame to disk next to the clip — second run reads from cache
    instead of re-running FFmpeg.

    Windows + eventlet pitfalls avoided here:
      1. NO capture_output=True — that spawns Popen reader threads which
         deadlock under eventlet's monkey-patched threading on Windows.
      2. NO NamedTemporaryFile keeping a handle open while ffmpeg writes —
         we use mkstemp + os.close before launching the subprocess so Windows
         doesn't lock the path.
    """
    # Map ratio → label for the cache filename
    if ratio <= 0.01:
        label = "start"
    elif ratio >= 0.99:
        label = "end"
    else:
        label = "mid"
    cache_path = clip_path + f".{label}.jpg"

    # Reuse cached frame if it exists and is non-empty
    if os.path.exists(cache_path):
        try:
            if os.path.getsize(cache_path) > 0:
                with open(cache_path, "rb") as f:
                    return f.read()
        except Exception:
            pass

    dur = _get_duration(clip_path)
    ts  = max(0.01, dur * max(0.0, min(1.0, ratio)))

    # Write directly to the cache_path so we keep the JPEG for next time
    data = b""
    try:
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", clip_path,
             "-vframes", "1", "-vf", "scale=1024:-2", "-q:v", "3", cache_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "rb") as f:
                data = f.read()
    except Exception as e:
        print(f"[movie_library] _frame_bytes error: {e}", flush=True)

    return data


_BATCH_PROMPT = """\
You are looking at 3 frames extracted from a single short video clip (start, middle, end).

CRITICAL RULES — read carefully:
1. Describe ONLY what is actually visible in these 3 frames. Do NOT invent characters, locations or actions.
2. If you cannot clearly identify a character — leave the characters list EMPTY. Do NOT guess.
3. If the frames are too dark, blurry, or contain only text/credits/logos — say so honestly and set is_blurry: true.
4. If all 3 frames look almost identical — set is_static: true.
5. Do NOT use the movie title to guess what is happening. Look at the actual pixels.

Return a JSON object with these fields:
- characters: list of character names you can clearly identify (or [] if uncertain)
- emotion: one of: joy, sadness, fear, anger, determination, vulnerability, shame, guilt, pride, neutral
- scene_type: one of: training, fight, emotional_dialogue, rejection, acceptance, flashback, celebration, isolation, comedy, action, quiet_moment, transformation, credits, title_card, landscape, transition
- themes: 2-4 from: [growth, trauma, impostor_syndrome, false_self, identity, rejection, acceptance, vulnerability, shame, fear, determination, healing, connection, isolation, anger, betrayal, grief, love, trust]
- description: 1-2 sentences describing EXACTLY what you see (objects, lighting, composition). Be literal.
- tags: 5-10 concrete visual tags based on what you actually see (objects, environment, mood)
- is_blurry: true if the frames are out of focus, too dark, or contain only text/logos
- is_static: true if all 3 frames look nearly identical

Reply with a JSON array containing EXACTLY ONE object. No markdown, no extra commentary."""


# Single-clip prompt — much more accurate than batch because the model can't
# mix up which frames belong to which clip when there's only one clip.
_SINGLE_PROMPT = """\
You are looking at exactly 3 frames from ONE short video clip from the movie "{movie_name}".
The frames are shown in order: start, middle, end of the clip.

Analyze ONLY what you actually see in these 3 frames. Do NOT guess based on the movie name.
If a frame is dark, empty, or has no characters — say so honestly. Do not invent characters.

Return a JSON object with:
- characters: list of character names ACTUALLY visible (empty list [] if none / unclear / not characters from the movie)
- emotion: one of: joy, sadness, fear, anger, determination, vulnerability, shame, guilt, pride, neutral
- scene_type: one of: training, fight, emotional_dialogue, rejection, acceptance, flashback, celebration, isolation, comedy, action, quiet_moment, transformation
- themes: 2-4 from: [growth, trauma, impostor_syndrome, false_self, identity, rejection, acceptance, vulnerability, shame, fear, determination, healing, connection, isolation, anger, betrayal, grief, love, trust]
- description: 1-2 sentences describing ONLY what is visually shown in these 3 frames
- tags: 6-12 specific tags for what is shown (location, objects, actions, mood)
- is_blurry: true if most frames are out of focus or heavily motion-blurred
- is_static: true if all 3 frames look nearly identical (frozen/no motion)

Reply ONLY with a single JSON object, no markdown, no commentary."""


def _analyze_batch(items: list, movie_name: str, client, model: str) -> list:
    from google.genai import types

    contents = []
    for i, item in enumerate(items):
        contents.append(f"CLIP {i + 1}:")
        for fb in item["frames"]:
            contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    contents.append(_BATCH_PROMPT)

    r = client.models.generate_content(model=model, contents=contents)
    text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
    text = re.sub(r"\s*```$", "", text)
    m    = re.search(r"\[.*\]", text, re.DOTALL)
    raw  = json.loads(m.group() if m else text)

    results = []
    for i, item in enumerate(items):
        a = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        a.setdefault("characters", [])
        a.setdefault("emotion", "neutral")
        a.setdefault("scene_type", "quiet_moment")
        a.setdefault("themes", [])
        a.setdefault("description", "")
        a.setdefault("tags", [])
        a.setdefault("is_blurry", False)
        a.setdefault("is_static", False)
        results.append(a)
    return results


def _analyze_single_gigacoder(item: dict, movie_name: str) -> dict:
    """
    Analyze ONE clip via GigaCoder. Much more accurate than batching because
    the model can't confuse which frames belong to which clip — there's only one.
    Takes ~2-4s per clip but eliminates hallucinated descriptions.
    """
    import base64
    import urllib.request

    settings = config.load_settings()
    gc_keys  = settings.get("gigacoder_api_keys", [])
    gc_url   = settings.get("gigacoder_api_url", "https://www.gigacoder.org/api/v1/chat/completions")
    gc_model = settings.get("gigacoder_model", "gpt-5.4-mini")
    if not gc_keys:
        raise RuntimeError("No gigacoder_api_keys configured")

    # Build content: 3 labeled frames + analysis prompt
    content_parts = [{"type": "text", "text": "Frame 1 (start of clip):"}]
    if len(item["frames"]) >= 1:
        b64 = base64.b64encode(item["frames"][0]).decode("ascii")
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if len(item["frames"]) >= 2:
        content_parts.append({"type": "text", "text": "Frame 2 (middle of clip):"})
        b64 = base64.b64encode(item["frames"][1]).decode("ascii")
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if len(item["frames"]) >= 3:
        content_parts.append({"type": "text", "text": "Frame 3 (end of clip):"})
        b64 = base64.b64encode(item["frames"][2]).decode("ascii")
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content_parts.append({"type": "text", "text": _SINGLE_PROMPT.format(movie_name=movie_name)})

    payload = json.dumps({
        "model": gc_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise visual analyzer. Describe ONLY what is visible in the frames. "
                    "Never invent characters or scenes that aren't shown. "
                    "If frames are dark, empty, or unclear, say so honestly. "
                    "Reply with valid JSON only — no markdown, no commentary."
                ),
            },
            {"role": "user", "content": content_parts},
        ],
        "max_tokens": 1024,
    }).encode("utf-8")

    last_err = None
    start_idx, rotated_keys = _next_gigacoder_key(gc_keys)
    for key in rotated_keys:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    gc_url, data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?\s*", "", text.strip())
                text = re.sub(r"\s*```$", "", text)
                m = re.search(r"\{.*\}", text, re.DOTALL)
                a = json.loads(m.group() if m else text)
                a.setdefault("characters", [])
                a.setdefault("emotion", "neutral")
                a.setdefault("scene_type", "quiet_moment")
                a.setdefault("themes", [])
                a.setdefault("description", "")
                a.setdefault("tags", [])
                a.setdefault("is_blurry", False)
                a.setdefault("is_static", False)
                return a
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Retry on network/DNS errors with backoff (1s, 3s, 7s)
                is_network = any(s in err_str for s in (
                    "lookup timed out", "11002", "timed out", "connection",
                    "temporarily unavailable", "name or service",
                ))
                if is_network and attempt < 2:
                    time.sleep([1, 3, 7][attempt])
                    continue
                # Other errors — break inner loop, try next key
                break

    raise RuntimeError(f"GigaCoder single-clip analyze failed: {last_err}")


def _analyze_batch_gigacoder(items: list, movie_name: str) -> list:
    """
    Analyze a batch of clips by calling _analyze_single_gigacoder on each one.
    True batching (multiple clips in one request) caused severe hallucinations
    because the model mixed up frames between clips. Single-clip analysis is
    slower but accurate — and since we now process batches sequentially, total
    throughput is the same.
    """
    results = []
    for item in items:
        try:
            a = _analyze_single_gigacoder(item, movie_name)
        except Exception as e:
            print(f"[movie_library] Single-clip analysis failed for "
                  f"{item['clip'].get('id', '?')}: {e}", flush=True)
            raise
        results.append(a)
    return results


def _analyze_batch_pioneer(items: list, movie_name: str) -> list:
    """Fallback: analyze clips via Pioneer API (multimodal, base64 frames)."""
    import base64
    import urllib.request

    settings = config.load_settings()
    api_keys = settings.get("pioneer_api_keys", [])
    api_url = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")
    api_model = settings.get("pioneer_model", "gemini-3.5-flash")
    if not api_keys:
        raise RuntimeError("No pioneer_api_keys configured")

    content_parts = []
    for i, item in enumerate(items):
        content_parts.append({"type": "text", "text": f"CLIP {i + 1}:"})
        for fb in item["frames"]:
            b64 = base64.b64encode(fb).decode("ascii")
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content_parts.append({"type": "text", "text": _BATCH_PROMPT})

    payload = json.dumps({
        "model": api_model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 4096,
    }).encode("utf-8")

    last_err = None
    start_idx, rotated_keys = _next_pioneer_key(api_keys)
    for offset, key in enumerate(rotated_keys):
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    api_url, data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?\s*", "", text.strip())
                text = re.sub(r"\s*```$", "", text)
                m = re.search(r"\[.*\]", text, re.DOTALL)
                raw = json.loads(m.group() if m else text)
                results = []
                for i, item in enumerate(items):
                    a = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
                    a.setdefault("characters", [])
                    a.setdefault("emotion", "neutral")
                    a.setdefault("scene_type", "quiet_moment")
                    a.setdefault("themes", [])
                    a.setdefault("description", "")
                    a.setdefault("tags", [])
                    a.setdefault("is_blurry", False)
                    a.setdefault("is_static", False)
                    results.append(a)
                return results
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Retry on network/DNS errors with backoff
                is_network = any(s in err_str for s in (
                    "lookup timed out", "11002", "timed out", "connection",
                    "temporarily unavailable", "name or service",
                ))
                if is_network and attempt < 2:
                    time.sleep([1, 3, 7][attempt])
                    continue
                break

    raise RuntimeError(f"Pioneer analyze failed: {last_err}")


def _analysis_cached(clip_path: str) -> dict | None:
    ap = clip_path + ".analysis.json"
    if not os.path.exists(ap):
        return None
    try:
        st = os.stat(clip_path)
        with open(ap, encoding="utf-8") as f:
            cached = json.load(f)
        if (cached.get("_size") == st.st_size and
                cached.get("_mtime") == round(st.st_mtime, 2)):
            return cached
    except Exception:
        pass
    return None


def _save_analysis(clip_path: str, analysis: dict):
    try:
        st = os.stat(clip_path)
        analysis["_size"]  = st.st_size
        analysis["_mtime"] = round(st.st_mtime, 2)
    except Exception:
        pass
    ap = clip_path + ".analysis.json"
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)


def _analyze_all_clips(clips: list, movie_name: str, emit=None) -> list:
    # eventlet's monkey-patched threading + subprocess works correctly with
    # eventlet.GreenPool. Plain ThreadPoolExecutor hangs under eventlet.
    try:
        import eventlet
        _USE_GREEN_POOL = True
    except ImportError:
        _USE_GREEN_POOL = False
    from concurrent.futures import ThreadPoolExecutor

    to_analyze     = []
    cached_results = []

    for c in clips:
        cached = _analysis_cached(c["file"])
        if cached:
            cached["id"]   = c["id"]
            cached["file"] = c["file"]
            cached_results.append(cached)
        else:
            to_analyze.append(c)

    if not to_analyze:
        return cached_results

    print(f"[movie_library] Analyzing {len(to_analyze)} clips "
          f"({len(cached_results)} cached)...", flush=True)

    # ── Frame extraction with progress ─────────────────────────────────────
    # Sequential extraction — no thread/green pools.
    # eventlet's monkey-patching makes any pool unreliable with subprocess on
    # Windows; sequential is dead-simple and never hangs. ~0.1-0.3s per clip
    # so even 1500 clips finish in 5-8 min, with live per-clip progress.
    frames_start = time.time()
    total_clips  = len(to_analyze)

    start_msg = f"Starting frame extraction for {total_clips} clips (sequential)..."
    print(f"[movie_library] {start_msg}", flush=True)
    if emit:
        emit("movie", start_msg)

    items = []
    last_log = time.time()

    for d, clip in enumerate(to_analyze, start=1):
        frames = []
        try:
            for ratio in [0.0, 0.5, 1.0]:
                fb = _frame_bytes(clip["file"], ratio)
                if fb:
                    frames.append(fb)
        except Exception as e:
            print(f"[movie_library] Frame extract error for {clip.get('id', '?')}: {e}", flush=True)

        if frames:
            items.append({"clip": clip, "frames": frames})

        # Log every clip for first 5, then every 10 clips, OR every 5 seconds
        now = time.time()
        if d <= 5 or d % 10 == 0 or d == total_clips or (now - last_log) >= 5:
            last_log = now
            elapsed  = now - frames_start
            rate     = d / max(0.1, elapsed)
            eta      = (total_clips - d) / max(0.1, rate)
            pct      = int(d / total_clips * 100)
            msg = (
                f"Extracting frames: {d}/{total_clips} "
                f"({pct}%, {rate:.1f}/s, ~{eta/60:.1f}min left)"
            )
            print(f"[movie_library] {msg}", flush=True)
            if emit:
                emit("movie", msg)
            # Yield to eventlet so SocketIO can flush messages to UI
            try:
                import eventlet as _ev
                _ev.sleep(0)
            except Exception:
                pass

    extract_msg = f"Frame extraction done: {len(items)} clips ready in {(time.time()-frames_start)/60:.1f} min"
    print(f"[movie_library] {extract_msg}", flush=True)
    if emit:
        emit("movie", extract_msg)

    batches     = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    done        = [0]
    lock        = threading.Lock()
    results     = []
    worker_errors = []

    def _process_batch(batch):
        for attempt in range(3):
            try:
                # Pioneer is the PRIMARY analyzer (more accurate descriptions,
                # no Cloudflare delays). GigaCoder is the fallback.
                analyses = _analyze_batch_pioneer(batch, movie_name)
                for item, analysis in zip(batch, analyses):
                    clip = item["clip"]
                    analysis["id"]   = clip["id"]
                    analysis["file"] = clip["file"]
                    _save_analysis(clip["file"], analysis)
                    with lock:
                        done[0] += 1
                        results.append(analysis)
                    if emit:
                        emit("movie", f"Analyzed {done[0]}/{len(items)} clips...")
                return
            except Exception as e:
                err = str(e).lower()
                is_rate = "429" in str(e) or "quota" in err or "resource_exhausted" in err
                if is_rate and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                else:
                    # Fallback to GigaCoder
                    print(f"[movie_library] Pioneer failed, trying GigaCoder: {e}", flush=True)
                    try:
                        analyses = _analyze_batch_gigacoder(batch, movie_name)
                        for item, analysis in zip(batch, analyses):
                            clip = item["clip"]
                            analysis["id"]   = clip["id"]
                            analysis["file"] = clip["file"]
                            _save_analysis(clip["file"], analysis)
                            with lock:
                                done[0] += 1
                                results.append(analysis)
                            if emit:
                                emit("movie", f"Analyzed {done[0]}/{len(items)} clips...")
                        return
                    except Exception as fb_e:
                        print(f"[movie_library] GigaCoder fallback also failed: {fb_e}", flush=True)
                    for item in batch:
                        clip = item["clip"]
                        fallback = {
                            "characters": [], "emotion": "neutral",
                            "scene_type": "quiet_moment", "themes": [],
                            "description": "unknown", "tags": [],
                            "is_blurry": False, "is_static": False,
                            "id": clip["id"], "file": clip["file"],
                        }
                        _save_analysis(clip["file"], fallback)
                        with lock:
                            done[0] += 1
                            results.append(fallback)
                    return

    # Process analysis batches with bounded parallelism for the HTTP calls.
    # Frame extraction was already sequential. The remaining work is HTTP-only
    # (Pioneer/GigaCoder API), which works fine under eventlet's monkey-patched
    # socket. We run up to PARALLEL_API requests at a time so all 4 Pioneer keys
    # are exercised simultaneously instead of one-by-one.
    PARALLEL_API = 4
    api_start    = time.time()

    try:
        import eventlet as _ev
        _USE_EVENTLET = True
    except Exception:
        _USE_EVENTLET = False

    if _USE_EVENTLET and len(batches) > 1:
        sem = _ev.semaphore.Semaphore(PARALLEL_API)

        def _wrapped(batch, b_idx):
            with sem:
                try:
                    _process_batch(batch)
                except Exception as e:
                    print(f"[movie_library] Worker error on batch {b_idx}: {e}", flush=True)
                    worker_errors.append(e)

        threads = [_ev.spawn(_wrapped, b, i) for i, b in enumerate(batches, start=1)]
        for t in threads:
            t.wait()
    else:
        for b_idx, b in enumerate(batches, start=1):
            try:
                _process_batch(b)
            except Exception as e:
                print(f"[movie_library] Worker error on batch {b_idx}: {e}", flush=True)
                worker_errors.append(e)
            if _USE_EVENTLET:
                _ev.sleep(0)

    api_done_msg = f"API analysis done: {done[0]}/{len(items)} clips in {(time.time()-api_start)/60:.1f} min"
    print(f"[movie_library] {api_done_msg}", flush=True)
    if emit:
        emit("movie", api_done_msg)

    if worker_errors:
        raise RuntimeError(f"[movie_library] Clip analysis failed: {worker_errors[0]}")

    return cached_results + results


# ── Gemini validation (0.85, як в FAA) ────────────────────────────────────────

def _validation_cache_path(clip_path: str, section_text: str) -> str:
    import hashlib
    h = hashlib.md5(section_text[:300].encode()).hexdigest()[:12]
    return clip_path + f".movval_{h}.json"


def validate_clip(clip_path: str, section_text: str) -> float:
    """
    Gemini оцінює наскільки кліп підходить для сегменту нарації.
    Результат кешується поряд з кліпом.
    Повертає float 0.0–1.0.
    """
    cache_path = _validation_cache_path(clip_path, section_text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return float(json.load(f).get("score", 0.0))
        except Exception:
            pass

    from google.genai import types
    client, model = _gemini()

    parts = []
    for ratio in [0.0, 0.5, 1.0]:
        fb = _frame_bytes(clip_path, ratio)
        if fb:
            parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    if not parts:
        return 0.0

    prompt = (
        "These are 3 frames from an animated movie clip.\n"
        f'The narration script section this clip should illustrate: "{section_text[:300]}"\n'
        "Rate how well this clip fits this narration — considering the characters shown, "
        "their emotional state, and the psychological theme being discussed.\n"
        'JSON only: {"score": 0.0} where 0.0=completely wrong, 1.0=perfect fit.'
    )
    parts.append(prompt)

    got_response = False
    try:
        r = client.models.generate_content(model=model, contents=parts)
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group() if m else text)
        score = float(data.get("score", 0.0))
        got_response = True
    except Exception as e:
        if _is_gemini_auth_error(e):
            raise RuntimeError(f"[movie_library] Gemini auth/config error: {e}") from e
        score = 0.0

    if got_response:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"score": score}, f)
        except Exception:
            pass

    return score


_BATCH_VALIDATION_PROMPT = """\
You are evaluating {n} animated movie clips against a narration segment.

Narration: "{section_text}"

For each clip, 3 frames are shown (start/middle/end), labeled CLIP 1, CLIP 2, etc.

Rate how well each clip fits this narration — considering characters shown,
their emotional state, and the psychological theme being discussed.

Reply ONLY with a JSON array of exactly {n} numbers (0.0–1.0), e.g. [0.9, 0.3, 0.7]
where 0.0=completely wrong, 1.0=perfect fit. No markdown."""


def validate_clips_batch(clip_paths: list, section_text: str) -> list:
    """
    Batch Gemini validation: оцінює список кліпів за один запит.
    Повертає list[float] тієї ж довжини що й clip_paths.
    Кешує кожен результат окремо.
    """
    from google.genai import types

    # Перевіряємо кеш для кожного кліпу
    scores   = [None] * len(clip_paths)
    to_fetch = []  # (original_index, clip_path)

    for i, cp in enumerate(clip_paths):
        cache_path = _validation_cache_path(cp, section_text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    scores[i] = float(json.load(f).get("score", 0.0))
                continue
            except Exception:
                pass
        to_fetch.append((i, cp))

    if not to_fetch:
        return scores

    client, model = _gemini()

    # Збираємо фрейми для некешованих кліпів
    items = []
    for orig_idx, cp in to_fetch:
        frames = []
        for ratio in [0.0, 0.5, 1.0]:
            fb = _frame_bytes(cp, ratio)
            if fb:
                frames.append(fb)
        items.append({"orig_idx": orig_idx, "clip_path": cp, "frames": frames})

    # Відправляємо батчами по BATCH_SIZE
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        contents = []
        for j, item in enumerate(batch):
            contents.append(f"CLIP {j + 1}:")
            for fb in item["frames"]:
                contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
        contents.append(_BATCH_VALIDATION_PROMPT.format(
            n=len(batch), section_text=section_text[:300]
        ))

        batch_scores = None
        for attempt in range(3):
            try:
                r    = client.models.generate_content(model=model, contents=contents)
                text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
                text = re.sub(r"\s*```$", "", text)
                m    = re.search(r"\[.*?\]", text, re.DOTALL)
                raw  = json.loads(m.group() if m else text)
                if isinstance(raw, list) and len(raw) >= len(batch):
                    batch_scores = [float(x) for x in raw[:len(batch)]]
                break
            except Exception as e:
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[movie_library] Gemini auth error: {e}") from e
                is_rate = "429" in str(e) or "quota" in str(e).lower()
                if is_rate and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                else:
                    print(f"[movie_library] Batch validation error: {e}", flush=True)
                    break

        for j, item in enumerate(batch):
            s = (batch_scores[j] if batch_scores and j < len(batch_scores) else 0.0)
            scores[item["orig_idx"]] = s
            try:
                cache_path = _validation_cache_path(item["clip_path"], section_text)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"score": s}, f)
            except Exception:
                pass

    # Заповнюємо None → 0.0 на випадок помилок
    return [s if s is not None else 0.0 for s in scores]


# ── Text-only ranking (NEW: replaces visual validation in movie pipeline) ─────

_TEXT_RANK_PROMPT = """\
You are picking the best clip out of {n} candidates for a narration segment.
{main_char_note}
CONTEXT (what's being said around this moment):
Previous: {prev_text}
CURRENT: {current_text}
Next: {next_text}

CANDIDATES — each one already has a description, tags and themes from a prior visual analysis:

{candidates_block}

First decide what the CURRENT segment is mainly doing:
- talking about the MAIN CHARACTER of the whole video
- talking about some OTHER named character
- making a GENERAL psychological point

Score every candidate from 0.0 (totally unrelated) to 1.0 (perfect fit) based on
how well it illustrates the CURRENT narration. Use Previous/Next only as context
to disambiguate the current segment — don't reward clips that fit the next or
previous sentence better than the current one.

Scoring anchors:
- 0.90-1.00 = near-perfect visual match for the current narration
- 0.70-0.89 = strong usable match
- 0.40-0.69 = weak or partial match
- 0.00-0.39 = poor or misleading match

Important:
- Prefer meaning-level matching, not exact word matching.
- Character names may be translated, declined, inflected, or slightly misspelled.
- When the segment is general but the whole video is about one main character,
  prefer clips that keep that character visually present.
- Literal visual relevance beats vague mood similarity.

Reply with JSON only, no markdown, exactly:
{{"scores": [0.0, 0.0, 0.0, 0.0, 0.0]}}
The list must have exactly {n} numbers in the same order as the candidates."""


def _format_candidates_block(candidates: list) -> str:
    """Render candidates 1..N as a compact text block for the ranking prompt."""
    lines = []
    for i, c in enumerate(candidates, start=1):
        chars = ", ".join(c.get("characters", []) or []) or "—"
        tags = ", ".join((c.get("tags", []) or [])[:8]) or "—"
        themes = ", ".join(c.get("themes", []) or []) or "—"
        emotion = c.get("emotion", "neutral")
        scene = c.get("scene_type", "—")
        desc = (c.get("description", "") or "").strip().replace("\n", " ")
        lines.append(
            f"CLIP {i}:\n"
            f"  description: {desc}\n"
            f"  characters: {chars}\n"
            f"  emotion: {emotion}\n"
            f"  scene_type: {scene}\n"
            f"  tags: {tags}\n"
            f"  themes: {themes}"
        )
    return "\n\n".join(lines)


def _call_text_ranker(prompt: str) -> list | None:
    """
    Send the ranking prompt to Pioneer (primary) → GigaCoder (fallback).
    Returns list[float] or None on total failure.
    """
    import urllib.request
    import urllib.error

    settings = config.load_settings()

    def _parse_scores(body_text: str, expected_n: int) -> list | None:
        if not body_text or not body_text.strip():
            print("[ranker] EMPTY content from model (likely max_tokens too small / thinking model)", flush=True)
            return None
        try:
            text = re.sub(r"^```(?:json)?\s*", "", body_text.strip())
            text = re.sub(r"\s*```$", "", text)

            arr = None
            # Shape 1: {"scores": [...]}  — find an object that has a scores key
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict) and isinstance(data.get("scores"), list):
                        arr = data["scores"]
                except Exception:
                    pass
            # Shape 2: bare JSON array [0.9, 0.3, ...]
            if arr is None:
                m2 = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
                if m2:
                    try:
                        cand = json.loads(m2.group())
                        if isinstance(cand, list):
                            arr = cand
                    except Exception:
                        pass
            # Shape 3: last resort — pull all floats out of the text
            if arr is None:
                nums = re.findall(r"-?\d*\.?\d+", text)
                if nums:
                    arr = nums

            if not isinstance(arr, list) or not arr:
                print(f"[ranker] WARNING: no scores found in response: {text[:300]!r}", flush=True)
                return None

            out = []
            for x in arr[:expected_n]:
                try:
                    out.append(max(0.0, min(1.0, float(x))))
                except Exception:
                    out.append(0.0)
            while len(out) < expected_n:
                out.append(0.0)
            return out
        except Exception as e:
            print(f"[ranker] PARSE FAIL ({e}): body={body_text[:300]!r}", flush=True)
            return None

    expected_n = prompt.count("CLIP ")  # crude but OK — we control the prompt

    # Try Pioneer first
    pio_keys = settings.get("pioneer_api_keys", [])
    pio_url = settings.get("pioneer_api_url", "")
    pio_model = settings.get("pioneer_model", "gemini-3.5-flash")
    if pio_keys and pio_url:
        _, rotated = _next_pioneer_key(pio_keys)
        for key in rotated:
            for attempt in range(3):
                try:
                    payload = json.dumps({
                        "model": pio_model,
                        "messages": [
                            {"role": "system", "content": "You rank clip candidates for a narration. Reply with strict JSON only, e.g. {\"scores\":[0.9,0.3]}. No prose, no reasoning."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 2000,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        pio_url, data=payload,
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    msg = body.get("choices", [{}])[0].get("message", {}) or {}
                    text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
                    if not text:
                        fr = body.get("choices", [{}])[0].get("finish_reason")
                        print(f"[ranker:pioneer] EMPTY content finish_reason={fr} usage={body.get('usage')}", flush=True)
                    scores = _parse_scores(text, expected_n)
                    if scores is not None:
                        return scores
                    print(f"[ranker:pioneer] PARSE FAIL — raw response: {text[:300]!r}", flush=True)
                    break  # got a response but couldn't parse — try next key once
                except Exception as e:
                    err = str(e).lower()
                    is_net = any(s in err for s in (
                        "lookup timed out", "11002", "timed out", "connection",
                        "temporarily unavailable", "name or service",
                    ))
                    if is_net and attempt < 2:
                        time.sleep([1, 3, 7][attempt])
                        continue
                    break

    # Fallback: GigaCoder
    gc_keys = settings.get("gigacoder_api_keys", [])
    gc_url = settings.get("gigacoder_api_url", "")
    gc_model = settings.get("gigacoder_model", "gpt-5.4-mini")
    if gc_keys and gc_url:
        _, rotated = _next_gigacoder_key(gc_keys)
        for key in rotated:
            for attempt in range(3):
                try:
                    payload = json.dumps({
                        "model": gc_model,
                        "messages": [
                            {"role": "system", "content": "You rank clip candidates for a narration. Reply with strict JSON only, e.g. {\"scores\":[0.9,0.3]}. No prose, no reasoning."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 2000,
                        "temperature": 0,
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        gc_url, data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {key}",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                            "Accept": "application/json",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    msg = body.get("choices", [{}])[0].get("message", {}) or {}
                    text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
                    scores = _parse_scores(text, expected_n)
                    if scores is not None:
                        return scores
                    print(f"[ranker:gigacoder] PARSE FAIL — raw response: {text[:300]!r}", flush=True)
                    break
                except Exception as e:
                    err = str(e).lower()
                    is_net = any(s in err for s in (
                        "lookup timed out", "11002", "timed out", "connection",
                        "temporarily unavailable", "name or service",
                    ))
                    if is_net and attempt < 2:
                        time.sleep([1, 3, 7][attempt])
                        continue
                    break

    return None


def _normalize_character_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _detect_mentioned_characters(text: str, all_known_chars: set) -> set:
    """Find which known characters are explicitly mentioned in the segment text."""
    if not text or not all_known_chars:
        return set()
    normalized_text = f" {_normalize_character_text(text)} "
    found = set()
    for ch in all_known_chars:
        if not ch or len(ch) < 2:
            continue
        normalized_ch = _normalize_character_text(ch)
        if normalized_ch and f" {normalized_ch} " in normalized_text:
            found.add(ch)
    return found


def _apply_score_modifiers(
    candidate: dict,
    base_score: float,
    segment_text: str,
    main_characters: list,
    mentioned_characters: set,
    score_rules: dict,
) -> tuple:
    """
    Apply niche-defined score rules on top of the base Pioneer score.
    Returns (final_score, breakdown_dict) for logging/debugging.
    """
    rules = score_rules or {}
    breakdown = {"base": base_score}

    char_bonus = 0.0
    char_pen   = 0.0
    clip_chars = set(candidate.get("characters", []) or [])
    main_set   = {c.strip() for c in (main_characters or []) if c and c.strip()}

    # Direct match: clip contains a character explicitly mentioned in segment
    matched_mentioned = clip_chars & mentioned_characters
    if matched_mentioned:
        char_bonus += float(rules.get("character_mentioned_bonus", 0.40))

    # Main character bonus when no explicit mention or the mentioned one is also main
    if main_set and (clip_chars & main_set):
        char_bonus += float(rules.get("main_character_bonus", 0.20))

    # Penalty: segment mentions a non-main character, but clip shows ONLY main
    if mentioned_characters and not (clip_chars & mentioned_characters):
        non_main_mentioned = mentioned_characters - main_set
        if non_main_mentioned and clip_chars and (clip_chars <= main_set):
            char_pen += float(rules.get("wrong_character_penalty", -0.20))

    # Penalty when clip has no relevant character and no main char either
    if not (clip_chars & mentioned_characters) and not (clip_chars & main_set):
        if clip_chars or mentioned_characters or main_set:
            char_pen += float(rules.get("no_relevant_character_penalty", -0.15))

    # Cap the positive character bonus
    cap = float(rules.get("character_bonus_cap", 0.50))
    char_bonus = min(char_bonus, cap)
    breakdown["char_bonus"] = char_bonus
    breakdown["char_penalty"] = char_pen

    # Scene type penalty
    scene_pen = 0.0
    scene_penalties = rules.get("scene_penalties", {}) or {}
    scene_type = candidate.get("scene_type", "")
    if scene_type and scene_type in scene_penalties:
        scene_pen = float(scene_penalties[scene_type])
    breakdown["scene_penalty"] = scene_pen

    # Theme bonus (soft signal)
    theme_bonus = 0.0
    psych_themes = set(rules.get("psychology_themes", []) or [])
    clip_themes = set(candidate.get("themes", []) or [])
    bonus_per = float(rules.get("theme_bonus_per_match", 0.05))
    seg_lower = (segment_text or "").lower()
    matches = 0
    for theme in clip_themes & psych_themes:
        # Theme bonus if theme word appears in the segment text
        keyword = theme.replace("_", " ")
        if keyword in seg_lower:
            matches += 1
    theme_bonus = min(matches * bonus_per, float(rules.get("theme_bonus_cap", 0.20)))
    breakdown["theme_bonus"] = theme_bonus

    final = base_score + char_bonus + char_pen + scene_pen + theme_bonus
    breakdown["final"] = final
    return final, breakdown


def rank_clips_by_text(
    candidates: list,
    segment_text: str,
    prev_text: str = "",
    next_text: str = "",
    main_characters: list = None,
    score_rules: dict = None,
    all_known_chars: set = None,
) -> list:
    """
    Rank up to N candidates against the narration using a single text-only API call,
    then apply niche score modifiers locally.

    Returns a sorted list of (candidate, final_score, breakdown) — best first.
    """
    if not candidates:
        return []

    # Step 1: build prompt and get base scores from Pioneer/GigaCoder
    block = _format_candidates_block(candidates)

    # Build extended rules block — encodes everything score_rules used to do locally,
    # so the model itself applies all the bonuses/penalties (works in any language).
    rules = score_rules or {}
    rule_lines = ["SCORING RULES (apply all of these when scoring):"]

    if main_characters:
        names = ", ".join(main_characters)
        rule_lines.append(
            f"- VIDEO FOCUS: this whole video is about {names}. The narration is in another "
            f"language (Polish, German, French, Spanish, Portuguese), so {names} may appear "
            f"there translated (e.g. Tigress → tygrysica, Tigerin, tigresse, tigresa). "
            f"Recognize all of these as the same character."
        )
        rule_lines.append(
            f"- Strongly PREFER clips showing {names}. Give them a bonus of about +{rules.get('main_character_bonus', 0.20):.2f}."
        )
        rule_lines.append(
            f"- If the CURRENT narration explicitly mentions a specific named character "
            f"(in any language) AND that character is visible in the clip, give an even "
            f"stronger bonus of about +{rules.get('character_mentioned_bonus', 0.40):.2f}."
        )
        rule_lines.append(
            f"- If the narration is clearly about a different character (e.g. Tai Lung, Shifu, Po) "
            f"and the clip only shows {names} without that mentioned character, "
            f"apply a penalty of about {rules.get('wrong_character_penalty', -0.20):.2f}."
        )
        rule_lines.append(
            f"- If the clip shows NO character relevant to the narration and NO {names} either, "
            f"apply a small penalty of about {rules.get('no_relevant_character_penalty', -0.15):.2f}."
        )

    # Scene type penalties — credits, title cards, transitions
    scene_pen = rules.get("scene_penalties") or {}
    if scene_pen:
        bad_scenes = [s for s, v in scene_pen.items() if v <= -0.5]
        soft_bad = [s for s, v in scene_pen.items() if -0.5 < v < 0]
        if bad_scenes:
            rule_lines.append(
                f"- HARD PENALTY: clips marked as scene_type {bad_scenes} are almost always wrong "
                f"(credits / title cards / logos). Score them at most 0.05."
            )
        if soft_bad:
            rule_lines.append(
                f"- Soft penalty for scene_type {soft_bad} (transitions, generic shots) — "
                f"only pick them when nothing else fits."
            )

    # Theme matching (psychology themes etc.)
    themes_list = rules.get("psychology_themes") or []
    if themes_list:
        sample = ", ".join(themes_list[:6])
        rule_lines.append(
            f"- Small bonus when the clip's themes match emotional/psychological themes in "
            f"the narration (e.g. {sample}). Don't over-reward this — it's a tiebreaker."
        )

    # Generic guardrails
    rule_lines.append(
        "- Don't reward clips that fit the previous or next sentence better than the current one."
    )
    rule_lines.append(
        "- A clip's literal description should match the action being narrated. Visual mood "
        "alone is not enough."
    )

    rules_block = "\n".join(rule_lines)

    prompt = _TEXT_RANK_PROMPT.format(
        n=len(candidates),
        main_char_note="\n" + rules_block + "\n",
        prev_text=(prev_text or "—").strip()[:300],
        current_text=(segment_text or "").strip()[:400],
        next_text=(next_text or "—").strip()[:300],
        candidates_block=block,
    )

    base_scores = _call_text_ranker(prompt)
    if base_scores is None:
        print(f"[ranker] _call_text_ranker returned None — API or parsing failed (candidates={len(candidates)})", flush=True)
        base_scores = [0.5] * len(candidates)
    elif len(base_scores) < len(candidates):
        print(f"[ranker] returned {len(base_scores)} scores for {len(candidates)} candidates — padding", flush=True)
        base_scores = [0.5] * len(candidates)
    elif all(float(s) <= 0.0001 for s in base_scores):
        print("[ranker] returned all-zero scores; using neutral fallback scores", flush=True)
        base_scores = [0.5] * len(candidates)

    # Step 2: detect mentioned characters in current segment
    if all_known_chars is None:
        # Auto-derive from candidate pool if caller didn't supply
        all_known_chars = set()
        for c in candidates:
            for ch in c.get("characters", []) or []:
                if ch:
                    all_known_chars.add(ch)

    mentioned = _detect_mentioned_characters(segment_text or "", all_known_chars)

    # Step 3: apply local modifiers
    ranked = []
    for c, base in zip(candidates, base_scores):
        final, breakdown = _apply_score_modifiers(
            candidate=c,
            base_score=base,
            segment_text=segment_text or "",
            main_characters=main_characters or [],
            mentioned_characters=mentioned,
            score_rules=score_rules or {},
        )
        ranked.append((c, final, breakdown))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ── Uniqualization ─────────────────────────────────────────────────────────────

def _uniqualize_movie_clip(input_path: str, output_path: str, params: dict):
    """Агресивніша унікалізація для кліпів фільму."""
    grain = random.uniform(8, 15)
    flip  = params.get("flip", False)

    vf_parts = [
        f"scale=iw*{params['zoom']:.4f}:ih*{params['zoom']:.4f}",
        "crop=1920:1080",
        (f"eq=brightness={params['brightness']:.3f}"
         f":contrast={params['contrast']:.3f}"
         f":saturation={params['saturation']:.3f}"),
        f"noise=alls={grain:.1f}:allf=t+u",
    ]
    if flip:
        vf_parts.append("hflip")

    subprocess.run(
        [FFMPEG, "-y", "-i", input_path,
         "-vf", ",".join(vf_parts),
         *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p", "-an",
         output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    )


def make_uniq_params() -> dict:
    return {
        "zoom":       random.uniform(1.04, 1.08),
        "brightness": random.uniform(-0.05, 0.05),
        "contrast":   random.uniform(0.95, 1.08),
        "saturation": random.uniform(0.90, 1.15),
        "flip":       random.random() < 0.30,
    }


# ── Публічний API ──────────────────────────────────────────────────────────────

def process_movie(movie_path: str, movie_name: str, emit=None) -> dict:
    """
    Нарізати фільм на кліпи, проаналізувати через Gemini, зберегти індекс в GDrive.
    Якщо фільм вже проіндексований — повертає кеш.
    """
    def log(msg):
        print(f"[movie_library:{movie_name}] {msg}", flush=True)
        if emit:
            emit("movie", msg)

    idx_path = _index_path(movie_name)
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        log(f"Already indexed: {len(index['clips'])} clips")
        return index

    clips_out_dir = _clips_dir(movie_name)
    os.makedirs(clips_out_dir, exist_ok=True)

    log("Getting duration...")
    total_dur = _get_duration(movie_path)
    if total_dur < 10:
        raise ValueError(f"Movie too short or unreadable: {movie_path}")

    log(f"Duration: {total_dur / 60:.1f} min. Detecting scene changes...")
    movie_id = re.sub(r"[^\w]", "_", movie_name.lower())[:20]
    clips    = _cut_by_scenes(movie_path, clips_out_dir, movie_id, total_dur, emit=emit)
    log(f"Cut {len(clips)} clips. Starting Gemini analysis...")

    analyzed = _analyze_all_clips(clips, movie_name, emit=emit)

    # Відфільтрувати blurry/static
    good = [a for a in analyzed
            if not a.get("is_blurry") and not a.get("is_static")]
    log(f"Analysis done: {len(good)}/{len(analyzed)} clips passed quality check")

    index = {
        "movie_name": movie_name,
        "movie_id":   movie_id,
        "total_dur":  total_dur,
        "clips":      good,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    log(f"Index saved: {idx_path}")
    return index


def process_movie_folder(folder_path: str, movie_name: str, emit=None) -> dict:
    """
    Обробити всі mp4 файли в папці як один фільм.
    Підтримує інкрементальну індексацію — нові файли додаються до існуючого індексу.
    Зберігає прогрес після кожного файлу (якщо впаде — можна продовжити).
    """
    def log(msg):
        print(f"[movie_library:{movie_name}] {msg}", flush=True)
        if emit:
            emit("movie", msg)

    # Знайти всі mp4 файли
    try:
        all_files = sorted([
            os.path.join(folder_path, fn)
            for fn in os.listdir(folder_path)
            if fn.lower().endswith(".mp4")
        ])
    except Exception as e:
        raise ValueError(f"Cannot read folder: {folder_path}: {e}")

    if not all_files:
        raise ValueError(f"No mp4 files found in: {folder_path}")

    log(f"Found {len(all_files)} mp4 file(s): {[os.path.basename(f) for f in all_files]}")

    idx_path      = _index_path(movie_name)
    clips_out_dir = _clips_dir(movie_name)
    os.makedirs(clips_out_dir, exist_ok=True)

    # Завантажити існуючий індекс (якщо є)
    existing_index    = {}
    processed_sources = set()
    all_good_clips    = []
    total_dur         = 0.0
    movie_id          = re.sub(r"[^\w]", "_", movie_name.lower())[:20]

    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            existing_index = json.load(f)
        processed_sources = set(existing_index.get("processed_sources", []))
        all_good_clips    = existing_index.get("clips", [])
        total_dur         = existing_index.get("total_dur", 0.0)
        log(f"Existing index: {len(all_good_clips)} clips, "
            f"{len(processed_sources)} source(s) already processed")

    new_files_processed = 0

    for file_idx, movie_path in enumerate(all_files):
        src_key = os.path.basename(movie_path)
        if src_key in processed_sources:
            log(f"Already processed: {src_key} — skipping")
            continue

        log(f"[{file_idx + 1}/{len(all_files)}] Processing: {src_key}")
        dur = _get_duration(movie_path)
        if dur < 10:
            log(f"Skipping (too short or unreadable): {src_key}")
            continue

        # Унікальний префікс для кожного файлу — щоб кліпи не перезаписувались
        file_prefix = f"{movie_id}_f{file_idx:02d}"

        # Reuse already-cut clips from previous interrupted runs.
        # If we find existing clips with this prefix in clips_out_dir, skip the
        # expensive scene-detection + cutting step and reuse what's on disk.
        existing_clips_on_disk = sorted([
            fn for fn in os.listdir(clips_out_dir)
            if fn.startswith(file_prefix + "_") and fn.endswith(".mp4")
        ])

        if existing_clips_on_disk:
            log(f"Found {len(existing_clips_on_disk)} clips already cut for {src_key} — reusing, skipping scene detection")
            clips = []
            for fn in existing_clips_on_disk:
                full = os.path.join(clips_out_dir, fn)
                clip_dur = _get_duration(full)
                if clip_dur < CLIP_MIN:
                    continue
                clip_id = fn[:-4]  # strip .mp4
                clips.append({
                    "id":    clip_id,
                    "file":  full,
                    "start": 0.0,
                    "end":   clip_dur,
                })
            log(f"Reused {len(clips)} clips. Starting Gemini analysis...")
        else:
            log(f"Duration: {dur / 60:.1f} min. Detecting scene changes...")
            clips = _cut_by_scenes(movie_path, clips_out_dir, file_prefix, dur, emit=emit)
            log(f"Cut {len(clips)} clips. Starting Gemini analysis...")

        analyzed = _analyze_all_clips(clips, movie_name, emit=emit)
        good = [a for a in analyzed
                if not a.get("is_blurry") and not a.get("is_static")]
        log(f"{len(good)}/{len(analyzed)} clips passed quality check for {src_key}")

        all_good_clips.extend(good)
        total_dur += dur
        processed_sources.add(src_key)
        new_files_processed += 1

        # Зберігаємо після кожного файлу — щоб не втратити прогрес при збої
        index = {
            "movie_name":        movie_name,
            "movie_id":          movie_id,
            "total_dur":         total_dur,
            "clips":             all_good_clips,
            "processed_sources": sorted(processed_sources),
            "created_at":        existing_index.get(
                                     "created_at",
                                     time.strftime("%Y-%m-%dT%H:%M:%S")),
            "updated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        log(f"Index saved: {len(all_good_clips)} total clips so far")

    if new_files_processed == 0:
        log(f"All files already processed. Total: {len(all_good_clips)} clips")
    else:
        log(f"Done! Processed {new_files_processed} new file(s). "
            f"Total: {len(all_good_clips)} clips")

    return {
        "movie_name": movie_name,
        "clips":      all_good_clips,
        "total_dur":  total_dur,
    }


def list_movies() -> list:
    """Список всіх проіндексованих фільмів."""
    movies_dir = _movies_dir()
    if not os.path.exists(movies_dir):
        return []
    result = []
    for name in os.listdir(movies_dir):
        idx = _index_path(name)
        if os.path.exists(idx):
            try:
                with open(idx, encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "name":       name,
                    "clip_count": len(data.get("clips", [])),
                    "duration":   data.get("total_dur", 0),
                    "created_at": data.get("created_at", ""),
                })
            except Exception:
                pass
    return result


def get_movie_clips(movie_name: str) -> list:
    """Всі кліпи з індексу фільму."""
    idx_path = _index_path(movie_name)
    if not os.path.exists(idx_path):
        return []
    with open(idx_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("clips", [])


_REJECT_KEYWORDS = {
    "credits", "credit", "end credits", "opening credits", "title card",
    "title screen", "text screen", "intertitle", "author", "authors",
    "directed by", "produced by", "written by", "cast", "crew",
    "copyright", "logo", "studio logo", "black screen", "blank",
    "the end", "fin",
}


def _score_clip(clip: dict, segment_text: str) -> float:
    """Розумний keyword-based скор без Gemini. Fuzzy matching:
    - Shifu знаходить Master Shifu
    - po знаходить kung fu panda, young panda
    - anger знаходить anger, angry, outburst of anger
    Rejects: credits, title cards, text-only screens → score = -1
    """
    desc_lower = clip.get("description", "").lower()
    tags_lower = [t.lower() for t in clip.get("tags", [])]
    scene_type = clip.get("scene_type", "").lower()

    for kw in _REJECT_KEYWORDS:
        if kw in desc_lower or kw in scene_type:
            return -1.0
        for tag in tags_lower:
            if kw in tag:
                return -1.0

    text_lower = segment_text.lower()
    text_words = set(text_lower.split())
    score = 0.0

    def _word_match(query: str, text: str, full_bonus: float = 1.0, partial_bonus: float = 0.4) -> float:
        """Повертає бонус якщо query входить в text (повністю або частково)."""
        q = query.lower().strip()
        if not q:
            return 0.0
        # Full match
        if q in text:
            return full_bonus
        # Partial: Shifu matches Master Shifu
        words = text.split()
        for w in words:
            if q in w or w in q:
                return partial_bonus
        return 0.0

    # Characters: "Master Shifu" matches "Shifu teaches Po" -> Shifu = partial match
    for char in clip.get("characters", []):
        score += _word_match(char, text_lower, full_bonus=5.0, partial_bonus=3.0)

    # Emotion: strong match
    if clip.get("emotion"):
        score += _word_match(clip["emotion"], text_lower, full_bonus=4.0, partial_bonus=2.5)

    # Scene type
    if clip.get("scene_type"):
        score += _word_match(clip["scene_type"], text_lower, full_bonus=3.0, partial_bonus=1.5)

    # Themes: each word of theme checked separately
    for theme in clip.get("themes", []):
        for w in theme.replace("_", " ").split():
            if len(w) > 2:
                score += _word_match(w, text_lower, full_bonus=2.5, partial_bonus=1.5)

    # Tags: full tag match + partial word match
    for tag in clip.get("tags", []):
        score += _word_match(tag, text_lower, full_bonus=2.0, partial_bonus=1.0)

    # Description: words > 4 chars
    for word in clip.get("description", "").lower().split():
        if len(word) > 4:
            score += _word_match(word, text_lower, full_bonus=0.8, partial_bonus=0.3)

    return score


def _has_embeddings(clips: list) -> bool:
    """
    True якщо семантичний пошук доцільний — тобто векторами покрита БІЛЬШІСТЬ
    кліпів. При частковому бекфілі (вектори лише в частини кліпів) повертаємо
    False, щоб не загубити кліпи без векторів — тоді працює keyword по всіх.
    """
    if not clips:
        return False
    with_emb = sum(1 for c in clips if c.get("embedding"))
    return with_emb >= len(clips) * 0.8


def _semantic_rank(segment_text: str, clips: list, used_ids: set, top_n: int) -> list:
    """
    Ранжувати кліпи за косинусною близькістю їхнього вектора до вектора сегмента.
    Кліпи-кредити/титри відсіюються (як і в keyword-режимі).
    Повертає список clip dict (найрелевантніші першими), без рандому.
    Якщо вектор сегмента порахувати не вдалось — повертає None (→ keyword fallback).
    """
    from backend import embeddings as _emb

    seg_vec = _emb.embed_text(segment_text)
    if not seg_vec:
        return None

    scored = []
    for clip in clips:
        if clip.get("id") in used_ids:
            continue
        if not os.path.exists(clip.get("file", "")):
            continue
        emb = clip.get("embedding")
        if not emb:
            continue
        # Відсів кредитів/титрів/текстових екранів (та сама логіка, що в _score_clip)
        if _score_clip(clip, segment_text) < 0:
            continue
        sim = _emb.cosine(seg_vec, emb)
        if sim >= SEMANTIC_MIN_SIM:
            scored.append((sim, clip))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]


def search_clips(segment_text: str, movie_name: str = None,
                 used_ids: set = None, top_n: int = 15,
                 gemini_validate: bool = False) -> list:
    """
    Знайти кліпи що підходять до тексту сегменту нарації.

    Крок 1: семантичний пошук за embedding-векторами (за СЕНСОМ, не за словами).
            Якщо вектори відсутні або недоступні — fallback на keyword scoring.
    Крок 2: Gemini validation 0.85 (якщо gemini_validate=True).
    Fallback: якщо нічого не пройшло валідацію — повертає top-5 без валідації.
    """
    if movie_name:
        all_clips = get_movie_clips(movie_name)
    else:
        all_clips = []
        for m in list_movies():
            all_clips.extend(get_movie_clips(m["name"]))

    used_ids = used_ids or set()

    # ── Крок 1: семантичний пошук (пріоритетний) ──
    top = None
    if _has_embeddings(all_clips):
        semantic = _semantic_rank(segment_text, all_clips, used_ids, top_n)
        if semantic is not None:
            top = semantic

    # ── Fallback: keyword scoring (якщо немає векторів або їх не порахувати) ──
    if top is None:
        candidates = []
        for clip in all_clips:
            if clip.get("id") in used_ids:
                continue
            if not os.path.exists(clip.get("file", "")):
                continue
            s = _score_clip(clip, segment_text)
            if s > 0:
                candidates.append((s, clip))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in candidates[:top_n]]

    if not gemini_validate or not top:
        return top

    # Gemini validation: оцінюємо топ-10, фільтруємо >= 0.85
    validated = []
    for clip in top[:10]:
        gem_score = validate_clip(clip["file"], segment_text)
        if gem_score >= VALIDATION_THRESHOLD:
            validated.append((gem_score, clip))

    if validated:
        validated.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in validated]

    # Fallback — повертаємо top-5 без валідації
    return top[:5]
