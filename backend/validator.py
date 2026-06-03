import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

CACHE_SUFFIX  = ".faa_score.json"
CACHE_VERSION = 2  # bumped: pioneer backend added
BATCH_SIZE    = 8


# ---------------------------------------------------------------------------
# Thread-safe Pioneer.ai key pool (round-robin)
# ---------------------------------------------------------------------------

class _PioneerKeyPool:
    """Distributes API keys across threads in round-robin fashion."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._index = 0
        self._keys  = []

    def _reload(self):
        settings = config.load_settings()
        keys = settings.get("pioneer_api_keys", [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        self._keys = [k for k in keys if k]

    def next_key(self) -> str | None:
        with self._lock:
            self._reload()
            if not self._keys:
                return None
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def available(self) -> bool:
        self._reload()
        return bool(self._keys)


_key_pool = _PioneerKeyPool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_vertex():
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)


def _get_duration(path: str) -> float:
    ffprobe = config.FFPROBE
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _extract_frames(video_path: str, tmp_dir: str) -> list:
    duration = _get_duration(video_path)
    if duration <= 0:
        return []
    ffmpeg   = config.FFMPEG
    settings = config.load_settings()
    positions = settings.get("clip_frames_positions", [0.25, 0.50, 0.75])
    frames = []
    for i, p in enumerate(positions):
        ts  = max(0.0, min(duration - 0.05, duration * p))
        out = os.path.join(tmp_dir, f"frame_{i:02d}.jpg")
        subprocess.run(
            [ffmpeg, "-y", "-ss", f"{ts:.3f}", "-i", video_path,
             "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", out],
            capture_output=True, timeout=30,
        )
        if os.path.exists(out) and os.path.getsize(out) > 500:
            frames.append(out)
    return frames


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


# ---------------------------------------------------------------------------
# Pioneer.ai scoring (OpenAI-compatible vision API)
# ---------------------------------------------------------------------------

def _score_pioneer(frames: list, description: str, api_key: str) -> dict:
    """Score clip frames using pioneer.ai (OpenAI-compatible Gemini proxy)."""
    import urllib.request

    settings  = config.load_settings()
    api_url   = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")
    model     = settings.get("pioneer_model", "a87f8985-e7d8-4012-adac-6d5c66287213")

    prompt = (
        f'Frames from a short video clip. '
        f'How visually relevant is this clip to: "{description}"?\n'
        f'JSON only: {{"score": 0.0-1.0, "reason": "one sentence"}}'
    )

    # Build content array: images first, then text
    content = []
    for frame_path in frames:
        with open(frame_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    text = body["choices"][0]["message"]["content"]
    return _parse_json(text)


# ---------------------------------------------------------------------------
# Vertex AI scoring (original backend)
# ---------------------------------------------------------------------------

def _make_vertex_client():
    from google import genai
    _setup_vertex()
    settings = config.load_settings()
    return genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )


def _score_vertex(frames: list, description: str) -> dict:
    from google.genai import types
    settings = config.load_settings()
    client   = _make_vertex_client()
    model    = settings.get("gemini_model", "gemini-2.5-flash")

    prompt = (
        f'Frames from a short video clip. '
        f'How visually relevant is this clip to: "{description}"?\n'
        f'JSON only: {{"score": 0.0-1.0, "reason": "one sentence"}}'
    )
    contents = []
    for f in frames:
        with open(f, "rb") as fh:
            contents.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))
    contents.append(prompt)

    r = client.models.generate_content(model=model, contents=contents)
    return _parse_json(r.text)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_cache(cache_path: str, description: str) -> dict | None:
    try:
        with open(cache_path, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("description") == description and c.get("version") == CACHE_VERSION:
            return {"score": float(c["score"]), "reason": c.get("reason", "")}
    except Exception:
        pass
    return None


def _write_cache(cache_path: str, description: str, result: dict):
    tmp = cache_path + ".part"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "description": description, **result}, f)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_clip(video_path: str, description: str) -> dict:
    """Score a video clip for relevance to description.

    Backend priority:
      1. Pioneer.ai (if keys configured) — free, no Vertex setup needed
      2. Vertex AI (Gemini) — fallback if pioneer not configured or fails
    """
    cache_path = video_path + CACHE_SUFFIX
    cached = _read_cache(cache_path, description)
    if cached:
        return cached

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = _extract_frames(video_path, tmp)
            if not frames:
                return {"score": 0.0, "reason": "no frames extracted"}

            # Try pioneer.ai first
            api_key = _key_pool.next_key()
            if api_key:
                try:
                    result = _score_pioneer(frames, description, api_key)
                    print(f"[validator] pioneer scored {os.path.basename(video_path)}: {result.get('score')}", flush=True)
                except Exception as e:
                    print(f"[validator] pioneer failed ({e}), falling back to Vertex", flush=True)
                    result = _score_vertex(frames, description)
            else:
                result = _score_vertex(frames, description)

    except Exception as e:
        return {"score": 0.0, "reason": f"error: {e}"}

    result["score"] = round(min(max(float(result.get("score", 0.0)), 0.0), 1.0), 4)
    result.setdefault("reason", "")
    _write_cache(cache_path, description, result)
    return result


def is_valid(video_path: str, description: str, threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = config.load_settings().get("clip_score_threshold", 0.85)
    result = score_clip(video_path, description)
    return result["score"] >= threshold