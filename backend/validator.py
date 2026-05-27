import json
import os
import re
import shutil
import subprocess
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

CACHE_SUFFIX  = ".faa_score.json"
CACHE_VERSION = 1
BATCH_SIZE    = 8


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


def _make_client():
    from google import genai
    _setup_vertex()
    settings = config.load_settings()
    return genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )


def _score_single(frames: list, description: str) -> dict:
    from google.genai import types
    settings = config.load_settings()
    client   = _make_client()
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


def score_clip(video_path: str, description: str) -> dict:
    cache_path = video_path + CACHE_SUFFIX
    cached = _read_cache(cache_path, description)
    if cached:
        return cached

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = _extract_frames(video_path, tmp)
            if not frames:
                return {"score": 0.0, "reason": "no frames extracted"}
            result = _score_single(frames, description)
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
