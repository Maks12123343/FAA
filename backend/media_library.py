import json
import os
import subprocess
import sys
import glob
import random
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend.validator import score_clip, is_valid

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE


def _get_duration(path: str) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _cut_clip(src: str, out: str, start: float, duration: float):
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "ultrafast",
         "-an", out],
        capture_output=True, timeout=60,
    )


def download_from_channel(video_url: str, niche_name: str, max_clips: int = 20) -> list:
    """Download a video and cut it into clips, save to library."""
    niche_dir = os.path.join(config.LIBRARY_DIR, niche_name, "raw")
    os.makedirs(niche_dir, exist_ok=True)

    settings     = config.load_settings()
    min_dur      = settings.get("clip_min_duration", 2)
    max_dur      = settings.get("clip_max_duration", 5)

    with tempfile.TemporaryDirectory() as tmp:
        out_tpl = os.path.join(tmp, "source.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
             "--merge-output-format", "mp4", "-o", out_tpl, video_url],
            check=True, capture_output=True, timeout=600,
        )
        files = glob.glob(os.path.join(tmp, "source.*"))
        if not files:
            return []
        src = files[0]
        total = _get_duration(src)
        if total < min_dur:
            return []

        clips_saved = []
        t = 0.0
        idx = 0
        while t < total and idx < max_clips:
            dur  = random.uniform(min_dur, max_dur)
            dur  = min(dur, total - t)
            if dur < min_dur:
                break

            clip_name = f"{os.path.splitext(os.path.basename(video_url.split('v=')[-1]))[0]}_{idx:04d}.mp4"
            clip_path = os.path.join(niche_dir, clip_name)

            _cut_clip(src, clip_path, t, dur)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 5000:
                clips_saved.append(clip_path)
                idx += 1
            t += dur

    return clips_saved


def validate_library(niche_name: str, description: str, threshold: float | None = None) -> dict:
    """Score all raw clips in a niche and move passing ones to validated/."""
    raw_dir = os.path.join(config.LIBRARY_DIR, niche_name, "raw")
    val_dir = os.path.join(config.LIBRARY_DIR, niche_name, "validated")
    os.makedirs(val_dir, exist_ok=True)

    if threshold is None:
        threshold = config.load_settings().get("clip_score_threshold", 0.85)

    clips = glob.glob(os.path.join(raw_dir, "*.mp4"))
    passed, failed = 0, 0

    for clip in clips:
        result = score_clip(clip, description)
        if result["score"] >= threshold:
            dest = os.path.join(val_dir, os.path.basename(clip))
            shutil.move(clip, dest)
            passed += 1
        else:
            failed += 1
        print(f"[library] {os.path.basename(clip)}: {result['score']:.2f} — {result['reason']}", flush=True)

    return {"passed": passed, "failed": failed, "total": passed + failed}


def get_validated_clips(niche_name: str) -> list:
    val_dir = os.path.join(config.LIBRARY_DIR, niche_name, "validated")
    if not os.path.exists(val_dir):
        return []
    return glob.glob(os.path.join(val_dir, "*.mp4"))


def get_library_stats(niche_name: str) -> dict:
    raw_dir = os.path.join(config.LIBRARY_DIR, niche_name, "raw")
    val_dir = os.path.join(config.LIBRARY_DIR, niche_name, "validated")
    raw   = len(glob.glob(os.path.join(raw_dir, "*.mp4"))) if os.path.exists(raw_dir) else 0
    valid = len(glob.glob(os.path.join(val_dir, "*.mp4"))) if os.path.exists(val_dir) else 0
    return {"raw": raw, "validated": valid}
