import json
import os
import re
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE
COOKIES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookies.txt")


def _cookies_arg(tmp_dir: str) -> list:
    """Return --cookies <tmpfile> args using a temp copy so yt-dlp can't overwrite the original."""
    if not os.path.exists(COOKIES_FILE):
        return []
    import shutil
    tmp = os.path.join(tmp_dir, f"cookies_{os.getpid()}_{threading.current_thread().ident}.txt")
    shutil.copy2(COOKIES_FILE, tmp)
    return ["--cookies", tmp]


SCENE_THRESHOLD = 0.35  # 0-1: lower = more sensitive, more scene cuts detected


def _clip_limits() -> tuple:
    s = config.load_settings()
    return float(s.get("clip_min_duration", 2)), float(s.get("clip_max_duration", 5))


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except (subprocess.TimeoutExpired, Exception):
        return 0.0


def _cut_clip(src: str, out: str, start: float, duration: float):
    r = subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}",
         "-c:v", "copy", "-an", out],
        capture_output=True, timeout=60,
    )
    ok = (r.returncode == 0
          and os.path.exists(out)
          and os.path.getsize(out) > 1000
          and _get_duration(out) >= 0.1)
    if not ok:
        print(f"[downloader] stream copy failed (returncode={r.returncode}, dur={_get_duration(out):.2f}s), re-encoding...", flush=True)
        if os.path.exists(out):
            os.unlink(out)
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
             "-t", f"{duration:.3f}",
             *config.get_video_encoder_args("ultrafast"), "-an", out],
            capture_output=True, timeout=60,
        )


def _get_transcript(video_url: str) -> str:
    try:
        from backend import transcriber
        result = transcriber.get_transcript(video_url)
        return result.get("text", "")
    except Exception:
        return ""


def _assign_text(transcript: str, start: float, end: float, total: float) -> str:
    if not transcript or total <= 0:
        return ""
    words = transcript.split()
    s = int((start / total) * len(words))
    e = int((end   / total) * len(words))
    return " ".join(words[s:e])


def _detect_scene_timestamps(video_path: str) -> list:
    """
    Run ffmpeg scene detection and return sorted list of scene-start timestamps.
    Uses showinfo filter to extract pts_time of frames where scene score > SCENE_THRESHOLD.
    Always includes 0.0. Returns empty list (only 0.0) if no changes detected.
    """
    try:
        r = subprocess.run(
            [FFMPEG, "-threads", "6", "-i", video_path,
             "-vf", f"select=gt(scene\\,{SCENE_THRESHOLD}),showinfo",
             "-vsync", "vfr", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"[downloader] Scene detection timed out — will use fixed cuts.", flush=True)
        return [0.0]
    if r.returncode != 0:
        print(
            f"[downloader] Scene detection failed (ffmpeg exit {r.returncode}) "
            f"— will use fixed cuts. Error: {r.stderr[-200:].strip()}",
            flush=True,
        )
        return [0.0]
    timestamps = [0.0]
    for line in r.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:(\d+\.?\d*)", line)
            if m:
                t = float(m.group(1))
                if t > 0.1:
                    timestamps.append(t)
    return sorted(set(timestamps))


def _cut_by_scenes(src_path: str, pool_dir: str, vid_id: str,
                   source_url: str, transcript: str, total_dur: float, emit=None) -> list:
    clip_min, clip_max = _clip_limits()
    scene_times = _detect_scene_timestamps(src_path)
    scene_times.append(total_dur)

    if len(scene_times) <= 2:
        print(f"[downloader] {vid_id}: no scene changes detected, using fixed cuts", flush=True)
        return _cut_fixed(src_path, pool_dir, vid_id, source_url, transcript, total_dur, emit=emit)

    clips = []
    idx   = 0
    total_scenes = len(scene_times) - 1
    if emit:
        emit("clips", f"{vid_id}: cutting {total_scenes} scenes...")

    for i in range(total_scenes):
        scene_start = scene_times[i]
        scene_end   = scene_times[i + 1]
        scene_dur   = scene_end - scene_start

        if scene_dur < clip_min:
            continue

        if scene_dur <= clip_max:
            out_path = os.path.join(pool_dir, f"{vid_id}_{idx:04d}.mp4")
            _cut_clip(src_path, out_path, scene_start, scene_dur)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
                clips.append({
                    "id":         f"{vid_id}_{idx:04d}",
                    "file":       out_path,
                    "start":      round(scene_start, 2),
                    "end":        round(scene_end, 2),
                    "text":       _assign_text(transcript, scene_start, scene_end, total_dur),
                    "source_url": source_url,
                })
                idx += 1
        else:
            t = scene_start
            while t + clip_min <= scene_end:
                chunk_dur = min(clip_max, scene_end - t)
                if chunk_dur < clip_min:
                    break
                out_path = os.path.join(pool_dir, f"{vid_id}_{idx:04d}.mp4")
                _cut_clip(src_path, out_path, t, chunk_dur)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
                    clips.append({
                        "id":         f"{vid_id}_{idx:04d}",
                        "file":       out_path,
                        "start":      round(t, 2),
                        "end":        round(t + chunk_dur, 2),
                        "text":       _assign_text(transcript, t, t + chunk_dur, total_dur),
                        "source_url": source_url,
                    })
                    idx += 1
                t += chunk_dur

        if emit and (i + 1) % 10 == 0:
            emit("clips", f"{vid_id}: cut {i+1}/{total_scenes} scenes ({idx} clips so far)...")

    return clips


def _cut_fixed(src_path: str, pool_dir: str, vid_id: str,
               source_url: str, transcript: str, total_dur: float, emit=None) -> list:
    clip_min, clip_max = _clip_limits()
    clips = []
    t     = 0.0
    idx   = 0
    total_chunks = int(total_dur / clip_max)
    if emit:
        emit("clips", f"{vid_id}: cutting ~{total_chunks} clips (fixed)...")
    while t + clip_max <= total_dur:
        out_path = os.path.join(pool_dir, f"{vid_id}_{idx:04d}.mp4")
        _cut_clip(src_path, out_path, t, clip_max)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
            clips.append({
                "id":         f"{vid_id}_{idx:04d}",
                "file":       out_path,
                "start":      round(t, 2),
                "end":        round(t + clip_max, 2),
                "text":       _assign_text(transcript, t, t + clip_max, total_dur),
                "source_url": source_url,
            })
            idx += 1
        if emit and idx % 10 == 0 and idx > 0:
            emit("clips", f"{vid_id}: {idx}/{total_chunks} clips cut...")
        t += clip_max
    return clips


def download_and_cut(video_url: str, pool_dir: str, emit=None) -> list:
    """
    Download a YouTube video and cut into clips by scene boundaries.
    Each clip entry: {id, file, start, end, text, source_url}
    Skips re-download if index already exists.
    """
    def _log(msg):
        print(f"[downloader] {msg}", flush=True)
        if emit:
            emit("clips", msg)

    os.makedirs(pool_dir, exist_ok=True)

    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', video_url)
    vid_id = m.group(1) if m else re.sub(r'[^\w\-]', '_', video_url.split("/")[-1].split("?")[0])[:16]
    src_path = os.path.join(pool_dir, f"_src_{vid_id}.mp4")
    idx_path = os.path.join(pool_dir, f"{vid_id}_index.json")

    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            clips = json.load(f)
        clips = [c for c in clips if os.path.exists(c["file"])]
        _log(f"{vid_id}: already cut ({len(clips)} clips)")
        return clips

    _log(f"{vid_id}: fetching transcript...")
    transcript = _get_transcript(video_url)

    _log(f"{vid_id}: downloading video...")
    import platform as _platform
    dl_cmd = ["yt-dlp"]
    if _platform.system() != "Windows":
        dl_cmd += ["--remote-components", "ejs:github", "--js-runtimes", "node"]
    dl_cmd += [
        "-f", "bestvideo[height>=1080][ext=mp4]/bestvideo[height>=1080]/bestvideo[ext=mp4]/bestvideo",
        "--no-audio", "--no-playlist",
        "-o", src_path,
    ]
    dl_cmd += _cookies_arg(pool_dir)
    dl_cmd.append(video_url)
    try:
        dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600)
        if dl.returncode != 0:
            print(f"[downloader] yt-dlp error: {dl.stderr[-300:].strip()}", flush=True)
    except subprocess.TimeoutExpired:
        _log(f"{vid_id}: download timed out (10min limit)")
        return []

    # Remove cookie temp copies left by _cookies_arg
    for _f in os.listdir(pool_dir):
        if _f.startswith("cookies_") and _f.endswith(".txt"):
            try:
                os.unlink(os.path.join(pool_dir, _f))
            except Exception:
                pass

    if not os.path.exists(src_path):
        _log(f"{vid_id}: download failed")
        return []

    total = _get_duration(src_path)
    if total < 10:
        os.remove(src_path)
        return []

    _log(f"{vid_id}: detecting scenes ({total:.0f}s video)...")
    clips = _cut_by_scenes(src_path, pool_dir, vid_id, video_url, transcript, total, emit=emit)
    os.remove(src_path)

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    print(f"[downloader] {vid_id}: {len(clips)} clips cut", flush=True)
    return clips


def build_pool(youtube_urls: list, pool_dir: str, emit=None) -> tuple:
    """
    Download all URLs in parallel (2 at a time), cut clips by scenes, save combined index.
    Returns (clip_paths: list[str], clips_index: list[dict])
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    urls = [u.strip() for u in youtube_urls if u.strip()]
    results: dict = {}

    def _download_one(idx_url):
        idx, url = idx_url
        clips = download_and_cut(url, pool_dir, emit=emit)
        return idx, clips

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_download_one, (i, url)): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            try:
                idx, clips = future.result()
                results[idx] = clips
            except Exception as e:
                print(f"[downloader] ERROR downloading video: {e}", flush=True)

    # Preserve original order
    all_clips = []
    for idx in sorted(results):
        all_clips.extend(results[idx])

    index_path = os.path.join(pool_dir, "clips_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, ensure_ascii=False, indent=2)

    print(f"[downloader] Pool ready: {len(all_clips)} clips total", flush=True)
    return [c["file"] for c in all_clips], all_clips
