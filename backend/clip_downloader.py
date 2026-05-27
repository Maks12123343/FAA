import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE
COOKIES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookies.txt")

SCENE_THRESHOLD = 0.35  # 0-1: lower = more sensitive, more scene cuts detected


def _clip_limits() -> tuple:
    s = config.load_settings()
    return float(s.get("clip_min_duration", 2)), float(s.get("clip_max_duration", 5))


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
    r = subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "ultrafast", "-an", out],
        capture_output=True, timeout=60,
    )
    if r.returncode != 0:
        print(f"[downloader] ffmpeg cut error: {r.stderr[-200:].strip()}", flush=True)


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
    r = subprocess.run(
        [FFMPEG, "-i", video_path,
         "-vf", f"select=gt(scene\\,{SCENE_THRESHOLD}),showinfo",
         "-vsync", "vfr", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
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
                   source_url: str, transcript: str, total_dur: float) -> list:
    clip_min, clip_max = _clip_limits()
    scene_times = _detect_scene_timestamps(src_path)
    scene_times.append(total_dur)

    if len(scene_times) <= 2:
        print(f"[downloader] {vid_id}: no scene changes detected, using fixed cuts", flush=True)
        return _cut_fixed(src_path, pool_dir, vid_id, source_url, transcript, total_dur)

    clips = []
    idx   = 0

    for i in range(len(scene_times) - 1):
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

    return clips


def _cut_fixed(src_path: str, pool_dir: str, vid_id: str,
               source_url: str, transcript: str, total_dur: float) -> list:
    clip_min, clip_max = _clip_limits()
    clips = []
    t     = 0.0
    idx   = 0
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
        t += clip_max
    return clips


def download_and_cut(video_url: str, pool_dir: str) -> list:
    """
    Download a YouTube video and cut into clips by scene boundaries.
    Each clip entry: {id, file, start, end, text, source_url}
    Skips re-download if index already exists.
    """
    os.makedirs(pool_dir, exist_ok=True)

    vid_id   = video_url.split("v=")[-1].split("&")[0].split("/")[-1][:16]
    src_path = os.path.join(pool_dir, f"_src_{vid_id}.mp4")
    idx_path = os.path.join(pool_dir, f"{vid_id}_index.json")

    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            clips = json.load(f)
        clips = [c for c in clips if os.path.exists(c["file"])]
        print(f"[downloader] {vid_id}: already cut ({len(clips)} clips)", flush=True)
        return clips

    print(f"[downloader] {vid_id}: fetching transcript...", flush=True)
    transcript = _get_transcript(video_url)

    print(f"[downloader] Downloading: {video_url}", flush=True)
    dl_cmd = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "--no-audio", "--no-playlist",
        "-o", src_path,
    ]
    if os.path.exists(COOKIES_FILE):
        dl_cmd += ["--cookies", COOKIES_FILE]
    dl_cmd.append(video_url)
    dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600)
    if dl.returncode != 0:
        print(f"[downloader] yt-dlp error: {dl.stderr[-300:].strip()}", flush=True)

    if not os.path.exists(src_path):
        print(f"[downloader] Failed to download: {video_url}", flush=True)
        return []

    total = _get_duration(src_path)
    if total < 10:
        os.remove(src_path)
        return []

    print(f"[downloader] {vid_id}: detecting scenes ({total:.0f}s)...", flush=True)
    clips = _cut_by_scenes(src_path, pool_dir, vid_id, video_url, transcript, total)
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
        if emit:
            emit("clips", f"Downloading video {idx+1}/{len(urls)}...")
        clips = download_and_cut(url, pool_dir)
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
