import json
import multiprocessing.pool
import os
import random
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
            capture_output=True, text=True, timeout=30,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _audio_duration(path: str) -> float:
    return _get_duration(path)


def _prepare_clip(clip_path: str, out_path: str, width: int = 1920, height: int = 1080,
                  fps: int = 30, max_duration: float = 6.0, action: str = "use",
                  crop_percent: int = 0):
    """
    Normalize clip: scale, pad, set fps, trim to max_duration, no audio.
    action=crop_bottom -- zoom based on Gemini-measured subtitle bar height.
    action=crop_corner -- zoom based on Gemini-measured watermark size.
    crop_percent -- exact % of frame the subtitle/logo occupies (from Gemini analysis).
    """
    if action == "crop_bottom":
        # Add 30% safety margin on top of measured size, clamp between 10% and 35%
        zoom = 1.0 + max(0.10, min(0.35, (crop_percent / 100) * 1.3))
        scaled_h = int(height * zoom)
        vf = (
            f"scale={width}:{scaled_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:0:0,"
            f"fps={fps}"
        )
    elif action == "crop_corner":
        # Add 40% safety margin, clamp between 6% and 25%
        zoom = 1.0 + max(0.06, min(0.25, (crop_percent / 100) * 1.4))
        scaled_w = int(width  * zoom)
        scaled_h = int(height * zoom)
        vf = (
            f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={fps}"
        )
    else:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={fps}"
        )

    subprocess.run(
        [FFMPEG, "-y", "-i", clip_path,
         "-vf", vf,
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
         "-t", str(max_duration), out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    )


def _get_clip_action(clip_path: str) -> tuple:
    """Returns (action, crop_percent) from cached Gemini analysis."""
    ap = clip_path + ".analysis.json"
    if os.path.exists(ap):
        try:
            with open(ap, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("action", "use"), int(data.get("crop_percent", 0))
        except Exception:
            pass
    return "use", 0


def _process_one_clip(args):
    i, item, tmp, width, height, fps, clip_max, uniq_params = args
    if isinstance(item, tuple):
        cp, use_dur = item
    else:
        cp, use_dur = item, clip_max

    action, crop_pct = _get_clip_action(cp)
    norm_p   = os.path.join(tmp, f"norm_{i:04d}.mp4")
    unique_p = os.path.join(tmp, f"clip_{i:04d}.mp4")

    try:
        _prepare_clip(cp, norm_p, width, height, fps, max_duration=use_dur,
                      action=action, crop_percent=crop_pct)
    except Exception:
        return i, None
    if not (os.path.exists(norm_p) and os.path.getsize(norm_p) > 1000):
        return i, None

    try:
        ok = _uniqualize_clip(norm_p, unique_p, uniq_params)
    except Exception:
        ok = False
    return i, (unique_p if ok and os.path.exists(unique_p) else norm_p)


def _concat_clip_list(clip_paths: list, output: str):
    """Concat pre-normalized clips with hard cuts using concat demuxer."""
    list_file = output + ".txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            safe_p = p.replace("\\", "/")
            f.write(f"file '{safe_p}'\n")
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an", output],
            capture_output=True, timeout=3600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{r.stderr.decode(errors='replace')[-1000:]}")
    finally:
        if os.path.exists(list_file):
            os.unlink(list_file)


_XFADE_TRANSITIONS = [
    "fade", "fadeblack", "dissolve", "hblur",
    "fadegrays", "smoothleft", "smoothright", "fadewhite",
]


def _xfade_join(segment_files: list, output: str, fade_dur: float = 0.35):
    """
    Join segment files with varied crossfade transitions between them.
    Falls back to concat if any segment is too short for xfade.
    """
    n = len(segment_files)
    if n == 1:
        import shutil
        shutil.copy2(segment_files[0], output)
        return

    durations = [_get_duration(s) for s in segment_files]
    if any(d <= 0 for d in durations):
        _concat_clip_list(segment_files, output)
        return
    # Guard: if any segment is too short for a crossfade, fall back to simple concat
    if any(d <= fade_dur * 2 for d in durations):
        print(
            f"[montage] Segment too short for xfade (min={min(durations):.2f}s, "
            f"fade_dur={fade_dur}s) -- using hard cuts",
            flush=True,
        )
        _concat_clip_list(segment_files, output)
        return

    inputs = []
    for s in segment_files:
        inputs += ["-i", s]

    filters = []
    cumulative = 0.0
    prev_label = "0:v"

    for i in range(1, n):
        cumulative += durations[i - 1] - fade_dur
        out_label = f"x{i}" if i < n - 1 else "vout"
        offset = max(0.0, cumulative)
        transition = _XFADE_TRANSITIONS[(i - 1) % len(_XFADE_TRANSITIONS)]
        filters.append(
            f"[{prev_label}][{i}:v]xfade=transition={transition}"
            f":duration={fade_dur:.2f}:offset={offset:.3f}[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(filters)

    r = subprocess.run(
        [FFMPEG, "-y"] + inputs +
        ["-filter_complex", filter_complex,
         "-map", "[vout]",
         "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an", output],
        capture_output=True, timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg xfade failed:\n{r.stderr.decode(errors='replace')[-1000:]}")


def _build_concat(clip_items: list, output: str, width: int, height: int, fps: int,
                  uniq_params: dict = None):
    import multiprocessing
    settings  = config.load_settings()
    clip_max  = float(settings.get("clip_max_duration", 6))
    workers   = min(multiprocessing.cpu_count(), 2)

    GROUP_MIN = 3
    GROUP_MAX = 6
    FADE_DUR  = 0.35

    with tempfile.TemporaryDirectory() as tmp:
        args = [(i, item, tmp, width, height, fps, clip_max, uniq_params)
                for i, item in enumerate(clip_items)]

        results = {}
        with multiprocessing.pool.ThreadPool(processes=workers) as pool:
            for i, path in pool.imap_unordered(_process_one_clip, args):
                results[i] = path

        prepared = [results[i] for i in range(len(clip_items))
                    if results.get(i) is not None]

        if not prepared:
            raise RuntimeError("No clips prepared")

        groups = []
        i = 0
        while i < len(prepared):
            size = random.randint(GROUP_MIN, GROUP_MAX)
            groups.append(prepared[i:i + size])
            i += size

        if len(groups) <= 1:
            _concat_clip_list(prepared, output)
            return

        segment_files = []
        for g_idx, group in enumerate(groups):
            seg_path = os.path.join(tmp, f"seg_{g_idx:04d}.mp4")
            _concat_clip_list(group, seg_path)
            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 1000:
                segment_files.append(seg_path)

        if not segment_files:
            raise RuntimeError("No segments created")

        print(f"[montage] Applying crossfade between {len(segment_files)} segments...", flush=True)
        _xfade_join(segment_files, output, FADE_DUR)


def _fetch_bg_music(duration: float, project_dir: str) -> str | None:
    """Download a background music track from Pixabay matching the video duration."""
    import requests as _requests

    settings = config.load_settings()
    api_key = settings.get("pixabay_api_key", "")
    if not api_key:
        return None

    music_path = os.path.join(project_dir, "bg_music.mp3")
    if os.path.exists(music_path):
        return music_path

    try:
        r = _requests.get(
            "https://pixabay.com/api/",
            params={
                "key": api_key,
                "media_type": "music",
                "q": "cinematic ambient documentary",
                "per_page": 10,
                "min_duration": int(duration) - 30,
                "order": "popular",
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[montage] Pixabay music API error: {r.status_code}", flush=True)
            return None

        hits = r.json().get("hits", [])
        if not hits:
            r = _requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": api_key,
                    "media_type": "music",
                    "q": "ambient background",
                    "per_page": 10,
                    "order": "popular",
                },
                timeout=15,
            )
            hits = r.json().get("hits", []) if r.status_code == 200 else []

        if not hits:
            return None

        track = random.choice(hits[:5])
        dl_url = track.get("audio", "") or track.get("previewURL", "")
        if not dl_url:
            return None

        print(f"[montage] Downloading bg music: {track.get('tags', '')[:40]}", flush=True)
        resp = _requests.get(dl_url, timeout=60)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type or "text" in content_type:
                print(f"[montage] Pixabay returned non-audio content ({content_type}), skipping", flush=True)
                return None
            with open(music_path, "wb") as f:
                f.write(resp.content)
            # Validate with ffprobe: must have real audio and duration > 5s
            music_dur = _get_duration(music_path)
            if music_dur < 5.0:
                print(f"[montage] Downloaded music has bad duration ({music_dur:.1f}s), discarding", flush=True)
                try:
                    os.unlink(music_path)
                except Exception:
                    pass
                return None
            return music_path
    except Exception as e:
        print(f"[montage] Bg music fetch failed: {e}", flush=True)

    return None


def _add_audio(video_path: str, audio_path: str, output: str, music_path: str = None):
    audio_dur = _audio_duration(audio_path)

    if music_path and os.path.exists(music_path):
        r = subprocess.run(
            [FFMPEG, "-y",
             "-i", video_path,
             "-i", audio_path,
             "-i", music_path,
             "-filter_complex",
             f"[1:a]volume=1.0[voice];"
             f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{audio_dur},volume=0.02[music];"
             f"[voice][music]amix=inputs=2:duration=shortest[out]",
             "-map", "0:v", "-map", "[out]",
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k",
             "-t", str(audio_dur),
             "-movflags", "+faststart",
             output],
            capture_output=True, timeout=3600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg add_audio (music) failed:\n{r.stderr.decode(errors='replace')[-1000:]}")
    else:
        r = subprocess.run(
            [FFMPEG, "-y",
             "-i", video_path,
             "-i", audio_path,
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k",
             "-t", str(audio_dur),
             "-shortest",
             "-movflags", "+faststart",
             output],
            capture_output=True, timeout=3600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg add_audio failed:\n{r.stderr.decode(errors='replace')[-1000:]}")


def _uniqualize_clip(input_path: str, output_path: str, params: dict):
    """Apply uniqualization to a single clip using shared video-level params."""
    grain = random.uniform(4, 10)

    vf = (
        f"scale=iw*{params['zoom']:.4f}:ih*{params['zoom']:.4f},"
        f"crop=1920:1080,"
        f"eq=brightness={params['brightness']:.3f}"
        f":contrast={params['contrast']:.3f}"
        f":saturation={params['saturation']:.3f},"
        f"noise=alls={grain:.1f}:allf=t+u"
    )

    r = subprocess.run(
        [FFMPEG, "-y", "-i", input_path,
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-an",
         output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    )
    return r.returncode == 0


def assemble(
    clips: list,
    audio_path: str,
    output_path: str,
    text_overlays: list | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> str:
    """
    clips        -- list of video file paths (or (path, duration) tuples)
    audio_path   -- MP3 voiceover
    output_path  -- final output path
    text_overlays -- list of {"text": str, "start": float, "duration": float, "position": str}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    uniq_params = {
        "zoom":       random.uniform(1.02, 1.04),
        "brightness": random.uniform(-0.03, 0.03),
        "contrast":   random.uniform(0.97, 1.04),
        "saturation": random.uniform(0.95, 1.08),
    }

    project_dir = os.path.dirname(output_path)
    raw_video  = os.path.join(project_dir, "_raw_video.mp4")
    with_audio = os.path.join(project_dir, "_with_audio.mp4")

    def _valid_intermediate(path: str, min_size: int = 100_000, min_dur: float = 1.0) -> bool:
        """Return True only if the intermediate file exists, is large enough, and has real duration."""
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < min_size:
            return False
        return _get_duration(path) >= min_dur

    if not _valid_intermediate(raw_video):
        if os.path.exists(raw_video):
            print("[montage] raw_video exists but is corrupt/too small -- rebuilding...", flush=True)
            os.remove(raw_video)
        print(f"[montage] Building video from {len(clips)} clips...", flush=True)
        _build_concat(clips, raw_video, width, height, fps, uniq_params=uniq_params)
    else:
        print("[montage] raw_video cached, skipping concat...", flush=True)

    if not _valid_intermediate(with_audio):
        if os.path.exists(with_audio):
            print("[montage] with_audio exists but is corrupt/too small -- rebuilding...", flush=True)
            os.remove(with_audio)
        audio_dur = _audio_duration(audio_path)
        print("[montage] Fetching background music...", flush=True)
        music_path = _fetch_bg_music(audio_dur, project_dir)
        print("[montage] Adding audio...", flush=True)
        _add_audio(raw_video, audio_path, with_audio, music_path=music_path)
    else:
        print("[montage] with_audio cached, skipping...", flush=True)

    if text_overlays:
        print(f"[montage] Adding {len(text_overlays)} text overlays...", flush=True)
        from backend.text_renderer import apply_text_overlays
        apply_text_overlays(with_audio, text_overlays, output_path)
    else:
        import shutil
        shutil.copy2(with_audio, output_path)

    print(f"[montage] Done: {output_path}", flush=True)
    return output_path


def pick_clips(validated_clips: list, target_duration: float, min_dur: float = 2, max_dur: float = 5) -> list:
    """Pick random clips to fill target_duration seconds."""
    if not validated_clips:
        raise RuntimeError("No validated clips available")

    pool = list(validated_clips)
    random.shuffle(pool)
    selected = []
    total = 0.0

    while total < target_duration:
        if not pool:
            pool = list(validated_clips)
            random.shuffle(pool)
        clip = pool.pop()
        dur  = _get_duration(clip)
        if dur < min_dur:
            continue
        clip_use = min(dur, max_dur)
        selected.append(clip)
        total += clip_use
        if total >= target_duration * 1.05:
            break

    return selected
