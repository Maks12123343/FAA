import json
import os
import subprocess
import tempfile
import threading
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import torch
                import whisper
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # large-v3 needs ~10GB VRAM but gives the best Polish/German/Russian/etc.
                # Override via env var FAA_WHISPER_MODEL if you need to fall back.
                if device == "cuda":
                    model_name = os.environ.get("FAA_WHISPER_MODEL", "large-v3")
                else:
                    model_name = "base"
                print(f"[transcriber] Loading Whisper model ({model_name}) on {device}...", flush=True)
                _whisper_model = whisper.load_model(model_name, device=device)
    return _whisper_model


COOKIES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookies.txt")


def _cookies_arg(tmp_dir: str) -> list:
    """Return --cookies <tmpfile> args using a temp copy so yt-dlp can't overwrite the original."""
    if not os.path.exists(COOKIES_FILE):
        return []
    import shutil
    tmp = os.path.join(tmp_dir, "cookies_tmp.txt")
    shutil.copy2(COOKIES_FILE, tmp)
    return ["--cookies", tmp]


def _get_subtitles(url: str, tmp_dir: str) -> str | None:
    """Try to download auto-generated or manual subtitles via yt-dlp."""
    out_tpl = os.path.join(tmp_dir, "subs")
    cmd = [
        "yt-dlp", "--no-warnings", "--quiet",
        "--remote-components", "ejs:github",
        "--js-runtimes", "node",
        "--write-auto-subs", "--write-subs",
        "--sub-langs", "en,en-US,en-GB",
        "--sub-format", "json3",
        "--skip-download",
        "-o", out_tpl,
    ]
    cmd += _cookies_arg(tmp_dir)
    cmd.append(url)
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None

    for f in os.listdir(tmp_dir):
        if f.endswith(".json3"):
            return os.path.join(tmp_dir, f)
    return None


def _parse_json3(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            t = seg.get("utf8", "").strip()
            if t and t != "\n":
                parts.append(t)
    text = " ".join(parts)
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _whisper_transcribe(url: str, tmp_dir: str) -> str:
    """Download audio and transcribe with Whisper (auto-detects language)."""
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    dl_cmd = [
        "yt-dlp", "--no-warnings", "--quiet",
        "--remote-components", "ejs:github",
        "--js-runtimes", "node",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--extract-audio", "--audio-format", "mp3",
        "-o", audio_path,
    ]
    dl_cmd += _cookies_arg(tmp_dir)
    dl_cmd.append(url)
    try:
        subprocess.run(dl_cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Audio download timed out for {url}")
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio download failed for {url}")

    model = _get_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()


def _get_transcript_api(video_id: str) -> str | None:
    """Fetch transcript using youtube-transcript-api (no cookies needed)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(snippet.text for snippet in transcript.snippets)
        import re
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 200 else None
    except Exception as e:
        print(f"[transcriber] youtube-transcript-api failed: {e}", flush=True)
        return None


def _extract_video_id(url: str) -> str:
    import re
    m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else ""


def transcribe_segments(audio_path: str) -> list:
    """
    Transcribe a local audio file with Whisper and return timestamped segments.
    Returns list of {"text": str, "start": float, "end": float}.
    Uses cached model -- loads once, reuses across all languages.
    Sanitizes segments and merges short ones (< 2s), capped at 5s per merged segment.
    """
    import time as _time

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024) if os.path.exists(audio_path) else 0
    print(f"[transcriber] === WHISPER SEGMENTATION START ===", flush=True)
    print(f"[transcriber] File: {os.path.basename(audio_path)} ({file_size_mb:.1f} MB)", flush=True)

    print(f"[transcriber] Step 1/5: Loading Whisper model...", flush=True)
    t0 = _time.time()
    model = _get_whisper_model()
    print(f"[transcriber] Step 1/5: Model loaded in {_time.time()-t0:.1f}s", flush=True)

    print(f"[transcriber] Step 2/5: Running transcription (this may take a while on slow hardware)...", flush=True)
    t1 = _time.time()
    result = model.transcribe(audio_path, word_timestamps=False, task="transcribe")
    print(f"[transcriber] Step 2/5: Transcription done in {_time.time()-t1:.1f}s", flush=True)

    raw = result.get("segments", [])
    print(f"[transcriber] Step 3/5: Got {len(raw)} raw segments from Whisper", flush=True)
    if not raw:
        print(f"[transcriber] WARNING: Whisper returned 0 segments — audio may be silent or corrupted", flush=True)
        return []

    # Sanitize: drop segments with missing/invalid timestamps
    sanitized = []
    dropped = 0
    for seg in raw:
        try:
            start = float(seg["start"])
            end   = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if end <= start:
            dropped += 1
            continue
        sanitized.append({"text": str(seg.get("text", "")).strip(), "start": start, "end": end})

    if dropped:
        print(f"[transcriber] Step 3/5: Dropped {dropped} invalid segments", flush=True)

    if not sanitized:
        print(f"[transcriber] WARNING: All segments invalid after sanitization", flush=True)
        return []

    # Sort by start time
    sanitized.sort(key=lambda s: s["start"])
    total_dur = sanitized[-1]["end"] - sanitized[0]["start"]
    print(f"[transcriber] Step 4/5: {len(sanitized)} valid segments, covering {total_dur:.1f}s", flush=True)

    # Merge segments shorter than 2s into the previous one,
    # but cap merged segment at 5s to avoid huge gaps in montage.
    merged = [dict(sanitized[0])]
    for seg in sanitized[1:]:
        dur = seg["end"] - seg["start"]
        merged_dur = merged[-1]["end"] - merged[-1]["start"]
        if dur < 2.0 and merged_dur < 5.0:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(dict(seg))

    final = [{"text": s["text"].strip(), "start": s["start"], "end": s["end"]} for s in merged]
    print(f"[transcriber] Step 5/5: Merged into {len(final)} segments (2-5s each)", flush=True)
    print(f"[transcriber]   First: {final[0]['start']:.1f}-{final[0]['end']:.1f}s", flush=True)
    print(f"[transcriber]   Last:  {final[-1]['start']:.1f}-{final[-1]['end']:.1f}s", flush=True)
    print(f"[transcriber] === WHISPER SEGMENTATION DONE ===", flush=True)
    return final


def get_transcript(url: str, fallback_whisper: bool = True) -> dict:
    """
    Returns {"text": str, "source": "subtitles"|"whisper"|"transcript_api"}
    """
    # Method 1: youtube-transcript-api (no cookies, most reliable)
    video_id = _extract_video_id(url)
    if video_id:
        text = _get_transcript_api(video_id)
        if text:
            print(f"[transcriber] Transcript API OK ({len(text)} chars)", flush=True)
            return {"text": text, "source": "transcript_api"}

    # Method 2: yt-dlp subtitles
    with tempfile.TemporaryDirectory() as tmp:
        subs_path = _get_subtitles(url, tmp)
        if subs_path:
            text = _parse_json3(subs_path)
            if len(text) > 200:
                print(f"[transcriber] Subtitles OK ({len(text)} chars)", flush=True)
                return {"text": text, "source": "subtitles"}

        # Method 3: Whisper (requires audio download)
        if not fallback_whisper:
            raise RuntimeError("No transcript available and Whisper fallback disabled")

        print("[transcriber] No subtitles, falling back to Whisper...", flush=True)
        text = _whisper_transcribe(url, tmp)
        print(f"[transcriber] Whisper OK ({len(text)} chars)", flush=True)
        return {"text": text, "source": "whisper"}
