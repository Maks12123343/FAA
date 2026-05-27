import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

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
         "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-an", out],
        capture_output=True, timeout=120,
    )


# ── YouTube search ────────────────────────────────────────────────────────────

def _youtube_search(query: str, max_results: int = 6) -> list:
    from backend.youtube_api import yt_request
    try:
        data = yt_request("search", {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoDuration": "medium",
            "videoCaption": "closedCaption",
            "maxResults": max_results,
        })
    except Exception as e:
        print(f"[sourcer] YouTube search error: {e}", flush=True)
        return []
    return [
        {
            "id":    it["id"]["videoId"],
            "url":   f"https://www.youtube.com/watch?v={it['id']['videoId']}",
            "title": it["snippet"]["title"],
        }
        for it in data.get("items", [])
    ]


# ── Transcript extraction ─────────────────────────────────────────────────────

def _get_transcript(video_url: str) -> str | None:
    """
    Download auto-generated subtitles with yt-dlp (json3 format).
    Returns timestamped text like "[12.3s] some text", or None if unavailable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_tpl = os.path.join(tmp, "sub")
        res = subprocess.run(
            ["yt-dlp", "--skip-download",
             "--write-auto-sub", "--sub-lang", "en",
             "--sub-format", "json3",
             "--output", out_tpl,
             video_url],
            capture_output=True, text=True, timeout=60,
        )

        sub_files = [f for f in os.listdir(tmp) if f.endswith(".json3")]
        if not sub_files:
            return None

        with open(os.path.join(tmp, sub_files[0]), encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for event in data.get("events", []):
            if "segs" not in event:
                continue
            text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
            if not text or text == "\n":
                continue
            t = event.get("tStartMs", 0) / 1000.0
            segments.append(f"[{t:.1f}s] {text}")

        return "\n".join(segments) if segments else None


# ── Gemini timestamp analysis ─────────────────────────────────────────────────

def _gemini_find_timestamps(transcript: str, section_text: str) -> list:
    """
    Ask Gemini to find best timestamps in transcript that match section_text topic.
    Returns list of {"start": float, "duration": float}.
    """
    from google import genai

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()

    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("gemini_model", "gemini-2.5-flash")

    # Trim transcript to avoid token overflow
    transcript_cut = transcript[:6000] if len(transcript) > 6000 else transcript

    prompt = f"""You are selecting B-roll footage moments for a documentary.

VOICEOVER TEXT (what is being said):
"{section_text}"

VIDEO TRANSCRIPT (with timestamps):
{transcript_cut}

Find 2-3 moments in the transcript where the VIDEO TOPIC matches the voiceover topic.
Look for timestamps where similar concepts, facts, or subjects are discussed.

Return ONLY a JSON array (no explanation):
[
  {{"start": 45.2, "duration": 3.5}},
  {{"start": 120.0, "duration": 4.0}}
]

Rules:
- start = timestamp in seconds from transcript
- duration = 2.0 to 5.0 seconds
- Spread timestamps across the video
- Return [] if no match"""

    try:
        r = client.models.generate_content(model=model, contents=prompt)
        text = r.text.strip()
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []
    except Exception as e:
        print(f"[sourcer] Gemini timestamp error: {e}", flush=True)
        return []


# ── Segment download ──────────────────────────────────────────────────────────

def _get_video_url(video_url: str) -> str | None:
    """Get direct video stream URL from yt-dlp (no download)."""
    res = subprocess.run(
        ["yt-dlp", "-g",
         "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
         "--no-playlist", video_url],
        capture_output=True, text=True, timeout=30,
    )
    url = res.stdout.strip().split("\n")[0]
    return url if url.startswith("http") else None


def _download_segment(video_url: str, start: float, duration: float, out_path: str) -> bool:
    """Get direct URL then ffmpeg-cut the segment at exact timestamps."""
    direct_url = _get_video_url(video_url)
    if not direct_url:
        return False

    subprocess.run(
        [FFMPEG, "-y",
         "-ss", f"{start:.3f}",
         "-i", direct_url,
         "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-an", out_path],
        capture_output=True, timeout=120,
    )
    return os.path.exists(out_path) and os.path.getsize(out_path) > 10_000


# ── Main sourcing function ────────────────────────────────────────────────────

def source_clips_for_section(
    section_text: str,
    section_idx: int,
    project_dir: str,
    n_clips: int = 4,
) -> list:
    """
    Search YouTube → extract transcripts → Gemini finds timestamps → download segments.
    Returns list of downloaded clip paths.
    """
    from backend.youtube_api import _get_keys
    if not _get_keys():
        print("[sourcer] No YouTube API key configured — skipping", flush=True)
        return []

    query = " ".join(section_text.split()[:10])
    print(f"[sourcer] Section {section_idx}: '{query[:60]}'", flush=True)

    videos = _youtube_search(query, max_results=6)
    if not videos:
        return []

    clips_dir = os.path.join(project_dir, "clips", f"s{section_idx:03d}")
    os.makedirs(clips_dir, exist_ok=True)

    collected = []

    for video in videos:
        if len(collected) >= n_clips:
            break

        print(f"[sourcer]   → {video['title'][:50]}", flush=True)

        transcript = _get_transcript(video["url"])
        if not transcript:
            print("[sourcer]     no transcript", flush=True)
            continue

        timestamps = _gemini_find_timestamps(transcript, section_text)
        if not timestamps:
            print("[sourcer]     no matching timestamps", flush=True)
            continue

        for i, ts in enumerate(timestamps):
            if len(collected) >= n_clips:
                break

            start    = float(ts.get("start", 0))
            duration = max(2.0, min(5.0, float(ts.get("duration", 3.0))))
            out_name = f"{video['id']}_{i:02d}.mp4"
            out_path = os.path.join(clips_dir, out_name)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
                collected.append(out_path)
                continue

            print(f"[sourcer]     t={start:.1f}s dur={duration:.1f}s", flush=True)
            if _download_segment(video["url"], start, duration, out_path):
                collected.append(out_path)

            time.sleep(0.3)

    print(f"[sourcer] Section {section_idx}: {len(collected)} clips", flush=True)
    return collected


# ── Legacy compatibility ───────────────────────────────────────────────────────

def fetch_and_validate(niche_name: str, keywords: list, description: str) -> dict:
    """
    Called by /api/library/fetch_stocks — now triggers stock folder analysis.
    """
    from backend.stocks_library import scan_and_analyze
    result = scan_and_analyze()
    total = result.get("analyzed_new", 0) + result.get("already_done", 0)
    return {
        "fetched": result.get("analyzed_new", 0),
        "passed":  total,
        "failed":  0,
    }
