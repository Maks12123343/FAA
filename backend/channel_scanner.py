import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _ydl(args: list) -> dict:
    cmd = ["yt-dlp", "--no-warnings", "--quiet"] + args + ["--dump-json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return {}
    return json.loads(lines[0])


def _get_channel_videos(channel_url: str, max_videos: int = 15) -> list:
    cmd = [
        "yt-dlp", "--no-warnings", "--quiet",
        "--remote-components", "ejs:github",
        "--js-runtimes", "node",
        "--flat-playlist", "--playlist-end", str(max_videos),
        "--extractor-args", "youtubetab:approximate_date",
        "--print", "%(id)s\t%(title)s\t%(view_count)s\t%(upload_date)s\t%(duration)s",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[scanner] yt-dlp error: {result.stderr.strip()[:200]}", flush=True)
        return []
    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        vid_id, title, views, date = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""
        duration = parts[4] if len(parts) > 4 else "0"
        try:
            views = int(views)
        except (ValueError, TypeError):
            views = 0
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            duration = 0
        if views == 0 or str(views) == "NA":
            views = 0
        if str(date) == "NA":
            date = ""
        videos.append({
            "id": vid_id,
            "url": f"https://www.youtube.com/watch?v={vid_id}",
            "title": title,
            "views": views,
            "upload_date": date,
            "duration": duration,
            "channel_url": channel_url,
        })
    return videos


def _pick_long_enough(videos: list, min_seconds: int) -> dict | None:
    """Pick the top video that is at least min_seconds long. Fetches metadata for duration/views."""
    for v in videos:
        dur = v.get("duration", 0)
        if dur >= min_seconds:
            meta = get_video_metadata(v["url"])
            v["duration"] = meta.get("duration", dur)
            v["views"] = meta.get("view_count", meta.get("views", v.get("views", 0)))
            return v
        if dur > 0:
            print(f"[scanner] Skipping (too short: {dur//60:.0f}min): {v['title']}", flush=True)
            continue
        # duration unknown from flat-playlist — fetch metadata
        meta = get_video_metadata(v["url"])
        dur = meta.get("duration", 0)
        if dur >= min_seconds:
            v["duration"] = dur
            v["views"] = meta.get("view_count", meta.get("views", 0))
            return v
        print(f"[scanner] Skipping (too short: {dur//60:.0f}min): {v['title']}", flush=True)
    return None


def _matches_niche_keywords(title: str, niche: dict) -> bool:
    """Check if video title contains required keywords for this niche."""
    required = niche.get("title_keywords", [])
    if not required:
        return True
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in required)


def find_top_video(niche_path: str) -> dict:
    from datetime import datetime, timedelta

    with open(niche_path, "r", encoding="utf-8") as f:
        niche = json.load(f)

    channels = niche.get("channels", [])
    if not channels:
        raise ValueError("No channels in niche config")

    all_videos = []
    for ch_url in channels:
        try:
            videos = _get_channel_videos(ch_url, max_videos=15)
            all_videos.extend(videos)
        except Exception as e:
            print(f"[scanner] Failed {ch_url}: {e}", flush=True)

    if not all_videos:
        raise RuntimeError("Could not fetch videos from any channel.")

    # Filter by title keywords (e.g. must contain "China" / "Chinese")
    if niche.get("title_keywords"):
        filtered = [v for v in all_videos if _matches_niche_keywords(v["title"], niche)]
        skipped = len(all_videos) - len(filtered)
        if skipped:
            print(f"[scanner] Filtered out {skipped} videos (title doesn't match keywords)", flush=True)
        if not filtered:
            raise RuntimeError(f"No videos matching title_keywords: {niche['title_keywords']}")
        all_videos = filtered

    MIN_DURATION_SEC = 480  # 8 minutes minimum (script will be expanded if too short)

    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
    yesterday_videos = [v for v in all_videos if v.get("upload_date") == yesterday]

    if yesterday_videos:
        if any(v.get("views", 0) > 0 for v in yesterday_videos):
            yesterday_videos.sort(key=lambda v: v["views"], reverse=True)
        top = _pick_long_enough(yesterday_videos, MIN_DURATION_SEC)
        if top:
            print(f"[scanner] Top video (yesterday {yesterday}): {top['title']} ({top.get('views',0):,} views, {top.get('duration',0)//60:.0f}min)", flush=True)
            return top

    # Fallback: best video from last 7 days
    week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")
    recent = [v for v in all_videos if v.get("upload_date", "") >= week_ago]

    if not recent:
        raise RuntimeError(
            f"No videos from any channel in the last 7 days. "
            f"Checked {len(channels)} channel(s)."
        )

    if any(v.get("views", 0) > 0 for v in recent):
        recent.sort(key=lambda v: v["views"], reverse=True)
    top = _pick_long_enough(recent, MIN_DURATION_SEC)
    if not top:
        raise RuntimeError(f"No videos longer than {MIN_DURATION_SEC//60} minutes found in last 7 days.")
    print(f"[scanner] Top video: {top['title']} (date: {top['upload_date']}, {top.get('views',0):,} views, {top.get('duration',0)//60:.0f}min)", flush=True)
    return top


def get_video_metadata(url: str) -> dict:
    """
    Fetch full metadata for a video (title, description, tags, duration, views).
    Uses YouTube Data API v3 for reliability (yt-dlp gets blocked without cookies).
    """
    import re as _re
    from backend.youtube_api import yt_request

    vid_id = ""
    m = _re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if m:
        vid_id = m.group(1)
    if not vid_id:
        return {"url": url, "title": "", "description": "", "tags": [], "duration": 0, "views": 0}

    try:
        data = yt_request("videos", {
            "part": "snippet,contentDetails,statistics",
            "id": vid_id,
        })
        items = data.get("items", [])
        if not items:
            return {"url": url, "id": vid_id, "title": "", "description": "", "tags": [], "duration": 0, "views": 0}

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        # Parse ISO 8601 duration (PT1H2M3S)
        dur_str = content.get("duration", "")
        duration = _parse_iso_duration(dur_str)

        return {
            "url":         url,
            "id":          vid_id,
            "title":       snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "tags":        snippet.get("tags", []),
            "duration":    duration,
            "views":       int(stats.get("viewCount", 0)),
            "channel":     snippet.get("channelTitle", ""),
        }
    except Exception as e:
        print(f"[scanner] YouTube API metadata error: {e}", flush=True)
        return {"url": url, "id": vid_id, "title": "", "description": "", "tags": [], "duration": 0, "views": 0}


def _parse_iso_duration(dur_str: str) -> int:
    """Parse ISO 8601 duration like PT1H2M3S to seconds."""
    import re as _re
    if not dur_str:
        return 0
    m = _re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur_str)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s
