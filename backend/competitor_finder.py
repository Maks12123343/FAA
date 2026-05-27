import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend.youtube_api import yt_request as _yt

_ANALYSIS_MODEL = "gemini-2.5-pro"
_SCORE_BATCH_SIZE = 15


def _gemini(prompt: str, retries: int = 3) -> str:
    from google import genai
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("competitor_gemini_model", _ANALYSIS_MODEL)

    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            return r.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"[competitors] Rate limit, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    return ""


# ── Channel info ───────────────────────────────────────────────────────────────

def _channel_id_from_url(url: str) -> str:
    handle_m = re.search(r"@([\w\-]+)", url)
    if handle_m:
        data = _yt("channels", {"part": "id", "forHandle": handle_m.group(0)})
    elif "/channel/" in url:
        return url.split("/channel/")[1].split("/")[0].split("?")[0]
    else:
        data = _yt("channels", {"part": "id", "forUrl": url})
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"Channel not found: {url}")
    return items[0]["id"]


def _get_channel_info(channel_id: str) -> dict:
    data = _yt("channels", {"part": "snippet,statistics,contentDetails", "id": channel_id})
    item = data["items"][0]
    return {
        "id":               channel_id,
        "title":            item["snippet"]["title"],
        "description":      item["snippet"].get("description", ""),
        "subscribers":      int(item["statistics"].get("subscriberCount", 0)),
        "views":            int(item["statistics"].get("viewCount", 0)),
        "uploads_playlist": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def _get_recent_video_ids(uploads_playlist: str, max_results: int = 15) -> list:
    """Used only for seed channel analysis."""
    data = _yt("playlistItems", {
        "part": "contentDetails",
        "playlistId": uploads_playlist,
        "maxResults": max_results,
    })
    return [it["contentDetails"]["videoId"] for it in data.get("items", [])]


def _get_playlist_with_dates(uploads_playlist: str, max_results: int = 30) -> list:
    """Fetch recent videos with publish dates — used for activity filtering."""
    data = _yt("playlistItems", {
        "part": "contentDetails",
        "playlistId": uploads_playlist,
        "maxResults": max_results,
    })
    result = []
    for it in data.get("items", []):
        cd = it["contentDetails"]
        result.append({
            "id":        cd["videoId"],
            "published": cd.get("videoPublishedAt", ""),
        })
    return result


def _get_videos_details(video_ids: list) -> list:
    if not video_ids:
        return []
    results = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        data = _yt("videos", {"part": "snippet,statistics", "id": ",".join(batch)})
        for it in data.get("items", []):
            s = it["snippet"]
            results.append({
                "id":      it["id"],
                "channel": s.get("channelId", ""),
                "title":   s.get("title", ""),
                "views":   int(it["statistics"].get("viewCount", 0)),
            })
    return results


def _activity_stats(playlist_videos: list, video_views: dict, days: int = 30) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = 0
    views = 0
    for pv in playlist_videos:
        pub = pv.get("published", "")
        if not pub:
            continue
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if dt >= cutoff:
                count += 1
                views += video_views.get(pv["id"], 0)
        except Exception:
            pass
    return {"videos": count, "views": views}


def _get_featured_channels(channel_id: str) -> list:
    """Return list of channel IDs featured by this channel (from brandingSettings)."""
    try:
        data = _yt("channels", {"part": "brandingSettings", "id": channel_id})
        items = data.get("items", [])
        if not items:
            return []
        raw_urls = items[0].get("brandingSettings", {}).get("channel", {}).get("featuredChannelsUrls", [])
        result = []
        for url in raw_urls:
            if "/channel/" in url:
                cid = url.split("/channel/")[1].split("/")[0].split("?")[0]
                result.append(cid)
            elif re.match(r"^UC[\w\-]{20,}$", url):
                result.append(url)
            elif "@" in url:
                try:
                    full = url if url.startswith("http") else f"https://www.youtube.com/{url}"
                    result.append(_channel_id_from_url(full))
                except Exception:
                    pass
        return result
    except Exception:
        return []


# ── Step 1: Niche profile via Gemini Pro ──────────────────────────────────────

def _define_niche_profile(channel_info: dict, videos: list) -> dict:
    videos_text = "\n".join(f"- {v['title']}" for v in videos[:15])
    prompt = f"""Analyze this YouTube channel and define its EXACT niche with precision.

Channel: {channel_info['title']}
Subscribers: {channel_info['subscribers']:,}
Description: {channel_info['description'][:600]}
Recent videos:
{videos_text}

Tasks:
1. Define the exact sub-niche (very specific — not "finance" but e.g. "Chinese economic and geopolitical analysis aimed at English-speaking Western audience, documentary-style narration, 8-15 min videos")
2. List 4-6 criteria a channel MUST satisfy to be a true competitor
3. List 4-6 automatic disqualifiers (wrong language, wrong topic, entertainment mix, etc.)
4. Generate 18-20 YouTube search queries to find channels in this exact niche (vary angles, keywords, synonyms)

Return ONLY valid JSON, no extra text:
{{
  "niche_definition": "...",
  "must_have": ["...", "..."],
  "disqualifiers": ["...", "..."],
  "search_queries": ["...", "..."]
}}"""

    raw = _gemini(prompt)
    text = re.sub(r"^```(?:json)?\s*", "", raw)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {
        "niche_definition": channel_info["title"],
        "must_have": [],
        "disqualifiers": [],
        "search_queries": [channel_info["title"]],
    }


# ── Step 2: Search YouTube ────────────────────────────────────────────────────

def _search_channels_by_query(query: str, max_results: int = 25) -> list:
    data = _yt("search", {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": max_results,
        "order": "relevance",
    })
    seen = set()
    channels = []
    for it in data.get("items", []):
        cid = it["snippet"]["channelId"]
        if cid not in seen:
            seen.add(cid)
            channels.append({"id": cid, "title": it["snippet"]["channelTitle"]})
    return channels


def _search_channels_by_videos(queries: list, seed_id: str = "", max_per_query: int = 50) -> dict:
    """
    Search by video type, return {channel_id: hit_count} across all queries.
    Channels appearing in multiple queries = much stronger candidate signal.
    """
    counts = {}
    for query in queries:
        try:
            data = _yt("search", {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_per_query,
                "order": "relevance",
            })
            seen_in_query = set()
            for it in data.get("items", []):
                cid = it["snippet"]["channelId"]
                if cid != seed_id and cid not in seen_in_query:
                    seen_in_query.add(cid)
                    counts[cid] = counts.get(cid, 0) + 1
        except Exception:
            pass
        time.sleep(3)
    return counts


def _batch_channel_stats(channel_ids: list) -> dict:
    result = {}
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        data = _yt("channels", {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(batch),
        })
        for it in data.get("items", []):
            cid = it["id"]
            result[cid] = {
                "id":               cid,
                "title":            it["snippet"]["title"],
                "description":      it["snippet"].get("description", "")[:500],
                "subscribers":      int(it["statistics"].get("subscriberCount", 0)),
                "views":            int(it["statistics"].get("viewCount", 0)),
                "url":              f"https://www.youtube.com/channel/{cid}",
                "uploads_playlist": it["contentDetails"]["relatedPlaylists"]["uploads"],
            }
    return result


# ── Step 3: Strict Gemini Pro scoring (batched) ───────────────────────────────

def _score_batch(
    channels: dict,
    videos_by_channel: dict,
    activity_by_channel: dict,
    niche_profile: dict,
) -> dict:
    channels_text = ""
    for cid, info in channels.items():
        recent = videos_by_channel.get(cid, [])
        act = activity_by_channel.get(cid, {})
        avg_views = sum(v["views"] for v in recent) // max(len(recent), 1)
        titles = " | ".join(v["title"] for v in recent[:5]) or "no recent videos"
        channels_text += (
            f"\nID: {cid}\n"
            f"Name: {info['title']}\n"
            f"Subs: {info['subscribers']:,} | Avg views/video: {avg_views:,}\n"
            f"Last 30 days: {act.get('videos', 0)} videos, ~{act.get('views', 0):,} views\n"
            f"Description: {info['description'][:300]}\n"
            f"Recent titles: {titles}\n"
        )

    must_text = "\n".join(f"  - {c}" for c in niche_profile.get("must_have", []))
    disq_text = "\n".join(f"  - {d}" for d in niche_profile.get("disqualifiers", []))

    prompt = f"""You are a strict YouTube niche analyst. Evaluate each channel as a potential competitor.

TARGET NICHE:
{niche_profile.get("niche_definition", "")}

MUST-HAVE (channel must satisfy ALL to score above 0.70):
{must_text}

AUTOMATIC DISQUALIFIERS (any single one = score 0.0, no exceptions):
{disq_text}

SCORING:
0.90-1.0  = nearly identical — same topic, audience, format
0.80-0.89 = clear competitor — same niche, minor differences
0.70-0.79 = same broad topic but different angle or audience
0.00-0.69 = does NOT qualify

STRICT RULES:
- If not confident it passes ALL must-have criteria → score below 0.70.
- One disqualifier = 0.0, no exceptions.
- Your reason must explain WHY, especially for scores below 0.70.

CHANNELS:
{channels_text}

Return ONLY valid JSON:
{{
  "CHANNEL_ID": {{"score": 0.93, "reason": "Same niche: Chinese economy for Western English audience, documentary format", "active": true}},
  ...
}}"""

    raw = _gemini(prompt)
    text = re.sub(r"^```(?:json)?\s*", "", raw)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def _batch_score_channels(
    channels: dict,
    videos_by_channel: dict,
    activity_by_channel: dict,
    niche_profile: dict,
) -> dict:
    """Score all channels in batches to keep Gemini prompts focused and accurate."""
    all_scores = {}
    channel_ids = list(channels.keys())
    for i in range(0, len(channel_ids), _SCORE_BATCH_SIZE):
        batch_ids = channel_ids[i:i + _SCORE_BATCH_SIZE]
        batch = {cid: channels[cid] for cid in batch_ids}
        scores = _score_batch(batch, videos_by_channel, activity_by_channel, niche_profile)
        all_scores.update(scores)
    return all_scores


def _fetch_activity_for_channels(channel_stats: dict) -> tuple[dict, dict]:
    """
    Fetch playlist entries + video views for a set of channels.
    Returns (videos_by_channel, activity_by_channel).
    """
    playlist_entries = {}
    all_video_ids = []
    playlist_map = {}

    for cid, info in channel_stats.items():
        try:
            entries = _get_playlist_with_dates(info["uploads_playlist"], max_results=30)
            playlist_entries[cid] = entries
            for e in entries:
                playlist_map[e["id"]] = cid
            all_video_ids.extend(e["id"] for e in entries)
        except Exception:
            playlist_entries[cid] = []

    video_details = _get_videos_details(all_video_ids)
    video_views = {v["id"]: v["views"] for v in video_details}

    videos_by_channel: dict = {}
    for v in video_details:
        cid = playlist_map.get(v["id"], v.get("channel", ""))
        if cid:
            videos_by_channel.setdefault(cid, []).append(v)

    activity_by_channel = {
        cid: _activity_stats(playlist_entries.get(cid, []), video_views)
        for cid in channel_stats
    }
    return videos_by_channel, activity_by_channel


# ── Main ──────────────────────────────────────────────────────────────────────

def find_competitors(
    seed_url: str,
    min_score: float = 0.90,
    min_subs: int = 8_000,
    max_subs: int = 200_000,
    min_videos_month: int = 15,
    min_views_month: int = 30_000,
    emit=None,
) -> list:
    def log(msg):
        print(f"[competitors] {msg}", flush=True)
        if emit:
            emit(msg)

    # 1. Analyze seed channel
    log("Resolving seed channel...")
    seed_id = _channel_id_from_url(seed_url)
    seed_info = _get_channel_info(seed_id)
    log(f"Seed: {seed_info['title']} ({seed_info['subscribers']:,} subs)")

    log("Fetching seed videos...")
    seed_video_ids = _get_recent_video_ids(seed_info["uploads_playlist"], max_results=15)
    seed_videos = _get_videos_details(seed_video_ids)

    log("Gemini Pro — defining exact niche profile...")
    niche_profile = _define_niche_profile(seed_info, seed_videos)
    log(f"Niche: {niche_profile.get('niche_definition', '')[:120]}")
    queries = niche_profile.get("search_queries", [seed_info["title"]])
    log(f"Generated {len(queries)} search queries")

    # 2. Search YouTube — channel search + video-based search (union)
    log("Searching YouTube by channel names...")
    found: dict = {}
    for query in queries[:20]:
        log(f"  [ch] '{query}'")
        for ch in _search_channels_by_query(query, max_results=50):
            if ch["id"] != seed_id:
                found[ch["id"]] = ch
        time.sleep(3)

    log("Searching YouTube by video content (aggregating channels)...")
    video_hit_counts = _search_channels_by_videos(queries[:20], seed_id=seed_id, max_per_query=50)
    new_from_video = sum(1 for cid in video_hit_counts if cid not in found)
    for cid, count in video_hit_counts.items():
        if cid not in found:
            found[cid] = {"id": cid, "title": ""}
    log(f"  +{new_from_video} new channels from video search ({len(video_hit_counts)} total hits)")

    # Featured channels of seed — direct competitor signals from channel owner
    log("Fetching seed's featured channels...")
    featured_ids = _get_featured_channels(seed_id)
    new_from_featured = sum(1 for fid in featured_ids if fid != seed_id and fid not in found)
    for fid in featured_ids:
        if fid != seed_id and fid not in found:
            found[fid] = {"id": fid, "title": ""}
    log(f"  +{new_from_featured} from featured channels ({len(featured_ids)} listed)")

    if not found:
        log("No channels found.")
        return []

    log(f"Total unique candidates: {len(found)} — fetching stats...")
    channel_stats = _batch_channel_stats(list(found.keys()))

    # Filter 1: subscriber range
    before = len(channel_stats)
    channel_stats = {
        cid: info for cid, info in channel_stats.items()
        if min_subs <= info["subscribers"] <= max_subs and cid != seed_id
    }
    log(f"After subs filter ({min_subs:,}–{max_subs:,}): {len(channel_stats)} (removed {before - len(channel_stats)})")

    if not channel_stats:
        log("No channels left after subscriber filter.")
        return []

    # Fetch activity data
    log("Fetching recent videos for activity check...")
    videos_by_channel, activity_by_channel = _fetch_activity_for_channels(channel_stats)

    # Filter 2: minimum videos per month
    before = len(channel_stats)
    channel_stats = {
        cid: info for cid, info in channel_stats.items()
        if activity_by_channel.get(cid, {}).get("videos", 0) >= min_videos_month
    }
    log(f"After activity filter (≥{min_videos_month} videos/month): {len(channel_stats)} (removed {before - len(channel_stats)})")

    # Filter 3: minimum monthly views
    before = len(channel_stats)
    channel_stats = {
        cid: info for cid, info in channel_stats.items()
        if activity_by_channel.get(cid, {}).get("views", 0) >= min_views_month
    }
    log(f"After views filter (≥{min_views_month:,} views/month): {len(channel_stats)} (removed {before - len(channel_stats)})")

    if not channel_stats:
        log("No channels left after activity/views filters.")
        return []

    # 4. Strict Gemini Pro scoring in batches of _SCORE_BATCH_SIZE
    log(f"Gemini Pro — scoring {len(channel_stats)} channels in batches of {_SCORE_BATCH_SIZE} (threshold: {min_score})...")
    scores = _batch_score_channels(channel_stats, videos_by_channel, activity_by_channel, niche_profile)

    # 5. Second pass: get featured channels of top-3 competitors, score any new ones
    top_ids = sorted(
        [cid for cid, s in scores.items() if float(s.get("score", 0)) >= min_score],
        key=lambda cid: float(scores[cid].get("score", 0)),
        reverse=True,
    )[:3]

    if top_ids:
        log(f"Second pass: fetching featured channels of top {len(top_ids)} competitors...")
        extra_ids: set = set()
        for cid in top_ids:
            for fid in _get_featured_channels(cid):
                if fid and fid != seed_id and fid not in channel_stats:
                    extra_ids.add(fid)

        if extra_ids:
            log(f"  Found {len(extra_ids)} new channels via competitor featured lists — fetching stats...")
            extra_stats = _batch_channel_stats(list(extra_ids))
            extra_stats = {
                cid: info for cid, info in extra_stats.items()
                if min_subs <= info["subscribers"] <= max_subs and cid != seed_id
            }
            if extra_stats:
                extra_vbc, extra_abc = _fetch_activity_for_channels(extra_stats)
                extra_stats = {
                    cid: info for cid, info in extra_stats.items()
                    if extra_abc.get(cid, {}).get("videos", 0) >= min_videos_month
                    and extra_abc.get(cid, {}).get("views", 0) >= min_views_month
                }
                if extra_stats:
                    log(f"  Scoring {len(extra_stats)} extra channels from competitor featured lists...")
                    videos_by_channel.update(extra_vbc)
                    activity_by_channel.update(extra_abc)
                    extra_scores = _batch_score_channels(extra_stats, videos_by_channel, activity_by_channel, niche_profile)
                    scores.update(extra_scores)
                    channel_stats.update(extra_stats)

    # 6. Build results
    results = []
    for cid, info in channel_stats.items():
        s = scores.get(cid, {})
        score = float(s.get("score", 0.0))
        if score >= min_score:
            act = activity_by_channel.get(cid, {})
            results.append({
                "id":           cid,
                "title":        info["title"],
                "url":          info["url"],
                "subscribers":  info["subscribers"],
                "videos_month": act.get("videos", 0),
                "views_month":  act.get("views", 0),
                "score":        round(score, 2),
                "reason":       s.get("reason", ""),
                "query_hits":   video_hit_counts.get(cid, 0),
            })

    results.sort(key=lambda x: (x["score"], x["subscribers"]), reverse=True)

    if len(results) < 10:
        log(f"Found {len(results)} competitors (fewer than 10 — niche may be narrow, filters are strict by design)")
    else:
        log(f"Done: {len(results)} genuine competitors found")

    return results
