import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFPROBE = config.FFPROBE
WORDS_PER_MINUTE = 145  # average TTS speaking rate


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _split_into_chunks(script: str, chunk_words: int = 40) -> list:
    """Split script into chunks of ~chunk_words words each."""
    sentences = re.split(r'(?<=[.!?])\s+', script.strip())
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        current.append(sent)
        current_words += words
        if current_words >= chunk_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_duration(chunk: str) -> float:
    """Estimate duration in seconds for a text chunk based on speaking rate."""
    words = len(chunk.split())
    return (words / WORDS_PER_MINUTE) * 60.0


def _extract_keywords(chunk: str) -> list:
    """Extract meaningful keywords from a text chunk."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "this", "that",
        "these", "those", "it", "its", "as", "if", "when", "while", "than",
        "then", "there", "their", "they", "we", "our", "you", "your",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.lower())
    keywords = [w for w in words if w not in stop_words]
    # Return top unique keywords by frequency
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    sorted_kw = sorted(freq, key=freq.get, reverse=True)
    return sorted_kw[:5]


def _source_id(clip_path: str) -> str:
    """Extract source video ID from clip filename.
    Handles: 'abc123?si=X_0042.mp4' → 'abc123'
    Handles: 'abc123_0042.mp4' → 'abc123'
    """
    name = os.path.basename(clip_path)
    if "?" in name:
        return name.split("?")[0]
    m = re.match(r'^(.+?)_\d+\.mp4$', name)
    return m.group(1) if m else name


def _pick_best_clip(
    clips: list,
    last_source: str | None,
    source_counts: dict,
    max_per_source: int,
) -> str | None:
    """Pick a clip that:
    1. Is not from the same source as the previous clip (no consecutive same-source)
    2. Doesn't exceed each source's fair-share budget
    Falls back gracefully if constraints can't be fully satisfied.
    """
    shuffled = list(clips)
    random.shuffle(shuffled)

    # Pass 1: respect both constraints
    for clip in shuffled:
        src = _source_id(clip)
        if src != last_source and source_counts.get(src, 0) < max_per_source:
            return clip

    # Pass 2: at least avoid consecutive same-source (ignore count limit)
    for clip in shuffled:
        if _source_id(clip) != last_source:
            return clip

    # Pass 3: last resort — any clip
    return shuffled[0] if shuffled else None


def build_timeline(
    script: str,
    validated_clips: list,
    audio_duration: float,
    chunk_words: int = 35,
) -> list:
    """
    Build a list of timeline entries:
    [{"clip": path, "start": float, "duration": float, "chunk_text": str}, ...]

    Rules:
    - No two consecutive clips from the same source video
    - Each source video gets at most ~1.5x its fair share of clips
    """
    if not validated_clips:
        raise RuntimeError("No validated clips to build timeline")

    chunks = _split_into_chunks(script, chunk_words)
    if not chunks:
        raise RuntimeError("Script produced no chunks")

    settings = config.load_settings()
    clip_min = settings.get("clip_min_duration", 2)
    clip_max = settings.get("clip_max_duration", 5)

    # Calculate per-source clip budget
    unique_sources = len(set(_source_id(c) for c in validated_clips))
    avg_clip_dur = (clip_min + clip_max) / 2
    estimated_total = max(1, int(audio_duration / avg_clip_dur))
    max_per_source = max(1, int((estimated_total / max(1, unique_sources)) * 1.5))

    print(f"[aligner] Sources: {unique_sources}, est. clips: {estimated_total}, max/source: {max_per_source}", flush=True)

    timeline      = []
    last_source   = None
    source_counts = {}
    t             = 0.0

    for chunk in chunks:
        if t >= audio_duration:
            break

        chunk_dur = _chunk_duration(chunk)
        remaining = audio_duration - t

        filled = 0.0
        while filled < chunk_dur and t + filled < audio_duration:
            clip = _pick_best_clip(validated_clips, last_source, source_counts, max_per_source)
            if not clip:
                break

            clip_full_dur = _get_duration(clip)
            if clip_full_dur <= 0.0:
                validated_clips = [c for c in validated_clips if c != clip]
                if not validated_clips:
                    break
                continue

            use_dur = min(random.uniform(clip_min, clip_max), clip_full_dur, chunk_dur - filled, remaining - filled)
            use_dur = max(use_dur, clip_min)

            src = _source_id(clip)
            source_counts[src] = source_counts.get(src, 0) + 1
            last_source = src

            timeline.append({
                "clip":       clip,
                "start":      round(t + filled, 3),
                "duration":   round(use_dur, 3),
                "chunk_text": chunk[:60],
            })

            filled += use_dur

        t += chunk_dur

    # Pad to cover full audio if needed
    covered = sum(e["duration"] for e in timeline)
    while covered < audio_duration - 1.0:
        if not validated_clips:
            break
        clip = _pick_best_clip(validated_clips, last_source, source_counts, max_per_source)
        if not clip:
            break
        use_dur = min(random.uniform(clip_min, clip_max), _get_duration(clip), audio_duration - covered)
        if use_dur < clip_min:
            break
        src = _source_id(clip)
        source_counts[src] = source_counts.get(src, 0) + 1
        last_source = src
        timeline.append({
            "clip":       clip,
            "start":      round(covered, 3),
            "duration":   round(use_dur, 3),
            "chunk_text": "",
        })
        covered += use_dur

    # Log source distribution
    dist = sorted(source_counts.items(), key=lambda x: -x[1])
    dist_str = ", ".join(f"{s}:{n}" for s, n in dist)
    print(f"[aligner] Timeline: {len(timeline)} clips, {covered:.1f}s / {audio_duration:.1f}s | distribution: {dist_str}", flush=True)
    return timeline


def timeline_to_clip_list(timeline: list) -> list:
    """Extract just the ordered list of clip paths from a timeline."""
    return [entry["clip"] for entry in timeline]
