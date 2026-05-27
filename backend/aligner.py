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


def _pick_best_clip(clips: list, keywords: list, used: set) -> str | None:
    """Pick a clip not recently used. Rotate to avoid repetition."""
    # Try to find unused clip
    shuffled = list(clips)
    random.shuffle(shuffled)
    for clip in shuffled:
        if clip not in used:
            return clip
    # All used — reset and pick random
    return random.choice(clips) if clips else None


def build_timeline(
    script: str,
    validated_clips: list,
    audio_duration: float,
    chunk_words: int = 35,
) -> list:
    """
    Build a list of timeline entries:
    [{"clip": path, "start": float, "duration": float, "chunk_text": str}, ...]

    The clips are aligned to cover the full audio_duration.
    """
    if not validated_clips:
        raise RuntimeError("No validated clips to build timeline")

    chunks = _split_into_chunks(script, chunk_words)
    if not chunks:
        raise RuntimeError("Script produced no chunks")

    settings = config.load_settings()
    clip_min = settings.get("clip_min_duration", 2)
    clip_max = settings.get("clip_max_duration", 5)

    timeline    = []
    used_recent = set()
    t           = 0.0

    for chunk in chunks:
        if t >= audio_duration:
            break

        chunk_dur  = _chunk_duration(chunk)
        remaining  = audio_duration - t

        # Fill this chunk's duration with clips
        filled = 0.0
        while filled < chunk_dur and t + filled < audio_duration:
            clip = _pick_best_clip(validated_clips, _extract_keywords(chunk), used_recent)
            if not clip:
                break

            clip_full_dur = _get_duration(clip)
            if clip_full_dur <= 0.0:
                validated_clips = [c for c in validated_clips if c != clip]
                if not validated_clips:
                    break
                continue
            use_dur       = min(random.uniform(clip_min, clip_max), clip_full_dur, chunk_dur - filled, remaining - filled)
            use_dur       = max(use_dur, clip_min)

            timeline.append({
                "clip":       clip,
                "start":      round(t + filled, 3),
                "duration":   round(use_dur, 3),
                "chunk_text": chunk[:60],
            })

            used_recent.add(clip)
            if len(used_recent) > len(validated_clips) // 2:
                used_recent.clear()

            filled += use_dur

        t += chunk_dur

    # If timeline doesn't cover full audio, pad with random clips
    covered = sum(e["duration"] for e in timeline)
    while covered < audio_duration - 1.0:
        if not validated_clips:
            break
        clip     = random.choice(validated_clips)
        use_dur  = min(random.uniform(clip_min, clip_max), _get_duration(clip), audio_duration - covered)
        if use_dur < clip_min:
            break
        timeline.append({
            "clip":       clip,
            "start":      round(covered, 3),
            "duration":   round(use_dur, 3),
            "chunk_text": "",
        })
        covered += use_dur

    print(f"[aligner] Timeline: {len(timeline)} clips, {covered:.1f}s / {audio_duration:.1f}s", flush=True)
    return timeline


def timeline_to_clip_list(timeline: list) -> list:
    """Extract just the ordered list of clip paths from a timeline."""
    return [entry["clip"] for entry in timeline]
