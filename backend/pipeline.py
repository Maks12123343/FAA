import glob as _glob
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_cand_lock = threading.Lock()

# Max times a competitor clip can appear across all videos in one batch.
# Kept at 5 (not 2) so fresh-first selection has a wide enough fallback pool.
_COMPETITOR_MAX_USES = 3
# Stocks must be fully unique — max 1 use across all videos
_STOCK_MAX_USES = 1


def _load_global_used(prepare_dir: str) -> dict:
    """Load cross-video clip usage counter from prepare_dir."""
    path = os.path.join(prepare_dir, "global_used_clips.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        import shutil as _shutil
        corrupt = path + ".corrupt"
        try:
            _shutil.copy2(path, corrupt)
        except Exception:
            pass
        print(
            f"[pipeline] WARNING: global_used_clips.json is corrupt ({e}), "
            f"resetting to empty. Backup saved to {corrupt}",
            flush=True,
        )
        return {}


def _save_global_used(prepare_dir: str, global_used: dict):
    """Atomic save of cross-video clip usage counter — crash-safe."""
    path = os.path.join(prepare_dir, "global_used_clips.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(global_used, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


import config
from backend import channel_scanner, transcriber, rewriter, tts, montage, text_renderer
from backend.aligner import _split_into_chunks, _get_duration
from backend.clip_downloader import build_pool
from backend.clip_matcher import match_clips_multi, analyze_all_clips, batch_validate_candidates
from backend.stocks_library import pick_stock_clips

WORDS_PER_SECTION = 35


# -- Phase 1: Prepare ---------------------------------------------------------

def prepare(niche_name: str, source_url: str = None, emit=None) -> dict:
    """
    Phase 1: Find top video -> fetch metadata -> transcribe.
    If source_url is provided, skip channel scanning and use that video directly.
    Saves result to disk. Returns prepare_id + info for UI.
    """
    def log(step, msg):
        print(f"[pipeline:prepare:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    niche_path = os.path.join(config.NICHES_DIR, f"{niche_name}.json")
    if not os.path.exists(niche_path):
        raise FileNotFoundError(f"Niche not found: {niche_path}")

    prepare_id  = f"{niche_name}_{int(time.time())}"
    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    os.makedirs(prepare_dir, exist_ok=True)

    # Step 1: Find top video (or use provided URL)
    if source_url:
        log("scan", f"Using custom video URL: {source_url}")
        top_video = {"url": source_url, "title": "", "views": 0}
    else:
        log("scan", "Scanning channels for top video...")
        top_video = channel_scanner.find_top_video(niche_path)
        log("scan", f"Found: {top_video['title']}")

    # Step 2: Fetch source metadata (description + tags)
    log("scan", "Fetching source metadata...")
    source_meta = channel_scanner.get_video_metadata(top_video["url"])
    top_video.update(source_meta)

    # Step 3: Transcribe
    log("transcribe", "Extracting transcript...")
    transcript_result = transcriber.get_transcript(top_video["url"])
    transcript = transcript_result["text"]
    log("transcribe", f"Got {len(transcript)} chars via {transcript_result['source']}")

    # Save prepare state
    state = {
        "prepare_id":   prepare_id,
        "prepare_dir":  prepare_dir,
        "niche_name":   niche_name,
        "top_video":    top_video,
        "transcript":   transcript,
        "source_meta":  source_meta,
    }
    with open(os.path.join(prepare_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    log("prepare", "Ready. Waiting for YouTube URLs.")

    return {
        "prepare_id":    prepare_id,
        "source_url":    top_video["url"],
        "source_title":  top_video["title"],
        "source_views":  top_video.get("views", 0),
        "transcript":    transcript[:2000],
        "transcript_len": len(transcript),
    }


# -- Phase 2: Produce ---------------------------------------------------------

def produce(prepare_id: str, youtube_urls: list, language: str, emit=None) -> dict:
    """
    Phase 2: Download clips -> rewrite -> TTS -> montage.
    One language per call. Run multiple times for multiple languages.
    """
    def log(step, msg):
        print(f"[pipeline:produce:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    # Load prepare state
    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    state_path  = os.path.join(prepare_dir, "state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Prepare state not found: {prepare_id}")

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    niche_name  = state["niche_name"]
    top_video   = state["top_video"]
    transcript  = state["transcript"]
    source_meta = state["source_meta"]

    # Check if a previous incomplete run for this prepare+language already has
    # script/voiceover/whisper — if so, reuse them to save time
    def _find_reusable_project(niche: str, lang: str, pid: str) -> str | None:
        pattern = os.path.join(config.PROJECTS_DIR, f"{niche}_{lang}_*")
        candidates = sorted(_glob.glob(pattern), reverse=True)
        for folder in candidates:
            pid_file = os.path.join(folder, "_prepare_id.txt")
            if not os.path.exists(pid_file):
                continue
            with open(pid_file, encoding="utf-8") as f:
                if f.read().strip() != pid:
                    continue
            # Same prepare_id — check if has the expensive files but no output yet
            has_audio   = os.path.exists(os.path.join(folder, "voiceover.mp3"))
            has_whisper = os.path.exists(os.path.join(folder, "whisper_segments.json"))
            has_script  = os.path.exists(os.path.join(folder, "script.txt"))
            has_output  = os.path.exists(os.path.join(folder, "output.mp4"))
            if has_audio and has_whisper and has_script and not has_output:
                return folder
        return None

    reusable = _find_reusable_project(niche_name, language, prepare_id)
    if reusable:
        project_id  = os.path.basename(reusable)
        project_dir = reusable
        log("reuse", f"Resuming existing project: {project_id}")
    else:
        project_id  = f"{niche_name}_{language}_{int(time.time())}"
        project_dir = os.path.join(config.PROJECTS_DIR, project_id)
        os.makedirs(project_dir, exist_ok=True)

    # Tag project with prepare_id so future resume can find it
    with open(os.path.join(project_dir, "_prepare_id.txt"), "w", encoding="utf-8") as f:
        f.write(prepare_id)

    # -- Build clip pool (shared across languages, downloaded once) ------------
    pool_dir   = os.path.join(prepare_dir, "clip_pool")
    index_path = os.path.join(pool_dir, "clips_index.json")

    existing_clips = [
        p for p in _glob.glob(os.path.join(pool_dir, "*.mp4"))
        if not os.path.basename(p).startswith("_src")
    ] if os.path.exists(pool_dir) else []

    if not existing_clips:
        log("clips", f"Downloading {len(youtube_urls)} YouTube videos...")
        yt_pool, clips_index = build_pool(youtube_urls, pool_dir, emit=emit)
        if not clips_index:
            raise RuntimeError(
                "Clip pool is empty after download — all YouTube URLs may be "
                "geo-blocked, private, or yt-dlp was blocked. "
                "Check URLs and try again."
            )
    else:
        yt_pool = existing_clips
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                clips_index = json.load(f)
        else:
            clips_index = []
        if not clips_index:
            raise RuntimeError(
                f"Clip pool index is empty (pool_dir={pool_dir}). "
                "Delete the clip_pool folder and re-run to re-download."
            )
        log("clips", f"Using existing pool: {len(yt_pool)} clips")

    # -- Rewrite script (Claude) ----------------------------------------------
    script_path    = os.path.join(project_dir, "script.txt")
    metadata_path  = os.path.join(project_dir, "metadata.json")

    if os.path.exists(script_path) and os.path.exists(metadata_path):
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        title       = meta.get("title", top_video["title"])
        all_titles  = meta.get("all_titles", [])
        description = meta.get("description", "")
        tags        = meta.get("tags", [])
        log("rewrite", f"Script loaded from cache: \"{title}\"")
    else:
        log("rewrite", f"Rewriting script in {language}...")
        rewrite_result = rewriter.rewrite_all(
            transcript         = transcript,
            language           = language,
            source_title       = top_video["title"],
            source_description = source_meta.get("description", ""),
            source_tags        = source_meta.get("tags", []),
        )

        script      = rewrite_result.get("script", "")
        title       = rewrite_result.get("title", top_video["title"])
        all_titles  = rewrite_result.get("titles", [])
        description = rewrite_result.get("description", "")
        tags        = rewrite_result.get("tags", [])

        if len(script.strip()) < 100:
            raise RuntimeError(
                f"Rewriter returned an empty or too short script ({len(script)} chars). "
                "Check prompt files and Claude API key."
            )

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": title, "all_titles": all_titles,
                "description": description, "tags": tags,
            }, f, ensure_ascii=False, indent=2)
        with open(os.path.join(project_dir, "source.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"SOURCE VIDEO\n{'='*50}\n"
                f"URL:     {top_video['url']}\n"
                f"Title:   {top_video['title']}\n"
                f"Views:   {top_video.get('views', 0):,}\n"
                f"Date:    {top_video.get('upload_date', '')}\n"
                f"Channel: {top_video.get('channel_url', '')}\n"
            )
        log("rewrite", f"Done: \"{title}\"")

    # -- TTS ------------------------------------------------------------------
    audio_path = os.path.join(project_dir, "voiceover.mp3")
    if os.path.exists(audio_path) and montage._audio_duration(audio_path) >= 5:
        audio_dur = montage._audio_duration(audio_path)
        log("tts", f"Voiceover loaded from cache: {audio_dur:.0f}s")
    else:
        log("tts", f"Generating voiceover ({language})...")
        tts.generate(script, language, audio_path)
        audio_dur = montage._audio_duration(audio_path)
        if audio_dur < 5:
            raise RuntimeError(
                f"TTS produced audio that is too short ({audio_dur:.1f}s). "
                "The API may have failed silently or returned an empty file."
            )
        log("tts", f"Voiceover ready: {audio_dur:.0f}s")

    # -- Whisper: get real timestamps for clip cuts ---------------------------
    log("tts", "Segmenting voiceover with Whisper...")
    whisper_segs_path = os.path.join(project_dir, "whisper_segments.json")
    if os.path.exists(whisper_segs_path):
        with open(whisper_segs_path, encoding="utf-8") as f:
            whisper_segments = json.load(f)
        log("tts", f"Whisper segments loaded from cache ({len(whisper_segments)} segments)")
    else:
        whisper_segments = transcriber.transcribe_segments(audio_path)
        with open(whisper_segs_path, "w", encoding="utf-8") as f:
            json.dump(whisper_segments, f, ensure_ascii=False, indent=2)
        log("tts", f"Whisper done: {len(whisper_segments)} segments")

    # -- Pick clips -----------------------------------------------------------
    log("media", "Building clip list...")

    # Analyze clips with Gemini (one-time, cached per clip as .analysis.json)
    with _cand_lock:
        log("media", "Analyzing clips with Gemini (cached results reused)...")
        analyzed_index = analyze_all_clips(clips_index, emit=emit)

    # Build pre-validated candidate pool — shared across all languages, cached once
    val_path      = os.path.join(prepare_dir, "validated_candidates.json")
    section_texts = list(_split_into_chunks(transcript, WORDS_PER_SECTION))
    cur_settings  = config.load_settings()
    fingerprint   = _val_fingerprint(transcript, clips_index, cur_settings)

    validated_candidates = None
    if os.path.exists(val_path):
        try:
            with open(val_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fp") == fingerprint:
                validated_candidates = [[(item[0], item[1]) for item in sec] for sec in cached["data"]]
                log("media", f"Validated candidates loaded from cache ({len(validated_candidates)} sections)")
            else:
                log("media", "Validated candidates cache outdated (pool/model/threshold changed), rebuilding...")
                os.remove(val_path)
        except Exception:
            os.remove(val_path)

    if validated_candidates is None:
        log("media", "Matching and validating clips against original transcript (cached for all languages)...")
        raw_candidates = match_clips_multi(section_texts, analyzed_index, top_n=15, emit=emit)
        validated_candidates = batch_validate_candidates(
            raw_candidates, section_texts, cur_settings, emit=emit
        )
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump({"fp": fingerprint, "data": validated_candidates}, f, ensure_ascii=False)
        log("media", f"Validation done: {len(validated_candidates)} sections")

    # Load global used clips tracker (cross-video uniqueness)
    global_used = _load_global_used(prepare_dir)
    log("media", f"Global used clips loaded: {len(global_used)} clips tracked so far")

    # Pre-warm stock validation — precompute contexts once, reuse in assembly loop
    log("media", "Pre-validating stock clips in batch mode...")
    from backend.stocks_library import prewarm_stock_validation
    chunks_prebuilt   = _rechunk_segments(whisper_segments)
    stock_contexts    = _build_stock_contexts(chunks_prebuilt, title)
    synthetic_segs    = [{"text": ctx} for ctx in stock_contexts]
    prewarm_stock_validation(synthetic_segs, config.load_settings(), emit=emit)

    clips = _assemble_clips_from_candidates(
        validated_candidates, whisper_segments, yt_pool, audio_dur, global_used,
        video_title=title,
        precomputed_chunks=chunks_prebuilt,
        precomputed_stock_contexts=stock_contexts,
    )
    log("media", f"Total clips assembled: {len(clips)}")
    if not clips:
        raise RuntimeError(
            "No clips assembled — clip pool may be empty, all clips were rejected by Gemini, "
            "or Whisper produced no segments. Check logs above."
        )

    # Save updated global used clips tracker
    _save_global_used(prepare_dir, global_used)
    log("media", "Global used clips tracker saved")

    # -- Text overlays --------------------------------------------------------
    log("text", "Generating text overlays...")
    overlays = text_renderer.generate_stat_overlays(script, audio_dur)

    # -- Montage --------------------------------------------------------------
    log("montage", "Assembling video...")
    output_path = os.path.join(project_dir, "output.mp4")
    montage.assemble(clips, audio_path, output_path, text_overlays=overlays)

    # -- Final validation -----------------------------------------------------
    if not os.path.exists(output_path):
        raise RuntimeError(f"Render failed: output file not created: {output_path}")
    output_size = os.path.getsize(output_path)
    if output_size < 100_000:
        raise RuntimeError(f"Render failed: output file too small ({output_size} bytes)")
    output_dur = montage._audio_duration(output_path)
    if output_dur < 10:
        raise RuntimeError(f"Render failed: output duration too short ({output_dur:.1f}s)")
    if output_dur < audio_dur * 0.9:
        raise RuntimeError(
            f"Render failed: output ({output_dur:.1f}s) much shorter than voiceover ({audio_dur:.1f}s)"
        )
    log("done", f"Video ready: {output_path} ({output_size // 1_000_000}MB, {output_dur:.0f}s)")

    return {
        "project_id":   project_id,
        "output":       output_path,
        "title":        title,
        "all_titles":   all_titles,
        "description":  description,
        "tags":         tags,
        "source_video": top_video,
    }


def _stock_max() -> float:
    return float(config.load_settings().get("stock_max_duration", 6))


def _val_fingerprint(transcript: str, clips_index: list, settings: dict) -> str:
    import hashlib
    clip_parts = []
    for clip in clips_index:
        path = clip.get("file", "") if isinstance(clip, dict) else str(clip)
        try:
            st = os.stat(path)
            clip_parts.append(f"{path}|{st.st_size}|{round(st.st_mtime, 2)}")
        except Exception:
            clip_parts.append(path)
    key = "|".join([
        transcript[:1000],
        hashlib.md5("\n".join(sorted(clip_parts)).encode()).hexdigest(),
        settings.get("gemini_model", ""),
        str(settings.get("clip_score_threshold", 0.85)),
    ])
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _build_stock_contexts(chunks: list, video_title: str) -> list:
    """Build rich context strings per chunk for stock validation."""
    contexts = []
    for i, chunk in enumerate(chunks):
        parts = []
        if video_title:
            parts.append(f"Video topic: {video_title}")
        ctx_start = max(0, i - 1)
        ctx_end   = min(len(chunks), i + 2)
        ctx_text  = " ... ".join(c["text"] for c in chunks[ctx_start:ctx_end])
        parts.append(f"Context: {ctx_text}")
        parts.append(f"Segment to illustrate: {chunk['text']}")
        contexts.append("\n".join(parts))
    return contexts


def _rechunk_segments(
    whisper_segments: list,
    min_dur: float = 2.0,
    max_dur: float = 5.0,
) -> list:
    """
    Merge short Whisper segments and cap long ones so every chunk is 2-5 seconds.
    Returns [{"start": float, "end": float, "text": str}, ...]
    """
    chunks    = []
    buf_start = None
    buf_end   = None
    buf_texts = []

    def _save():
        dur  = buf_end - buf_start
        text = " ".join(buf_texts).strip()
        if dur >= min_dur or not chunks:
            chunks.append({"start": buf_start, "end": buf_end, "text": text})
        else:
            chunks[-1]["end"]  = buf_end
            chunks[-1]["text"] = (chunks[-1]["text"] + " " + text).strip()

    for seg in whisper_segments:
        s = seg["start"]
        e = seg["end"]
        t = seg.get("text", "")

        if buf_start is None:
            buf_start, buf_end, buf_texts = s, e, [t]
            continue

        if e - buf_start <= max_dur:
            buf_end = e
            buf_texts.append(t)
        else:
            _save()
            buf_start, buf_end, buf_texts = s, e, [t]

    if buf_start is not None:
        _save()

    # Split any chunks that still exceed max_dur (e.g. single long Whisper segment)
    final = []
    for chunk in chunks:
        dur = chunk["end"] - chunk["start"]
        if dur <= max_dur:
            final.append(chunk)
            continue
        words   = chunk["text"].split()
        t_start = chunk["start"]
        while chunk["end"] - t_start > 0.1:
            t_end     = min(t_start + max_dur, chunk["end"])
            remaining = chunk["end"] - t_end
            if 0 < remaining < min_dur:
                t_end = chunk["end"]  # absorb tiny tail into current chunk
            progress      = (t_start - chunk["start"]) / dur
            next_progress = (t_end   - chunk["start"]) / dur
            w_s = int(progress      * len(words))
            w_e = int(next_progress * len(words))
            final.append({
                "start": round(t_start, 3),
                "end":   round(t_end,   3),
                "text":  " ".join(words[w_s:w_e]).strip(),
            })
            t_start = t_end

    return final


def _assemble_clips_from_candidates(
    validated_candidates: list,
    whisper_segments: list,
    fallback_pool: list,
    audio_dur: float,
    global_used: dict,
    video_title: str = "",
    precomputed_chunks: list = None,
    precomputed_stock_contexts: list = None,
) -> list:
    """
    validated_candidates: list[list[(clip_path, score)]] sorted by score desc per section.
    One clip per rechunked Whisper segment. No gap-fill.
    Priority per chunk:
      1. Fresh clip (never used across videos) with score >= threshold
      2. Fresh clip regardless of score
      3. Any clip with score >= threshold
      4. Any available clip
      5. Fallback: random competitor clip
    """
    from backend.stocks_library import validate_stock_for_section, STOCK_SCORE_THRESHOLD

    import re as _re

    def _source_of(path: str) -> str:
        name = os.path.basename(path)
        if "?" in name:
            return name.split("?")[0]
        m = _re.match(r"^(.+?)_\d+\.mp4$", name)
        return m.group(1) if m else name

    settings        = config.load_settings()
    comp_ratio      = float(settings.get("competitor_ratio", 0.60))
    score_threshold = float(settings.get("clip_score_threshold", 0.85))

    chunks   = precomputed_chunks if precomputed_chunks is not None else _rechunk_segments(whisper_segments)
    n_chunks = len(chunks)
    n_cand   = len(validated_candidates)
    print(f"[pipeline] Whisper segs: {len(whisper_segments)} → chunks: {n_chunks}", flush=True)

    all_clips:          list       = []
    used_clips:         set        = set()
    source_clip_counts: dict       = {}
    last_source:        str | None = None
    non_fresh_used:     int        = 0
    max_non_fresh:      int        = max(1, int(n_chunks * 0.5))

    _unique_sources = len(set(_source_of(c) for c in fallback_pool)) or 1
    _max_per_source = max(2, int((n_chunks / _unique_sources) * 1.5))

    _analysis_cache: dict = {}

    def _clip_action(path: str) -> str:
        if path not in _analysis_cache:
            ap = path + ".analysis.json"
            try:
                with open(ap, encoding="utf-8") as f:
                    _analysis_cache[path] = json.load(f)
            except Exception:
                _analysis_cache[path] = {}
        return _analysis_cache[path].get("action", "use")

    def _record_used(path: str):
        is_new = path not in used_clips
        used_clips.add(path)
        if is_new:
            global_used[path] = global_used.get(path, 0) + 1
        source_clip_counts[_source_of(path)] = source_clip_counts.get(_source_of(path), 0) + 1

    def _is_usable(clip: str) -> bool:
        if not os.path.exists(clip):
            return False
        if clip in used_clips:
            return False
        if _clip_action(clip) == "reject":
            return False
        if global_used.get(clip, 0) >= _COMPETITOR_MAX_USES:
            return False
        if last_source and _source_of(clip) == last_source:
            return False
        return True

    # Fallback pool for when validated candidates are exhausted
    comp_cycle = list(fallback_pool)
    random.shuffle(comp_cycle)
    comp_idx = [0]

    def _next_competitor(fresh_only: bool = False) -> str | None:
        for _ in range(len(comp_cycle)):
            if comp_idx[0] >= len(comp_cycle):
                random.shuffle(comp_cycle)
                comp_idx[0] = 0
            clip = comp_cycle[comp_idx[0]]
            comp_idx[0] += 1
            if not _is_usable(clip):
                continue
            if fresh_only and global_used.get(clip, 0) > 0:
                continue
            if source_clip_counts.get(_source_of(clip), 0) >= _max_per_source:
                continue
            return clip
        if fresh_only:
            return None  # don't relax freshness constraint
        # Relax balance constraint
        for _ in range(len(comp_cycle)):
            if comp_idx[0] >= len(comp_cycle):
                comp_idx[0] = 0
            clip = comp_cycle[comp_idx[0]]
            comp_idx[0] += 1
            if _clip_action(clip) in ("reject",):
                continue
            if global_used.get(clip, 0) >= _COMPETITOR_MAX_USES:
                continue
            return clip
        print("[pipeline] WARNING: competitor pool exhausted", flush=True)
        return None

    for chunk_idx, chunk in enumerate(chunks):
        chunk_end = min(chunk["end"], audio_dur)
        chunk_dur = chunk_end - chunk["start"]
        if chunk_dur <= 0.1:
            continue

        chunk_text = chunk["text"]
        tr_idx     = min(int(chunk_idx * n_cand / max(1, n_chunks)), n_cand - 1) if n_cand > 0 else 0
        sec_cands  = validated_candidates[tr_idx] if 0 <= tr_idx < n_cand else []

        clip_path   = None
        is_stock    = False
        over_limit  = non_fresh_used >= max_non_fresh

        # Use precomputed stock context (same string used in prewarm → cache hits)
        if precomputed_stock_contexts and chunk_idx < len(precomputed_stock_contexts):
            stock_context = precomputed_stock_contexts[chunk_idx]
        else:
            parts = []
            if video_title:
                parts.append(f"Video topic: {video_title}")
            ctx_text = " ... ".join(c["text"] for c in chunks[max(0, chunk_idx-1):chunk_idx+2])
            parts.append(f"Context: {ctx_text}")
            parts.append(f"Segment to illustrate: {chunk_text}")
            stock_context = "\n".join(parts)

        # Try stock clip (use stock_context for candidate matching — same as prewarm)
        if random.random() >= comp_ratio and chunk_text.strip():
            for sc in pick_stock_clips(stock_context, n=5):
                if sc in used_clips or global_used.get(sc, 0) >= _STOCK_MAX_USES:
                    continue
                if _clip_action(sc) in ("reject",):
                    continue
                if last_source and _source_of(sc) == last_source:
                    continue
                if validate_stock_for_section(sc, stock_context) >= STOCK_SCORE_THRESHOLD:
                    clip_path = sc
                    is_stock  = True
                    break

        # Competitor selection: sec_cands sorted by score desc
        # If >= 50% non-fresh already used — only allow fresh clips
        if not clip_path:
            fresh_passes = [True] if over_limit else [True, False]
            for min_score in [score_threshold, 0.7]:
                for prefer_fresh in fresh_passes:
                    for candidate, score in sec_cands:
                        if score < min_score:
                            break
                        if not _is_usable(candidate):
                            continue
                        if prefer_fresh and global_used.get(candidate, 0) > 0:
                            continue
                        clip_path = candidate
                        break
                    if clip_path:
                        break
                if clip_path:
                    break

        # Last resort: search entire fallback pool
        # If over non-fresh limit — fresh clips only
        if not clip_path:
            clip_path = _next_competitor(fresh_only=over_limit)
        if not clip_path and over_limit:
            clip_path = _next_competitor(fresh_only=False)  # give up freshness constraint

        if not clip_path:
            print(f"[pipeline] WARNING: no clip for chunk {chunk_idx} ({chunk_dur:.1f}s)", flush=True)
            continue

        real_dur = _get_duration(clip_path)
        if real_dur < 0.5:
            continue

        use_dur = min(real_dur, _stock_max() if is_stock else chunk_dur, chunk_dur)
        if not is_stock and global_used.get(clip_path, 0) > 0:
            non_fresh_used += 1
        _record_used(clip_path)
        last_source = _source_of(clip_path)
        all_clips.append((clip_path, use_dur))

    return all_clips


# -- Legacy single-call (kept for compatibility) ------------------------------

def run(niche_name: str, language: str, emit=None) -> dict:
    result = prepare(niche_name, emit=emit)
    return produce(result["prepare_id"], [], language, emit=emit)
