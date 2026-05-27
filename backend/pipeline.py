import glob as _glob
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_slot_lock = threading.Lock()
_cand_lock = threading.Lock()

# Max times a competitor clip can appear across all videos in a batch (30% of 5 = ~2)
_COMPETITOR_MAX_USES = 2
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
from backend.aligner import _split_into_chunks, _chunk_duration, _get_duration
from backend.clip_downloader import build_pool
from backend.clip_matcher import match_clips_multi, analyze_all_clips
from backend.stocks_library import pick_stock_clips

WORDS_PER_SECTION = 35


# -- Phase 1: Prepare ---------------------------------------------------------

def prepare(niche_name: str, emit=None) -> dict:
    """
    Phase 1: Find top video -> fetch metadata -> transcribe.
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

    # Step 1: Find top video
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

    project_id  = f"{niche_name}_{language}_{int(time.time())}"
    project_dir = os.path.join(config.PROJECTS_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)

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

    with open(os.path.join(project_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script)

    with open(os.path.join(project_dir, "metadata.json"), "w", encoding="utf-8") as f:
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
    log("tts", f"Generating voiceover ({language})...")
    audio_path = os.path.join(project_dir, "voiceover.mp3")
    tts.generate(script, language, audio_path)
    audio_dur = montage._audio_duration(audio_path)
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
    slot_path = os.path.join(prepare_dir, "slot_counter.json")
    with _slot_lock:
        if os.path.exists(slot_path):
            with open(slot_path, encoding="utf-8") as f:
                slot = json.load(f)["count"]
        else:
            slot = 0
        with open(slot_path, "w", encoding="utf-8") as f:
            json.dump({"count": slot + 1}, f)
    log("media", f"Slot {slot} for '{language}' — each channel gets different clips")

    # Analyze clips with Gemini (one-time, cached per clip as .analysis.json)
    with _cand_lock:
        log("media", "Analyzing clips with Gemini (cached results reused)...")
        analyze_all_clips(clips_index, emit=emit)

    # Match clips to script sections by tags (free, no AI calls)
    candidates_path = os.path.join(prepare_dir, "candidates.json")
    if not os.path.exists(candidates_path):
        log("media", "Matching clips to script sections by tags...")
        section_texts_for_match = list(_split_into_chunks(transcript, WORDS_PER_SECTION))
        candidates = match_clips_multi(section_texts_for_match, clips_index, top_n=10, emit=emit)
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)
        log("media", f"Matching done: {sum(len(c) for c in candidates)} candidates across {len(candidates)} sections")
    else:
        with open(candidates_path, encoding="utf-8") as f:
            candidates = json.load(f)
        log("media", f"Candidates loaded from cache ({len(candidates)} sections, slot {slot})")

    # Load global used clips tracker (cross-video uniqueness)
    global_used = _load_global_used(prepare_dir)
    log("media", f"Global used clips loaded: {len(global_used)} clips tracked so far")

    # Pre-warm stock validation cache (batch mode — 8 clips per Gemini call, 3 workers)
    log("media", "Pre-validating stock clips in batch mode...")
    from backend.stocks_library import prewarm_stock_validation
    prewarm_stock_validation(whisper_segments, config.load_settings(), emit=emit)

    clips = _assemble_clips_from_candidates(
        candidates, whisper_segments, yt_pool, audio_dur, slot, global_used
    )
    log("media", f"Total clips assembled: {len(clips)}")

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


def _assemble_clips_from_candidates(
    candidates: list,
    whisper_segments: list,
    fallback_pool: list,
    audio_dur: float,
    slot: int,
    global_used: dict,
) -> list:
    """
    One clip per Whisper segment.
    candidates: per-section top-N clip paths (matched by tags).
    global_used: cross-video clip usage counter (mutated in place, saved by caller).
    """
    from backend.stocks_library import validate_stock_for_section, STOCK_SCORE_THRESHOLD

    settings         = config.load_settings()
    competitor_ratio = float(settings.get("competitor_ratio", 0.60))

    all_clips        = []
    n_cand           = len(candidates)
    n_segs           = len(whisper_segments)
    used_clips: set  = set()
    source_run: list = []
    SOURCE_RUN_MAX   = 3

    def _source_of(path: str) -> str:
        import re as _re
        name = os.path.basename(path)
        m = _re.match(r"^(.+)_\d{4}\.mp4$", name)
        return m.group(1) if m else name

    comp_cycle = list(fallback_pool)
    random.shuffle(comp_cycle)
    comp_idx = [0]

    def next_competitor(exclude_source=None, exclude_category=None, allow_reuse=False):
        """Pick next competitor clip respecting per-video and global uniqueness.
        allow_reuse=True relaxes per-video uniqueness but still respects global_used.
        """
        if not comp_cycle:
            return None
        # First pass: strict uniqueness
        for _ in range(len(comp_cycle)):
            if comp_idx[0] >= len(comp_cycle):
                random.shuffle(comp_cycle)
                comp_idx[0] = 0
            clip = comp_cycle[comp_idx[0]]
            comp_idx[0] += 1
            if clip in used_clips:
                continue
            if _clip_action(clip) in ("reject", "crop_bottom", "crop_corner"):
                continue
            if global_used.get(clip, 0) >= _COMPETITOR_MAX_USES:
                continue
            if exclude_source and _source_of(clip) == exclude_source:
                continue
            if exclude_category and _clip_category(clip) == exclude_category:
                continue
            return clip
        if not allow_reuse:
            return None
        # Second pass (gap-fill only): relax per-video uniqueness, keep global_used limit
        for _ in range(len(comp_cycle)):
            if comp_idx[0] >= len(comp_cycle):
                comp_idx[0] = 0
            clip = comp_cycle[comp_idx[0]]
            comp_idx[0] += 1
            if _clip_action(clip) in ("reject", "crop_bottom", "crop_corner"):
                continue
            if global_used.get(clip, 0) >= _COMPETITOR_MAX_USES:
                continue
            return clip
        print("[pipeline] WARNING: competitor pool exhausted (all clips hit global use limit)", flush=True)
        return None

    def _clip_analysis(path: str) -> dict:
        ap = path + ".analysis.json"
        if os.path.exists(ap):
            try:
                with open(ap, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _clip_action(path: str) -> str:
        return _clip_analysis(path).get("action", "use")

    def _clip_category(path: str) -> str:
        return _clip_analysis(path).get("category", "generic")

    CATEGORY_RUN_MAX  = 2
    recent_categories = []

    def _category_overused():
        if (len(recent_categories) >= CATEGORY_RUN_MAX and
                len(set(recent_categories[-CATEGORY_RUN_MAX:])) == 1):
            return recent_categories[-1]
        return None

    def _record_used(path: str, is_stock: bool = False):
        used_clips.add(path)
        global_used[path] = global_used.get(path, 0) + 1
        src = _source_of(path)
        source_run.append(src)
        if len(source_run) > SOURCE_RUN_MAX + 1:
            source_run.pop(0)
        cat = _clip_category(path)
        recent_categories.append(cat)
        if len(recent_categories) > CATEGORY_RUN_MAX + 1:
            recent_categories.pop(0)

    def _source_overused():
        if len(source_run) >= SOURCE_RUN_MAX and len(set(source_run[-SOURCE_RUN_MAX:])) == 1:
            return source_run[-1]
        return None

    for seg_idx, seg in enumerate(whisper_segments):
        seg_start = seg["start"]
        seg_end   = min(seg["end"], audio_dur)
        seg_dur   = seg_end - seg_start
        if seg_dur <= 0.1:
            continue

        seg_text = seg.get("text", "")

        if n_segs > 0 and n_cand > 0:
            tr_idx = min(int(seg_idx * n_cand / n_segs), n_cand - 1)
        else:
            tr_idx = 0

        section_candidates = candidates[tr_idx] if 0 <= tr_idx < n_cand else []
        effective_slot     = (tr_idx + slot) % len(section_candidates) if section_candidates else 0

        clip_path   = None
        is_stock    = False
        blocked_src = _source_overused()
        blocked_cat = _category_overused()

        want_stock = random.random() >= competitor_ratio

        if want_stock and seg_text.strip():
            stock_candidates = pick_stock_clips(seg_text, n=5)
            for sc in stock_candidates:
                if sc in used_clips:
                    continue
                if global_used.get(sc, 0) >= _STOCK_MAX_USES:
                    continue
                if _clip_action(sc) in ("reject",):
                    continue
                score = validate_stock_for_section(sc, seg_text)
                if score >= STOCK_SCORE_THRESHOLD:
                    clip_path = sc
                    is_stock  = True
                    break
            if not clip_path:
                clip_path = next_competitor(exclude_source=blocked_src, exclude_category=blocked_cat)

        else:
            if section_candidates:
                ordered = section_candidates[effective_slot:] + section_candidates[:effective_slot]
                for candidate in ordered[:10]:
                    if not os.path.exists(candidate):
                        continue
                    if candidate in used_clips:
                        continue
                    if _clip_action(candidate) in ("reject", "crop_bottom", "crop_corner"):
                        continue
                    if global_used.get(candidate, 0) >= _COMPETITOR_MAX_USES:
                        continue
                    if blocked_src and _source_of(candidate) == blocked_src:
                        continue
                    if blocked_cat and _clip_category(candidate) == blocked_cat:
                        continue
                    clip_path = candidate
                    break
            if not clip_path:
                clip_path = next_competitor(exclude_source=blocked_src, exclude_category=blocked_cat)

        if not clip_path:
            continue

        real_dur = _get_duration(clip_path)
        if real_dur < 0.5:
            clip_path = next_competitor(allow_reuse=True)
            if not clip_path:
                continue
            real_dur = _get_duration(clip_path)
            is_stock = False

        if is_stock:
            use_dur = min(real_dur, _stock_max(), seg_dur)
        else:
            use_dur = min(real_dur, seg_dur)

        _record_used(clip_path, is_stock=is_stock)
        all_clips.append((clip_path, use_dur))

        gap = seg_dur - use_dur
        while gap > 0.5:
            fill = next_competitor(allow_reuse=True)
            if not fill:
                break
            fill_real = _get_duration(fill)
            fill_dur  = min(fill_real, gap)
            if fill_dur < 0.1:
                break
            _record_used(fill)
            all_clips.append((fill, fill_dur))
            gap -= fill_dur

    return all_clips


# -- Legacy single-call (kept for compatibility) ------------------------------

def run(niche_name: str, language: str, emit=None) -> dict:
    result = prepare(niche_name, emit=emit)
    return produce(result["prepare_id"], [], language, emit=emit)
