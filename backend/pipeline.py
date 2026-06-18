import glob as _glob
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_cand_lock = threading.Lock()

# Max times any competitor clip can appear across the whole batch.
# Set to 1 — every clip is used at most once per video, no exceptions.
# Reused across multiple-language batches via global_used_clips.json (cross-video uniqueness).
_COMPETITOR_MAX_USES = 1
# Stocks must be fully unique — max 1 use across all videos
_STOCK_MAX_USES = 1


def _load_session_blocked(prepare_dir: str) -> set:
    """Load session-blocked clips (every 3rd used clip per video is blocked for subsequent videos)."""
    path = os.path.join(prepare_dir, "session_blocked.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_session_blocked(prepare_dir: str, blocked: set):
    """Atomic save of session-blocked clips."""
    path = os.path.join(prepare_dir, "session_blocked.json")
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(blocked), f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
from backend.translator import translate_sections

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

    # Load niche config to get montage_style
    niche_path = os.path.join(config.NICHES_DIR, f"{niche_name}.json")
    with open(niche_path, encoding="utf-8") as _nf:
        _niche_data = json.load(_nf)
    montage_style = _niche_data.get("montage_style", "standard")
    log("media", f"Montage style: {montage_style}")

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
        if not youtube_urls:
            # No URLs provided — try to fall back to library clips for this niche
            lib_dir = os.path.join(config.LIBRARY_DIR, niche_name, "raw")
            lib_clips = _glob.glob(os.path.join(lib_dir, "*.mp4")) if os.path.exists(lib_dir) else []
            if lib_clips:
                _from_movie_library = False
                log("clips", f"No YouTube URLs provided — using {len(lib_clips)} clips from Library ({niche_name})")
                os.makedirs(pool_dir, exist_ok=True)
                import shutil as _shutil
                for lc in lib_clips:
                    dst = os.path.join(pool_dir, os.path.basename(lc))
                    if not os.path.exists(dst):
                        _shutil.copy2(lc, dst)
                yt_pool = [os.path.join(pool_dir, os.path.basename(lc)) for lc in lib_clips]
                clips_index = [{"path": p, "score": 1.0} for p in yt_pool]
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(clips_index, f)
            else:
                # Check movie library (niche config: "movie_library": ["Movie Name", ...])
                movie_names = _niche_data.get("movie_library", [])
                movie_clips_index = []
                for movie_name in movie_names:
                    movie_idx_path = os.path.join(config.get_movies_dir(), movie_name, "index.json")
                    if os.path.exists(movie_idx_path):
                        with open(movie_idx_path, encoding="utf-8") as _mf:
                            movie_idx = json.load(_mf)
                        for clip in movie_idx.get("clips", []):
                            clip_file = clip.get("file", "")
                            if not clip_file or not os.path.exists(clip_file):
                                continue
                            if clip.get("is_blurry") or clip.get("is_static"):
                                continue
                            # Merge themes + emotion into tags for better semantic matching
                            merged_tags = list(clip.get("tags", []))
                            for t in clip.get("themes", []):
                                if t not in merged_tags:
                                    merged_tags.append(t)
                            if clip.get("emotion") and clip["emotion"] not in merged_tags:
                                merged_tags.append(clip["emotion"])
                            movie_clips_index.append({
                                "file":               clip_file,
                                "id":                 clip.get("id", ""),
                                "score":              1.0,
                                "description":        clip.get("description", ""),
                                "tags":               merged_tags,
                                "category":           clip.get("scene_type", "generic"),
                                "is_blurry":          False,
                                "is_static":          False,
                                "action":             "use",
                                "crop_percent":       0,
                                "watermark_position": "none",
                                "_size":              clip.get("_size"),
                                "_mtime":             clip.get("_mtime"),
                            })
                if movie_clips_index:
                    log("clips", f"No YouTube URLs — using {len(movie_clips_index)} clips from Movie Library ({', '.join(movie_names)})")
                    os.makedirs(pool_dir, exist_ok=True)
                    with open(index_path, "w", encoding="utf-8") as f:
                        json.dump(movie_clips_index, f)
                    yt_pool = [c["file"] for c in movie_clips_index]
                    clips_index = movie_clips_index
                    _from_movie_library = True
                else:
                    raise RuntimeError(
                        f"No YouTube URLs provided and Library is empty for niche '{niche_name}'. "
                        "Add YouTube URLs in Step 2, or first populate the Library via the Library page."
                    )
        else:
            _from_movie_library = False
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
        # Detect movie library mode from cached index (clips have pre-analyzed description)
        _from_movie_library = bool(clips_index and clips_index[0].get("description"))
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

    # Check if this niche uses random clip selection (skip all validation)
    _skip_validation = _niche_data.get("skip_validation", False)

    # Rechunk Whisper segments first — we match clips per Whisper chunk (1:1)
    log("media", f"Rechunking {len(whisper_segments)} Whisper segments (style={montage_style})...")
    chunks_prebuilt = _rechunk_segments(whisper_segments, style=montage_style)
    chunk_texts     = [c["text"] for c in chunks_prebuilt]
    if chunks_prebuilt:
        log("media", f"Rechunked: {len(chunk_texts)} chunks, "
            f"first={chunks_prebuilt[0]['start']:.1f}-{chunks_prebuilt[0]['end']:.1f}s, "
            f"last={chunks_prebuilt[-1]['start']:.1f}-{chunks_prebuilt[-1]['end']:.1f}s")
    else:
        log("media", "WARNING: rechunk produced 0 chunks — Whisper may have failed")

    # Load global used clips tracker (cross-video uniqueness)
    global_used = _load_global_used(prepare_dir)
    log("media", f"Global used clips loaded: {len(global_used)} clips tracked so far")

    # Load session blocked clips (every 3rd clip from previous videos in this session)
    session_blocked = _load_session_blocked(prepare_dir)
    log("media", f"Session blocked clips: {len(session_blocked)} clips unavailable this session")

    if _skip_validation:
        # ── Random clip selection (no Gemini/Pioneer/translation) ────────────
        log("media", "Random clip selection mode (skip_validation=true)")
        clips = _assemble_clips_random(
            yt_pool, chunks_prebuilt, audio_dur, global_used,
            session_blocked=session_blocked,
            niche_data=_niche_data,
        )
    else:
        # ── Full validation pipeline (Gemini + Pioneer) ─────────────────────
        # Analyze clips with Gemini (one-time, cached per clip as .analysis.json)
        with _cand_lock:
            log("media", "Analyzing clips with Gemini (cached results reused)...")
            analyzed_index = analyze_all_clips(clips_index, emit=emit)

        # Translate chunk texts to English for clip matching (tags/descriptions are English)
        chunk_texts_en = translate_sections(chunk_texts, language, project_dir=project_dir, emit=emit)

        # Build validated candidate pool — per language/project (cached in project_dir)
        val_path     = os.path.join(project_dir, "validated_candidates.json")
        cur_settings = config.load_settings()
        fingerprint  = _val_fingerprint(chunk_texts_en, clips_index, cur_settings)

        validated_candidates = None
        if os.path.exists(val_path):
            try:
                with open(val_path, encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("fp") == fingerprint:
                    validated_candidates = [[(item[0], item[1]) for item in sec] for sec in cached["data"]]
                    log("media", f"Validated candidates loaded from cache ({len(validated_candidates)} chunks)")
                else:
                    log("media", "Validated candidates cache outdated, rebuilding...")
                    os.remove(val_path)
            except Exception:
                os.remove(val_path)

        if validated_candidates is None:
            # Match per Whisper chunk — 1 chunk → top_n candidates
            _top_n = 5 if _from_movie_library else 10
            log("media", f"Matching clips per chunk (top_n={_top_n}, movie_library={_from_movie_library}, chunks={len(chunk_texts_en)})...")
            raw_candidates = match_clips_multi(chunk_texts_en, analyzed_index, top_n=_top_n, emit=emit)
            validated_candidates = batch_validate_candidates(
                raw_candidates, chunk_texts_en, cur_settings, emit=emit,
                movie_library_mode=_from_movie_library,
            )
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump({"fp": fingerprint, "data": validated_candidates}, f, ensure_ascii=False)
            log("media", f"Validation done: {len(validated_candidates)} chunks")

        # Pre-warm stock validation — skip entirely for movie_library mode (only movie clips used)
        if not _from_movie_library:
            log("media", "Pre-validating stock clips in batch mode...")
            from backend.stocks_library import prewarm_stock_validation
            stock_contexts = _build_stock_contexts(chunks_prebuilt, title)
            synthetic_segs = [{"text": ctx} for ctx in stock_contexts]
            prewarm_stock_validation(synthetic_segs, config.load_settings(), emit=emit)
        else:
            log("media", "movie_library mode — skipping stock pre-warm (only movie clips will be used)")
            stock_contexts = []

        clips = _assemble_clips_from_candidates(
            validated_candidates, whisper_segments, yt_pool, audio_dur, global_used,
            video_title=title,
            precomputed_chunks=chunks_prebuilt,
            precomputed_stock_contexts=stock_contexts,
            session_blocked=session_blocked,
            movie_library_mode=_from_movie_library,
        )

    log("media", f"Total clips assembled: {len(clips)}")
    if not clips:
        raise RuntimeError(
            "No clips assembled — clip pool may be empty, all clips were rejected by Gemini, "
            "or Whisper produced no segments. Check logs above."
        )

    # -- Text overlays --------------------------------------------------------
    log("text", "Generating text overlays...")
    overlays = text_renderer.generate_stat_overlays(script, audio_dur)

    # -- Montage --------------------------------------------------------------
    log("montage", "Assembling video...")
    output_path = os.path.join(project_dir, "output.mp4")
    montage.assemble(clips, audio_path, output_path, text_overlays=overlays, montage_style=montage_style)

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

    # Save global used clips tracker AFTER successful render
    _save_global_used(prepare_dir, global_used)
    log("media", "Global used clips tracker saved")

    # Save session blocked clips AFTER successful render
    _save_session_blocked(prepare_dir, session_blocked)
    log("media", f"Session blocked clips saved: {len(session_blocked)} total")

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


def _val_fingerprint(chunk_texts_or_transcript, clips_index: list, settings: dict) -> str:
    import hashlib
    pioneer_keys = settings.get("pioneer_api_keys", []) or []
    if isinstance(pioneer_keys, str):
        pioneer_keys = [k.strip() for k in pioneer_keys.split(",") if k.strip()]
    # Accept both list[str] (new: whisper chunks) and str (legacy: transcript)
    if isinstance(chunk_texts_or_transcript, list):
        text_key = " ".join(chunk_texts_or_transcript)[:1000]
    else:
        text_key = str(chunk_texts_or_transcript)[:1000]
    clip_parts = []
    for clip in clips_index:
        path = clip.get("file", "") if isinstance(clip, dict) else str(clip)
        try:
            st = os.stat(path)
            clip_parts.append(f"{path}|{st.st_size}|{round(st.st_mtime, 2)}")
        except Exception:
            clip_parts.append(path)
    key = "|".join([
        "validation_fp_v3_movie_text_parallel",
        text_key,
        hashlib.md5("\n".join(sorted(clip_parts)).encode()).hexdigest(),
        settings.get("gemini_model", ""),
        str(settings.get("clip_score_threshold", 0.85)),
        str(settings.get("pioneer_model", "")),
        str(len(pioneer_keys)),
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


def _dynamic_max_dur(t: float) -> float:
    """
    Return max clip duration based on timeline position for dynamic pacing:
      0–30s  → 3s  (fast cuts at the start to hook the viewer)
      30–90s → 4s  (medium pace in the middle)
      90s+   → 5s  (slower, more relaxed towards the end)
    """
    if t < 30:
        return 3.0
    elif t < 90:
        return 4.0
    else:
        return 5.0


def _rechunk_segments(
    whisper_segments: list,
    min_dur: float = 2.0,
    max_dur: float = 5.0,
    style: str = "standard",
) -> list:
    """
    Merge short Whisper segments and cap long ones so every chunk is 2-5 seconds.
    cinematic style: uses dynamic max duration (3/4/5s) based on timeline position.
    standard style:  uses fixed max_dur=5s throughout.
    Returns [{"start": float, "end": float, "text": str}, ...]
    """
    _use_dynamic = (style == "cinematic")

    chunks    = []
    buf_start = None
    buf_end   = None
    buf_texts = []

    def _cur_max() -> float:
        if _use_dynamic:
            return _dynamic_max_dur(buf_start) if buf_start is not None else max_dur
        return max_dur

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

        if e - buf_start <= _cur_max():
            buf_end = e
            buf_texts.append(t)
        else:
            _save()
            buf_start, buf_end, buf_texts = s, e, [t]

    if buf_start is not None:
        _save()

    # Split any chunks that still exceed their max_dur
    final = []
    for chunk in chunks:
        dur     = chunk["end"] - chunk["start"]
        cur_max = _dynamic_max_dur(chunk["start"]) if _use_dynamic else max_dur
        if dur <= cur_max:
            final.append(chunk)
            continue
        words   = chunk["text"].split()
        t_start = chunk["start"]
        while chunk["end"] - t_start > 0.1:
            seg_max   = _dynamic_max_dur(t_start) if _use_dynamic else max_dur
            t_end     = min(t_start + seg_max, chunk["end"])
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
    session_blocked: set = None,
    movie_library_mode: bool = False,
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

    session_blocked: set of clip paths blocked for this session (every 3rd used clip
    from previous videos). Updated in-place — every 3rd clip used in THIS video
    is added to session_blocked so subsequent videos avoid it.
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
    score_threshold = 0.75 if movie_library_mode else float(settings.get("clip_score_threshold", 0.85))

    chunks   = precomputed_chunks if precomputed_chunks is not None else _rechunk_segments(whisper_segments)
    n_chunks = len(chunks)
    n_cand   = len(validated_candidates)
    if n_cand == 0:
        print("[pipeline] WARNING: validated_candidates is empty — all clips will come from fallback pool", flush=True)
    print(f"[pipeline] Whisper segs: {len(whisper_segments)} → chunks: {n_chunks}", flush=True)

    all_clips:          list       = []
    used_clips:         set        = set()
    source_clip_counts: dict       = {}
    last_source:        str | None = None
    last_clip_path:     str | None = None  # for adjacent-index check (prevents clip_N → clip_N+1)
    non_fresh_used:     int        = 0
    max_non_fresh:      int        = max(1, int(n_chunks * 0.5))

    _unique_sources = len(set(_source_of(c) for c in fallback_pool)) or 1
    _max_per_source = max(2, int((n_chunks / _unique_sources) * 1.5))

    _analysis_cache: dict = {}

    def _clip_index_in_source(path: str) -> int | None:
        """
        Extract numeric index from clip filename like {source}_0042.mp4.
        Returns int or None if name doesn't match the expected pattern.
        """
        name = os.path.basename(path)
        m = _re.match(r"^.+_(\d+)\.mp4$", name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _is_adjacent(prev_path: str | None, candidate: str) -> bool:
        """
        True if candidate is the immediate next clip from the same source as prev_path
        (e.g. prev = source_X_0042.mp4 and candidate = source_X_0043.mp4).
        Used to avoid copyright issues from playing original-order consecutive clips.
        """
        if not prev_path:
            return False
        if _source_of(prev_path) != _source_of(candidate):
            return False
        prev_idx = _clip_index_in_source(prev_path)
        cand_idx = _clip_index_in_source(candidate)
        if prev_idx is None or cand_idx is None:
            return False
        return cand_idx == prev_idx + 1

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
        if session_blocked and clip in session_blocked:
            return False
        if last_source and _source_of(clip) == last_source:
            return False
        # Block immediate next-index clip from the same source as the previous one
        # (e.g. previous was X_0042.mp4 → block X_0043.mp4 as the next pick).
        if _is_adjacent(last_clip_path, clip):
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
            if clip in used_clips:
                continue
            if _is_adjacent(last_clip_path, clip):
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
        # 1:1 mapping: each Whisper chunk has its own candidate pool
        tr_idx    = chunk_idx if chunk_idx < n_cand else n_cand - 1
        sec_cands = validated_candidates[tr_idx] if 0 <= tr_idx < n_cand else []

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

        # Try stock clip — disabled in movie_library mode (only movie clips allowed)
        if not movie_library_mode and random.random() >= comp_ratio and chunk_text.strip():
            for sc in pick_stock_clips(stock_context, n=3):
                if sc in used_clips or global_used.get(sc, 0) >= _STOCK_MAX_USES:
                    continue
                if _clip_action(sc) in ("reject",):
                    continue
                if last_source and _source_of(sc) == last_source:
                    continue
                if _is_adjacent(last_clip_path, sc):
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

        # Movie library fallback: if all validated candidates are below 0.75,
        # still use the highest-scored available one before falling back randomly.
        if not clip_path and movie_library_mode:
            fresh_passes = [True] if over_limit else [True, False]
            for prefer_fresh in fresh_passes:
                for candidate, _score in sec_cands:
                    if not _is_usable(candidate):
                        continue
                    if prefer_fresh and global_used.get(candidate, 0) > 0:
                        continue
                    clip_path = candidate
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
        last_clip_path = clip_path
        all_clips.append((clip_path, use_dur))

    # Session blocking: every 3rd used clip is blocked for subsequent videos in this session.
    # Update session_blocked in-place so the caller can persist it.
    if session_blocked is not None:
        for i, (clip_path, _) in enumerate(all_clips):
            if (i + 1) % 3 == 0:  # 3rd, 6th, 9th, ...
                session_blocked.add(clip_path)
        print(
            f"[pipeline] Session blocked: {len(session_blocked)} clips total "
            f"(added {sum(1 for i in range(len(all_clips)) if (i+1) % 3 == 0)} from this video)",
            flush=True,
        )

    return all_clips


def _assemble_clips_random(
    pool: list,
    chunks: list,
    audio_dur: float,
    global_used: dict,
    session_blocked: set = None,
    niche_data: dict = None,
) -> list:
    """
    Random clip assignment: shuffle pool, assign one clip per chunk.
    No repeats within a single video. Between languages — repeats allowed.
    Avoids playing the immediate next-index clip from the same source right after
    its predecessor (e.g. X_0042 → X_0043) to reduce copyright detection risk.
    Supports watermark_crop from niche config.
    """
    import re as _re
    import subprocess as _sp

    niche_data = niche_data or {}
    crop_top = float(niche_data.get("crop_top_pct", 0))
    crop_bottom = float(niche_data.get("crop_bottom_pct", 0))
    do_crop = (crop_top > 0 or crop_bottom > 0)

    def _source_of(path: str) -> str:
        name = os.path.basename(path)
        if "?" in name:
            return name.split("?")[0]
        m = _re.match(r"^(.+?)_\d+\.mp4$", name)
        return m.group(1) if m else name

    def _clip_index_in_source(path: str) -> int | None:
        name = os.path.basename(path)
        m = _re.match(r"^.+_(\d+)\.mp4$", name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _is_adjacent(prev_path: str | None, candidate: str) -> bool:
        if not prev_path:
            return False
        if _source_of(prev_path) != _source_of(candidate):
            return False
        prev_idx = _clip_index_in_source(prev_path)
        cand_idx = _clip_index_in_source(candidate)
        if prev_idx is None or cand_idx is None:
            return False
        return cand_idx == prev_idx + 1

    available = [p for p in pool if os.path.exists(p)]
    if not available:
        print("[pipeline:random] ERROR: no clips available in pool", flush=True)
        return []
    random.shuffle(available)

    all_clips = []
    used_in_video = set()
    last_clip_path: str | None = None
    pool_idx = 0

    for chunk in chunks:
        chunk_end = min(chunk["end"], audio_dur)
        chunk_dur = chunk_end - chunk["start"]
        if chunk_dur <= 0.1:
            continue

        clip_path = None
        start_idx = pool_idx
        # First pass: respect both used-in-video and adjacency rules
        while True:
            if pool_idx >= len(available):
                pool_idx = 0
            candidate = available[pool_idx]
            pool_idx += 1
            if candidate not in used_in_video and not _is_adjacent(last_clip_path, candidate):
                clip_path = candidate
                break
            if pool_idx == start_idx:
                break

        # Fallback 1: relax adjacency but keep uniqueness
        if not clip_path:
            start_idx = pool_idx
            while True:
                if pool_idx >= len(available):
                    pool_idx = 0
                candidate = available[pool_idx]
                pool_idx += 1
                if candidate not in used_in_video:
                    clip_path = candidate
                    break
                if pool_idx == start_idx:
                    break

        if not clip_path:
            print(f"[pipeline:random] WARNING: pool exhausted, reusing clips", flush=True)
            clip_path = random.choice(available)

        real_dur = _get_duration(clip_path)
        if real_dur < 0.5:
            clip_path = random.choice(available)
            real_dur = _get_duration(clip_path)

        use_dur = min(real_dur, chunk_dur)
        used_in_video.add(clip_path)
        global_used[clip_path] = global_used.get(clip_path, 0) + 1
        last_clip_path = clip_path

        # Watermark crop: crop top/bottom % then scale back to 1920x1080
        if do_crop:
            cropped_path = clip_path + f".crop{int(crop_top)}_{int(crop_bottom)}.mp4"
            if not os.path.exists(cropped_path):
                h = 1080
                crop_y = int(h * crop_top / 100)
                crop_h = int(h * (1 - crop_top / 100 - crop_bottom / 100))
                vf = f"crop=1920:{crop_h}:0:{crop_y},scale=1920:1080"
                try:
                    _sp.run(
                        [config.FFMPEG, "-y", "-i", clip_path,
                         "-vf", vf,
                         *config.get_video_encoder_args("ultrafast"),
                         "-an", "-t", str(use_dur), cropped_path],
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=30,
                    )
                except Exception:
                    cropped_path = clip_path
            if os.path.exists(cropped_path) and os.path.getsize(cropped_path) > 1000:
                clip_path = cropped_path

        all_clips.append((clip_path, use_dur))

    # Session blocking
    if session_blocked is not None:
        for i, (cp, _) in enumerate(all_clips):
            if (i + 1) % 3 == 0:
                session_blocked.add(cp)

    print(f"[pipeline:random] Assembled {len(all_clips)} clips from pool of {len(available)} "
          f"(unique in video: {len(used_in_video)})", flush=True)
    return all_clips


# -- Legacy single-call (kept for compatibility) ------------------------------

def run(niche_name: str, language: str, emit=None) -> dict:
    result = prepare(niche_name, emit=emit)
    return produce(result["prepare_id"], [], language, emit=emit)
