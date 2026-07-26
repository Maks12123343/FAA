"""
War/Library Pipeline — виробництво відео з бібліотеки категоризованих кліпів.
Окрема гілка, незалежна від movie_pipeline.

Flow:
  1. Transcribe source URL
  2. Rewrite script (chunked, dedicated rewrite key)
  3. TTS voiceover
  4. Whisper → 2-5s сегменти
  5. Vertex embed всіх сегментів одним запитом
  6. Cosine similarity: для кожного сегмента шукаємо найкращий кліп у всій бібліотеці.
  7. Reuse кліпу до MAX_CLIP_USES разів у одному відео.
  8. Normalize + uniqualize (parallel) → montage (reuse із movie_pipeline).
"""

import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend import tts, api_client
from backend import languages as lang_utils
from backend.transcriber import get_transcript
from backend.rewriter import rewrite_all
from backend.aligner import _get_duration
import urllib.request as _urlreq

def _download_thumbnail(video_id: str, dst_path: str) -> bool:
    """Download max-res YouTube thumbnail. Returns True on success."""
    for name in ("maxresdefault", "sddefault", "hqdefault"):
        url = f"https://i.ytimg.com/vi/{video_id}/{name}.jpg"
        try:
            with _urlreq.urlopen(url, timeout=15) as r:
                data = r.read()
            if len(data) > 2000:
                with open(dst_path, "wb") as f:
                    f.write(data)
                return True
        except Exception:
            continue
    return False
from backend.movie_pipeline import (
    _segments_from_audio,
    _prepare_movie_clip,
    _build_movie_video,
    make_uniq_params_for_language,
    MIN_AUDIO_DURATION,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_CLIP_USES = 2                     # user preference: до 2 разів у одному відео
# ── Index loading ─────────────────────────────────────────────────────────────

_INDEX_CACHE = {}
_INDEX_LOCK = threading.Lock()


def _index_path_for(niche: str) -> str:
    """Path до index.json для war-style ніші."""
    return os.path.join(config.PROJECTS_DIR, "..", "movies", niche, "index.json")


def _load_library_index(niche: str) -> list:
    """
    Читає index.json для ніші, повертає список clips з ембеддингами.
    Кешується в пам'яті — 8802 кліпів × 768 float = ~27 MB, ок.
    """
    with _INDEX_LOCK:
        if niche in _INDEX_CACHE:
            return _INDEX_CACHE[niche]

    # Пробуємо кілька відомих шляхів
    candidates = [
        f"/workspace/FAA/movies/{niche}/index.json",
        os.path.join(config.PROJECTS_DIR, "..", "movies", niche, "index.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "movies", niche, "index.json"),
    ]
    index_path = None
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            index_path = p
            break
    if not index_path:
        raise RuntimeError(f"Index not found for niche '{niche}'. Tried: {candidates}")

    print(f"[war_pipeline] Loading index from {index_path}...", flush=True)
    t0 = time.time()
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    clips = data.get("clips", [])
    print(f"[war_pipeline] Loaded {len(clips)} clips in {time.time()-t0:.1f}s", flush=True)

    # Валідація: залишаємо тільки з ембеддингом та існуючим файлом
    valid = []
    missing_file = 0
    missing_emb = 0
    for c in clips:
        if not c.get("embedding"):
            missing_emb += 1
            continue
        if not c.get("file") or not os.path.exists(c["file"]):
            missing_file += 1
            continue
        valid.append(c)
    if missing_file or missing_emb:
        print(f"[war_pipeline] Filtered: {missing_file} missing files, {missing_emb} without embedding, {len(valid)} kept", flush=True)

    with _INDEX_LOCK:
        _INDEX_CACHE[niche] = valid
    return valid


# ── Clip selection via cosine ─────────────────────────────────────────────────

def _select_clips_semantic(segments: list, clips: list, emit=None) -> list:
    """
    Embed segments, compare against every clip in the library by cosine
    similarity, and pick the best available clip with MAX_CLIP_USES limit.
    Повертає [{"file": ..., "duration": ..., "id": ...}, ...]
    """
    import numpy as np
    from backend.embeddings import embed_texts

    n = len(segments)
    if not n or not clips:
        return []

    # 1. Ембеддимо всі сегменти одним batch-запитом
    seg_texts = [(seg.get("text", "") or "war footage") for seg in segments]
    if emit:
        emit("clips", f"Embedding {n} segments (Vertex batch)...")
    t0 = time.time()
    seg_vecs = embed_texts(seg_texts, emit=lambda step, msg: emit("clips", msg) if emit else None)
    if not seg_vecs:
        raise RuntimeError("Failed to embed segments (Vertex unavailable)")
    print(f"[war_pipeline] Segment embeddings ready in {time.time()-t0:.1f}s", flush=True)

    seg_matrix = np.array(seg_vecs, dtype=np.float32)
    seg_norms = np.linalg.norm(seg_matrix, axis=1, keepdims=True)
    seg_norms[seg_norms == 0] = 1.0
    seg_matrix_norm = seg_matrix / seg_norms

    # 2. Готуємо повну матрицю кліпів
    clip_matrix_full = np.array([c["embedding"] for c in clips], dtype=np.float32)
    clip_norms_full = np.linalg.norm(clip_matrix_full, axis=1, keepdims=True)
    clip_norms_full[clip_norms_full == 0] = 1.0
    clip_matrix_full_norm = clip_matrix_full / clip_norms_full

    print(f"[war_pipeline] Clip pool: {len(clips)} clips, selecting globally by cosine", flush=True)

    # 3. Для кожного сегмента — cosine по всій бібліотеці
    use_counts = {}  # clip_idx -> кількість використань
    selected = []
    if emit:
        emit("clips", f"Matching {n} segments against all {len(clips)} clips (cosine)...")

    for i, seg in enumerate(segments):
        seg_dur = max(0.5, seg.get("end", 0) - seg.get("start", 0))

        sims = clip_matrix_full_norm @ seg_matrix_norm[i]

        # Сортуємо по similarity, беремо перший який не використаний MAX_CLIP_USES
        order = np.argsort(-sims)
        picked = None
        for global_idx in order:
            global_idx = int(global_idx)
            if use_counts.get(global_idx, 0) >= MAX_CLIP_USES:
                continue
            picked = global_idx
            break

        if picked is None:
            # Крайній випадок — просто беремо найкращий (може повторитись >2 разів)
            picked = int(np.argmax(clip_matrix_full_norm @ seg_matrix_norm[i]))
            search_mode = "OVERFLOW"
        else:
            search_mode = "ALL"

        use_counts[picked] = use_counts.get(picked, 0) + 1
        clip = clips[picked]
        selected.append({
            "file": clip["file"],
            "duration": seg_dur,
            "id": clip.get("id", os.path.basename(clip["file"])),
            "search_scope": search_mode,
        })

        if emit and (i + 1) % 50 == 0:
            emit("clips", f"Matched {i+1}/{n} clips")

    reused = sum(1 for v in use_counts.values() if v > 1)
    print(f"[war_pipeline] Selected {len(selected)} clips ({reused} reused up to {MAX_CLIP_USES}×)", flush=True)
    return selected


# ── War-specific text overlay planning ───────────────────────────────────────
#
# Стиль плашки (референс — жовтий бокс "Crimean Peninsula"):
#   голуба суцільна підложка + жирний білий/чорний текст, лівий низ,
#   ~2 сек, лише коли у сегменті згадується ІМЕНОВАНИЙ об'єкт:
#     • конкретне місце (Bakhmut, Crimea, Kursk region, Kyiv)
#     • назва зброї / техніки (HIMARS, T-90, Bayraktar, Storm Shadow)
#     • назва операції / підрозділу (Operation Overlord, 47th Brigade)
#     • конкретна дата чи цифра втрат (24.02.2022, 500,000 casualties)
# НЕ вибирати емоційні фрази, гасла, загальні терміни ("war", "soldiers").

_WAR_OVERLAY_PROMPT = """\
You are planning MINIMAL text overlays for a documentary-style war/history video.
The video has a voiceover narration. Below are the script segments with timestamps.

Script segments:
{segments_json}

Select ONLY segments that mention a SPECIFIC NAMED ENTITY worth pinning on screen:
  1. Geographic locations — cities, regions, rivers, oblasts (Bakhmut, Crimea, Kursk, Dnipro River)
  2. Weapon systems / military tech (HIMARS, T-90, Bayraktar TB2, Storm Shadow, Iskander)
  3. Named military units or operations (47th Mechanized Brigade, Wagner Group, Operation Overlord)
  4. Concrete dates or casualty numbers (February 24 2022, 500,000 troops, 3 million refugees)
  5. Named people that are pivotal to the moment (Zelensky, Prigozhin) — sparingly, max 1-2 per video

STRICT RULES:
- Select AT MOST 8-12 overlays for the entire video. Fewer is better than more.
- No two overlays within 15 seconds of each other.
- Do NOT select emotional phrases, slogans, generic terms ("war", "soldiers", "battle"), rhetorical questions.
- Text must be the ENTITY NAME itself — 1-4 words, Title Case, in the target language: {language_name}.
- If a segment mentions multiple entities, pick the MOST important one and skip the rest.
- Return an empty array [] if nothing qualifies. Do NOT invent overlays.

Return ONLY a JSON array, no markdown, no commentary:
[
  {{"segment_index": 3, "text": "Bakhmut"}},
  {{"segment_index": 17, "text": "HIMARS"}},
  {{"segment_index": 42, "text": "24.02.2022"}}
]
"""

def _plan_text_overlays_war(segments_with_times: list, language: str, emit=None) -> list:
    """
    War-specific overlay planner: highlight ONLY named entities (places, tech,
    numbers, dates). Uses Byesu. Returns [] on
    failure — video will just render without overlays.
    """
    if not segments_with_times:
        return []

    lang_name = lang_utils.configured_language_name(language)

    seg_data = [
        {
            "index": s["index"],
            "start": round(s["start"], 1),
            "text": s["text"][:160],
        }
        for s in segments_with_times
    ]
    prompt = _WAR_OVERLAY_PROMPT.format(
        segments_json=json.dumps(seg_data, ensure_ascii=False, indent=2),
        language_name=lang_name,
    )

    try:
        text, _ = api_client.call_byesu(
            "You return JSON only. No markdown, no commentary.",
            [{"role": "user", "content": prompt}],
            timeout=120,
            max_retries=2,
            emit=emit,
            step_label="overlays",
            use_rewrite_model=False,
        )
    except Exception as e:
        print(f"[war_pipeline] War overlay API failed: {e}", flush=True)
        return []

    try:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        plan = json.loads(m.group() if m else text)
        if isinstance(plan, list):
            # Жорстка пост-фільтрація: ≤12 overlays, spacing ≥15s
            plan = [p for p in plan if isinstance(p, dict) and "segment_index" in p and "text" in p]
            plan.sort(key=lambda p: p["segment_index"])
            seg_map = {s["index"]: s for s in segments_with_times}
            filtered = []
            last_start = -999.0
            for p in plan:
                seg = seg_map.get(p["segment_index"])
                if not seg:
                    continue
                if seg["start"] - last_start < 15.0:
                    continue
                filtered.append(p)
                last_start = seg["start"]
                if len(filtered) >= 12:
                    break
            print(f"[war_pipeline] War overlays: {len(filtered)} entities (from {len(plan)} raw)", flush=True)
            return filtered
    except Exception as e:
        print(f"[war_pipeline] War overlay parsing failed: {e}", flush=True)
    return []


def _build_text_overlays_war(plan: list, segments_with_times: list) -> list:
    """
    Побудова overlay-об'єктів у форматі text_renderer.apply_text_overlays.
    Стиль плашки: голуба суцільна підложка (#4EA8FF, повна непрозорість),
    жирний білий текст, лівий низ, ~2 сек.
    """
    seg_map = {s["index"]: s for s in segments_with_times}
    overlays = []
    for item in plan:
        idx = item.get("segment_index")
        seg = seg_map.get(idx)
        if not seg:
            continue
        text = (item.get("text") or "").strip()[:40]
        if not text:
            continue
        start = seg["start"] + 0.2
        seg_dur = max(0.4, seg["end"] - seg["start"])
        dur = round(min(seg_dur - 0.2, 2.6), 2)
        if dur < 1.0:
            continue
        overlays.append({
            "text":     text,
            "start":    round(start, 2),
            "duration": dur,
            "position": "bottom-left",
            "size":     52,
            "color":    "white",
            "bg_color": "0x4EA8FF@1.0",
        })
    return overlays


# ── Main entry points ─────────────────────────────────────────────────────────

def prepare(source_url: str, emit=None) -> dict:
    """Фаза 1 — те саме що movie_pipeline.prepare: transcribe + збереження state."""
    def log(step, msg):
        print(f"[war_pipeline:prepare:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    prepare_id = f"war_{int(time.time())}"
    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    os.makedirs(prepare_dir, exist_ok=True)
    # ---- download YouTube thumbnail for library pipeline ----
    try:
        _m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", source_url or "")
        _vid = _m.group(1) if _m else ""
        if _vid:
            _thumb_dst = os.path.join(prepare_dir, "thumbnail.jpg")
            if _download_thumbnail(_vid, _thumb_dst):
                print(f"[war_pipeline:prepare] thumbnail saved: {_thumb_dst}", flush=True)
            else:
                print(f"[war_pipeline:prepare] thumbnail download failed for vid={_vid}", flush=True)
        else:
            print(f"[war_pipeline:prepare] no video_id in URL: {source_url}", flush=True)
    except Exception as _e:
        print(f"[war_pipeline:prepare] thumbnail step error: {_e!r}", flush=True)
    # ---- end thumbnail download ----

    log("transcribe", "Fetching transcript...")
    result = get_transcript(source_url)
    transcript = result["text"]
    log("transcribe", f"Got {len(transcript)} chars via {result['source']}")

    from backend import channel_scanner
    meta = channel_scanner.get_video_metadata(source_url)

    state = {
        "prepare_id": prepare_id,
        "prepare_dir": prepare_dir,
        "source_url": source_url,
        "source_title": meta.get("title", ""),
        "source_description": meta.get("description", ""),
        "source_tags": meta.get("tags", []),
        "transcript": transcript,
    }
    with open(os.path.join(prepare_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    log("prepare", "Transcription done.")
    return {
        "prepare_id": prepare_id,
        "source_url": source_url,
        "source_title": meta.get("title", ""),
        "source_views": meta.get("view_count", 0),
        "transcript": transcript[:2000],
        "transcript_len": len(transcript),
    }


def produce(prepare_id: str, niche: str, language: str, emit=None,
            test_mode: bool = False) -> dict:
    """
    Фаза 2: rewrite → TTS → segments → embed → global cosine → montage.
    """
    def log(step, msg):
        print(f"[war_pipeline:produce:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    timings = {}
    _stage_started = time.time()

    def mark_timing(name):
        nonlocal _stage_started
        now = time.time()
        elapsed = round(now - _stage_started, 1)
        timings[name] = elapsed
        _stage_started = now
        log("timing", f"{name}: {elapsed}s")

    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    with open(os.path.join(prepare_dir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)

    transcript = state["transcript"]
    source_title = state.get("source_title", "")

    # Проект
    proj_id = f"{niche}_{language}_{int(time.time())}"
    proj_dir = os.path.join(config.PROJECTS_DIR, proj_id)
    os.makedirs(proj_dir, exist_ok=True)

    # ── Rewrite ────────────────────────────────────────────────────────────────
    script_path = os.path.join(proj_dir, "script.txt")
    meta_path = os.path.join(proj_dir, "metadata.json")
    _thumb_prompt = ""
    if os.path.exists(script_path):
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
        log("rewrite", f"Script cached ({len(script)} chars)")
    else:
        log("rewrite", "Rewriting script (chunked, dedicated rewrite key)...")
        result = rewrite_all(
            transcript=transcript,
            language=language,
            source_title=source_title,
            source_description=state.get("source_description", ""),
            source_tags=state.get("source_tags", []),
            test_mode=test_mode,
        )
        script = result["script"]
        if len(script.split()) < 100:
            raise RuntimeError(f"Script too short: {len(script.split())} words")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "script"}, f, ensure_ascii=False, indent=2)
        log("rewrite", f"Script done: {len(script)} chars")
    mark_timing("rewrite")

    # ---- thumbnail analysis (library pipeline only) ----
    _thumb_out = os.path.join(proj_dir, "thumbnail_prompt.txt")
    if os.path.exists(_thumb_out):
        try:
            with open(_thumb_out, encoding="utf-8") as _f:
                _thumb_prompt = _f.read()
            log("thumbnail", f"Prompt cached ({len(_thumb_prompt)} chars)")
        except Exception as _e:
            print(f"[war_pipeline] thumbnail prompt cache read failed: {_e!r}", flush=True)
            _thumb_prompt = ""
    else:
        try:
            from backend import thumbnail as _thumb_mod
            _thumb_path = os.path.join(prepare_dir, "thumbnail.jpg")
            if not os.path.exists(_thumb_path):
                for _root, _dirs, _files in os.walk(prepare_dir):
                    if "thumbnail.jpg" in _files:
                        _thumb_path = os.path.join(_root, "thumbnail.jpg")
                        break
            if os.path.exists(_thumb_path):
                _meta_for_thumb = {}
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as _mf:
                        _meta_for_thumb = json.load(_mf)
                _thumb_result = _thumb_mod.analyze_and_rewrite(
                    _thumb_path,
                    language,
                    title=_meta_for_thumb.get("title", source_title),
                    emit=emit,
                )
                _thumb_prompt = (_thumb_result.get("prompt") or "").strip()
                if _thumb_prompt:
                    with open(_thumb_out, "w", encoding="utf-8") as _f:
                        _f.write(_thumb_prompt)
                    if emit:
                        emit("thumbnail_prompt", _thumb_prompt)
                    print(f"[war_pipeline] thumbnail prompt saved: {_thumb_out}", flush=True)
            else:
                print(f"[war_pipeline] no thumbnail.jpg found under {prepare_dir}", flush=True)
        except Exception as _e:
            print(f"[war_pipeline] thumbnail step failed: {_e!r}", flush=True)
    # ---- end thumbnail ----
    mark_timing("thumbnail")

    # ── TTS ────────────────────────────────────────────────────────────────────
    audio_path = os.path.join(proj_dir, "voiceover.mp3")
    if not os.path.exists(audio_path):
        log("tts", "Generating voiceover...")
        tts.generate(script, language, audio_path)
        log("tts", "Voiceover done.")
    else:
        log("tts", "Voiceover cached.")

    audio_dur = _get_duration(audio_path)
    log("tts", f"Audio duration: {audio_dur:.1f}s")
    mark_timing("tts")
    if audio_dur < MIN_AUDIO_DURATION:
        raise RuntimeError(f"Voiceover too short: {audio_dur:.1f}s (min {MIN_AUDIO_DURATION}s)")

    # ── Segments (Whisper 2-5s) ────────────────────────────────────────────────
    segments = _segments_from_audio(audio_path, audio_dur)
    log("segments", f"{len(segments)} segments, last ends at {segments[-1]['end']:.1f}s")
    mark_timing("segments")

    # ── Text overlays (те саме що movie_pipeline) ──────────────────────────────
    log("overlays", "Planning text overlays...")
    overlay_plan = _plan_text_overlays_war(segments, language, emit=emit)
    text_overlays = _build_text_overlays_war(overlay_plan, segments)
    log("overlays", f"Planned {len(text_overlays)} text overlays.")
    mark_timing("overlays")

    # ── Load library index ─────────────────────────────────────────────────────
    log("library", f"Loading library index for '{niche}'...")
    clips = _load_library_index(niche)
    log("library", f"Library ready: {len(clips)} valid clips")
    mark_timing("library")

    # ── Semantic clip selection (global cosine) ───────────────────────────────
    clips_cache = os.path.join(proj_dir, "clips.json")
    if os.path.exists(clips_cache):
        with open(clips_cache, encoding="utf-8") as f:
            clip_data = json.load(f)
        clip_data = [c for c in clip_data if os.path.exists(c.get("file", ""))]
        log("clips", f"Clips cached: {len(clip_data)}")
    else:
        log("clips", "Selecting clips via global cosine similarity...")
        clip_data = _select_clips_semantic(segments, clips, emit=emit)
        if not clip_data:
            raise RuntimeError("No clips selected — check library index and embeddings")

        with open(clips_cache, "w", encoding="utf-8") as f:
            json.dump(clip_data, f, ensure_ascii=False)
        log("clips", f"Selected {len(clip_data)} clips.")
    mark_timing("clip_select")

    # ── Prepare clips (normalize + uniqualize, parallel 4 workers) ────────────
    log("clips", "Preparing clips (normalize + uniqualize)...")
    uniq_params = make_uniq_params_for_language(language, proj_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepared = []
        completed_count = [0]
        count_lock = threading.Lock()

        def _prepare_one(args):
            i, cd = args
            out = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
            ok = _prepare_movie_clip(
                cd["file"], out, uniq_params,
                max_dur=cd["duration"],
                effect="none",
                speed=1.0,
            )
            with count_lock:
                completed_count[0] += 1
                n = completed_count[0]
                if emit and (n % 5 == 0 or n == len(clip_data)):
                    try:
                        pct = int(n / len(clip_data) * 100)
                        emit("clips", f"Preparing clip {n}/{len(clip_data)} ({pct}%)")
                    except Exception:
                        pass
            return (i, out) if ok else None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_prepare_one, (i, cd)) for i, cd in enumerate(clip_data)]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    prepared.append(r)

        prepared.sort(key=lambda x: x[0])
        prepared = [out for _, out in prepared]

        if not prepared:
            raise RuntimeError("No clips survived preparation.")
        mark_timing("clip_prepare")

        log("montage", f"Assembling {len(prepared)} clips ({audio_dur:.1f}s audio)...")
        if emit:
            emit("montage", "Assembling video segments...")
        output_path = os.path.join(proj_dir, f"{proj_id}.mp4")
        _build_movie_video(
            clips=prepared,
            audio_path=audio_path,
            output_path=output_path,
            text_overlays=text_overlays,
            proj_id=proj_id,
            emit=emit,
        )
        mark_timing("montage")

    log("done", f"Video ready: {output_path}")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "project_id": proj_id,
        "project_dir": proj_dir,
        "language": language,
        "language_name": lang_utils.configured_language_name(language),
        "thumbnail_prompt": _thumb_prompt,
        "output_path": output_path,
        "audio_dur": round(audio_dur, 1),
        "clips_used": len(prepared),
        "timings": timings,
        "title": meta.get("title", source_title),
        "all_titles": meta.get("titles", []),
        "description": meta.get("description", ""),
        "tags": meta.get("tags", []),
        "tags_raw": meta.get("tags_raw", ""),
    }
