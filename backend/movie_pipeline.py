"""
Movie Pipeline — виробництво cartoon-psychology відео на основі фільмів.
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend import tts
from backend.aligner import _split_into_chunks, _chunk_duration, _get_duration
from backend.transcriber import get_transcript
from backend.rewriter import rewrite_all
from backend.text_renderer import apply_text_overlays
from backend.movie_library import (
    search_clips, get_movie_clips,
    _uniqualize_movie_clip, make_uniq_params,
    VALIDATION_THRESHOLD,
)
from backend.clip_matcher import _validate_movie_clips_text_pioneer_batch

WORDS_PER_SECTION  = 35
MIN_AUDIO_DURATION = 60.0   # секунд — менше цього вважається помилкою TTS

# ── Whisper model cache (loaded once, reused across calls) ──────────────────────
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()

def _get_whisper_model():
    """Завантажити модель Whisper один раз і кешувати."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                import torch
                import whisper as _whisper
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = "medium" if device == "cuda" else "tiny"
                print(f"[movie_pipeline] Loading Whisper model ({model_name}) on {device}...", flush=True)
                _WHISPER_MODEL = _whisper.load_model(model_name, device=device)
                print("[movie_pipeline] Whisper model loaded.", flush=True)
    return _WHISPER_MODEL


# ── Real timestamps from audio ────────────────────────────────────────────────

def _segments_from_audio(audio_path: str, audio_dur: float) -> list:
    """
    Ділить аудіо на природні сегменти по паузах (2-5 секунд) через Whisper.
    Whisper сам визначає межі речень/пауз — ідеально для монтажу.
    Fallback — рівномірний розподіл по 3 секунди.
    Повертає: [{"index": i, "text": "...", "start": 0.0, "end": 3.2}, ...]
    """
    TARGET_MIN = 2.0
    TARGET_MAX = 5.0

    try:
        import time as _time

        print(f"[movie_pipeline] === WHISPER SEGMENTATION START ===", flush=True)
        print(f"[movie_pipeline] File: {os.path.basename(audio_path)}, target audio: {audio_dur:.1f}s", flush=True)

        print(f"[movie_pipeline] Step 1/4: Loading Whisper model...", flush=True)
        t0 = _time.time()
        model = _get_whisper_model()
        print(f"[movie_pipeline] Step 1/4: Model ready ({_time.time()-t0:.1f}s)", flush=True)

        print(f"[movie_pipeline] Step 2/4: Transcribing (may be slow on weak hardware)...", flush=True)
        t1 = _time.time()
        result = model.transcribe(audio_path, word_timestamps=False, language=None)
        print(f"[movie_pipeline] Step 2/4: Transcription done ({_time.time()-t1:.1f}s)", flush=True)

        raw_segs = result.get("segments", [])
        print(f"[movie_pipeline] Step 3/4: Got {len(raw_segs)} raw segments", flush=True)
        if not raw_segs:
            raise ValueError("No segments from Whisper")

        # Merge занадто короткі сегменти (< 2s) з наступним
        merged = []
        buf_start = raw_segs[0]["start"]
        buf_end   = raw_segs[0]["end"]
        buf_text  = raw_segs[0]["text"].strip()

        for seg in raw_segs[1:]:
            if (buf_end - buf_start) < TARGET_MIN:
                # Merge з наступним
                buf_end  = seg["end"]
                buf_text += " " + seg["text"].strip()
            else:
                merged.append({"start": buf_start, "end": buf_end, "text": buf_text})
                buf_start = seg["start"]
                buf_end   = seg["end"]
                buf_text  = seg["text"].strip()

        if buf_text:
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text})

        # Split занадто довгі сегменти (> 5s) пропорційно по словах
        final = []
        for seg in merged:
            dur = seg["end"] - seg["start"]
            if dur <= TARGET_MAX:
                final.append(seg)
            else:
                words   = seg["text"].split()
                n_parts = max(2, round(dur / 3.5))
                part_sz = max(1, len(words) // n_parts)
                t = seg["start"]
                for i in range(0, len(words), part_sz):
                    part_words = words[i:i + part_sz]
                    part_dur   = (len(part_words) / max(len(words), 1)) * dur
                    final.append({
                        "start": round(t, 3),
                        "end":   round(t + part_dur, 3),
                        "text":  " ".join(part_words),
                    })
                    t += part_dur

        for i, s in enumerate(final):
            s["index"] = i

        print(f"[movie_pipeline] Step 4/4: {len(final)} final segments (2-5s each)", flush=True)
        print(f"[movie_pipeline]   First: {final[0]['start']:.1f}-{final[0]['end']:.1f}s", flush=True)
        print(f"[movie_pipeline]   Last:  {final[-1]['start']:.1f}-{final[-1]['end']:.1f}s", flush=True)
        print(f"[movie_pipeline] === WHISPER SEGMENTATION DONE ===", flush=True)
        return final

    except Exception as e:
        print(f"[movie_pipeline] Whisper skipped ({e}), uniform fallback", flush=True)

    # Fallback: рівномірний розподіл по 3 секунди
    n       = max(1, round(audio_dur / 3.0))
    seg_dur = audio_dur / n
    return [
        {
            "index": i, "text": "",
            "start": round(i * seg_dur, 3),
            "end":   round((i + 1) * seg_dur, 3),
        }
        for i in range(n)
    ]


# ── Language-seeded uniqualization ────────────────────────────────────────────

def make_uniq_params_for_language(language: str, proj_id: str) -> dict:
    """
    Детермінований (між запусками) унікальний набір параметрів для пари (мова, проект).
    Використовує hashlib.md5 замість hash() — стабільний між запусками Python.
    """
    raw  = hashlib.md5(f"{language}:{proj_id}".encode()).digest()
    seed = int.from_bytes(raw[:4], "big")
    rng  = random.Random(seed)
    return {
        "zoom":       rng.uniform(1.04, 1.08),
        "brightness": rng.uniform(-0.05, 0.05),
        "contrast":   rng.uniform(0.95, 1.10),
        "saturation": rng.uniform(0.88, 1.18),
        "flip":       rng.random() < 0.30,
        "grain":      rng.uniform(6, 14),     # зерно плівки
    }


# ── Text overlay planning (Claude API) ────────────────────────────────────────

_TEXT_OVERLAY_PROMPT = """\
You are planning text overlays for a psychological video essay about cartoon characters (YouTube style, like "Impostor Syndrome" / "Dark Psychology" analysis videos).

The video has a voiceover narration. Below are the script segments with approximate timestamps.

Script segments:
{segments_json}

Select 20-25% of segments to receive a text overlay. Choose emotionally impactful moments — phrases that hit hard, shocking facts, key psychological terms.

Rules:
- Text must be VERY SHORT: 1-6 words maximum
- Space them out: no two overlays within 12 seconds of each other
- Three types:
  * "text_screen": huge centered text — for the most powerful single phrases (max 5 per video), shown on black/dark background
  * "text_overlay": medium text over the clip — for strong moments, key terms, shocking stats
  * "text_caption": smaller italic-style text at bottom — for quotes, character names, context labels

Return ONLY a JSON array, no markdown:
[
  {{"segment_index": 3, "text": "SHORT PHRASE", "type": "text_screen"}},
  {{"segment_index": 7, "text": "ANOTHER PHRASE", "type": "text_overlay"}},
  {{"segment_index": 12, "text": "context label", "type": "text_caption"}}
]"""


def _plan_text_overlays(segments_with_times: list, emit=None) -> list:
    """Pioneer (Claude Opus) читає сегменти скрипту і повертає план текстових оверлеїв."""
    from backend import api_client

    seg_data = [
        {"index": s["index"], "start": round(s["start"], 1), "text": s["text"][:120]}
        for s in segments_with_times
    ]
    prompt = _TEXT_OVERLAY_PROMPT.format(
        segments_json=json.dumps(seg_data, ensure_ascii=False, indent=2)
    )

    try:
        text, _ = api_client.call_pioneer(
            system="You are a creative video editor planning text overlays.",
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        m    = re.search(r"\[.*\]", text, re.DOTALL)
        plan = json.loads(m.group() if m else text)
        if isinstance(plan, list):
            return plan
    except Exception as e:
        print(f"[movie_pipeline] Text overlay planning failed: {e}", flush=True)

    return []


def _build_text_overlays(plan: list, segments_with_times: list) -> list:
    """Перетворити план оверлеїв у формат для text_renderer.apply_text_overlays."""
    seg_map  = {s["index"]: s for s in segments_with_times}
    overlays = []

    for item in plan:
        idx  = item.get("segment_index")
        seg  = seg_map.get(idx)
        if not seg:
            continue

        otype   = item.get("type", "text_overlay")
        text    = item.get("text", "")[:60]
        start   = seg["start"]
        seg_dur = seg["end"] - seg["start"]
        dur     = max(1.5, min(seg_dur, 3.5))

        if start + 0.2 >= seg["end"]:
            continue

        if otype == "text_screen":
            overlays.append({
                "text":     text.upper(),
                "start":    round(start + 0.2, 2),
                "duration": round(min(dur - 0.2, seg_dur - 0.2), 2),
                "position": "center",
                "size":     96,
                "color":    "white",
                "bg_color": "black@0.0",
            })
        elif otype == "text_caption":
            overlays.append({
                "text":     text,
                "start":    round(start + 0.2, 2),
                "duration": round(min(dur - 0.2, seg_dur - 0.2), 2),
                "position": "bottom-left",
                "size":     44,
                "color":    "white",
                "bg_color": "black@0.55",
            })
        else:  # text_overlay
            overlays.append({
                "text":     text.upper(),
                "start":    round(start + 0.2, 2),
                "duration": round(min(dur - 0.2, seg_dur - 0.2), 2),
                "position": "center",
                "size":     68,
                "color":    "white",
                "bg_color": "black@0.30",
            })

    return overlays


# ── Clip normalization + uniqualization ────────────────────────────────────────

def _prepare_movie_clip(clip_path: str, out_path: str, uniq_params: dict,
                        max_dur: float = 5.0,
                        effect: str = "none",
                        speed: float = 1.0) -> bool:
    """
    ОДИН FFmpeg pass: normalize + uniqualize + speed + vignette — все за раз.
    1 замість 4 процесів = в 4x швидше.
    """
    # Build filter chain
    filters = ["scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30"]

    # Uniqualize: zoom
    zoom = uniq_params.get("zoom", 1.0)
    if zoom > 1.0:
        crop_w = int(1920 / zoom)
        crop_h = int(1080 / zoom)
        x = (1920 - crop_w) // 2
        y = (1080 - crop_h) // 2
        filters.append(f"crop={crop_w}:{crop_h}:{x}:{y},scale=1920:1080")

    # brightness, contrast, saturation
    brightness = uniq_params.get("brightness", 0.0)
    contrast = uniq_params.get("contrast", 1.0)
    saturation = uniq_params.get("saturation", 1.0)
    if brightness != 0.0 or contrast != 1.0 or saturation != 1.0:
        filters.append(f"eq=brightness={brightness:.2f}:contrast={contrast:.2f}:saturation={saturation:.2f}")

    # flip
    if uniq_params.get("flip", False):
        filters.append("hflip")

    # grain
    grain = uniq_params.get("grain", 0)
    if grain > 0:
        filters.append(f"noise=alls={grain}:allf=t+u")

    # speed
    if abs(speed - 1.0) > 0.01:
        filters.append(f"setpts={1.0/speed:.4f}*PTS")

    vf = ",".join(filters)

    try:
        r = subprocess.run(
            [config.FFMPEG, "-y", "-i", clip_path,
             "-vf", vf,
             *config.get_video_encoder_args("ultrafast"),
             "-pix_fmt", "yuv420p", "-an", "-t", str(max_dur), out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 5000:
            if os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except Exception:
                    pass
            return False
        return True
    except Exception as e:
        print(f"[movie_pipeline] Clip prepare error: {e}", flush=True)
        return False




# ── Clip selection ─────────────────────────────────────────────────────────────

def _select_clips_for_segments(segments: list, movie_name: str,
                                audio_dur: float,
                                global_used_ids: set = None) -> list:
    """
    Для кожного сегменту:
    1. Знаходить 5 кандидатів (keyword scoring)
    2. Всі 5 йдуть в 1 запит до Pioneer → Gemini оцінює і повертає scores
    3. Обирається кліп з найвищим score

    4 Pioneer ключі працюють паралельно — секції розподіляються round-robin.
    1 секція = 1 API запит (5 кліпів за раз).

    Повертає: [{"file": path, "duration": seg_dur}, ...]
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    used_ids = global_used_ids if global_used_ids is not None else set()
    all_movie_clips = get_movie_clips(movie_name)

    # ── Phase 1: збираємо 5 кандидатів для кожного сегменту ──
    seg_data = []
    for seg_idx, seg in enumerate(segments):
        chunk   = seg.get("text", "")
        seg_dur = max(2.0, seg["end"] - seg["start"])

        candidates = search_clips(
            chunk, movie_name=movie_name,
            used_ids=used_ids, top_n=5,
            gemini_validate=False,
        )

        if not candidates:
            candidates = [
                c for c in all_movie_clips
                if c.get("id") not in used_ids
                and os.path.exists(c.get("file", ""))
            ]
            random.shuffle(candidates)
            candidates = candidates[:5]

        if not candidates:
            candidates = [
                c for c in all_movie_clips
                if os.path.exists(c.get("file", ""))
            ]
            random.shuffle(candidates)
            candidates = candidates[:5]

        pool = [c for c in candidates if os.path.exists(c.get("file", ""))][:5]
        if pool:
            seg_data.append((seg_idx, seg_dur, chunk, pool))

    if not seg_data:
        return []

    # ── Phase 2: паралельна валідація — 5 кліпів = 1 запит, 4 ключі паралельно ──
    settings = config.load_settings()
    pioneer_keys = settings.get("pioneer_api_keys", [])

    if not pioneer_keys:
        print("[movie_pipeline] WARNING: no Pioneer keys, skipping validation", flush=True)
        return [
            {"file": pool[0]["file"], "duration": seg_dur, "id": pool[0].get("id", pool[0]["file"])}
            for _, seg_dur, _, pool in seg_data
        ]

    scores_by_seg = {}
    scores_lock = threading.Lock()

    def _validate_section(seg_idx: int, chunk: str, pool: list, api_key: str):
        """5 кліпів → 1 запит → scores."""
        items = []
        for c in pool:
            items.append({
                "clip_path": c.get("file", ""),
                "section_text": chunk,
                "description": c.get("description", os.path.basename(c.get("file", ""))),
                "tags": c.get("tags", []),
            })
        try:
            scores = _validate_movie_clips_text_pioneer_batch(items, api_key)
            scores = [round(min(max(float(s), 0.0), 1.0), 4) for s in scores]
        except Exception as e:
            print(f"[movie_pipeline] Pioneer error seg {seg_idx}: {e}", flush=True)
            scores = [0.0] * len(pool)

        with scores_lock:
            scores_by_seg[seg_idx] = list(zip(pool, scores))

    n_workers = len(pioneer_keys)
    print(f"[movie_pipeline] Validating {len(seg_data)} segments × 5 clips, {n_workers} parallel workers", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as pool_exec:
        futures = []
        for i, (seg_idx, seg_dur, chunk, pool) in enumerate(seg_data):
            key = pioneer_keys[i % n_workers]
            futures.append(pool_exec.submit(_validate_section, seg_idx, chunk, pool, key))

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[movie_pipeline] Validation worker error: {e}", flush=True)

    # ── Phase 3: обираємо кліп з найвищим score ──
    selected = []
    used_ids_final = global_used_ids.copy() if global_used_ids else set()

    for seg_idx, seg_dur, chunk, pool in seg_data:
        scored_pool = scores_by_seg.get(seg_idx, [(pool[0], 0.0)])
        if not scored_pool:
            continue

        scored_sorted = sorted(scored_pool, key=lambda x: -x[1])

        chosen = None
        for clip, score in scored_sorted:
            cid = clip.get("id", clip["file"])
            if cid not in used_ids_final:
                chosen = clip
                break

        if not chosen:
            chosen = scored_sorted[0][0]

        file_path = chosen["file"]
        clip_id   = chosen.get("id", file_path)
        selected.append({"file": file_path, "duration": seg_dur, "id": clip_id})
        used_ids_final.add(clip_id)

    print(f"[movie_pipeline] Selected {len(selected)} clips for {audio_dur:.1f}s audio", flush=True)
    return selected


# ── Assembly ───────────────────────────────────────────────────────────────────

def _concat_clip_list(clip_paths: list, output: str):
    list_file = output + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            safe_p = p.replace(chr(92), '/').replace("'", "'\\''")
            f.write(f"file '{safe_p}'\n")
    try:
        subprocess.run(
            [config.FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p", "-an", output],
            capture_output=True, timeout=3600,
            check=True,
        )
    finally:
        if os.path.exists(list_file):
            os.unlink(list_file)


# Розширений набір переходів — різноманітний монтаж
_XFADE_TRANSITIONS = [
    "fade", "fadeblack", "dissolve", "hblur",
    "fadegrays", "smoothleft", "smoothright",
    "wipeleft", "wiperight", "slideleft", "slideright",
    "circlecrop", "rectcrop", "pixelize",
    "squeezeh", "squeezev",
]

# Переходи що виглядають "людськими" — використовуємо частіше
_NATURAL_TRANSITIONS = [
    "fade", "dissolve", "fadeblack", "hblur", "fadegrays",
]


def _pick_transition(i: int, rng: random.Random = None) -> str:
    """
    Вибирає перехід: 50% — природні (fade/dissolve), 50% — динамічні.
    Більш динамічний монтаж.
    """
    r = (rng or random).random()
    if r < 0.50:
        pool = _NATURAL_TRANSITIONS
    else:
        pool = _XFADE_TRANSITIONS
    return (rng or random).choice(pool)


def _xfade_join(segment_files: list, output: str, fade_dur: float = 0.35,
                rng: random.Random = None):
    n = len(segment_files)
    if n == 1:
        shutil.copy2(segment_files[0], output)
        return

    durations = [_get_duration(s) for s in segment_files]
    if any(d <= 0 for d in durations) or any(d <= fade_dur * 2 for d in durations):
        _concat_clip_list(segment_files, output)
        return

    inputs = []
    for s in segment_files:
        inputs += ["-i", s]

    filters    = []
    cumulative = 0.0
    prev_label = "0:v"

    for i in range(1, n):
        cumulative += durations[i - 1] - fade_dur
        out_label   = f"x{i}" if i < n - 1 else "vout"
        offset      = max(0.0, cumulative)
        transition  = _pick_transition(i, rng)
        filters.append(
            f"[{prev_label}][{i}:v]xfade=transition={transition}"
            f":duration={fade_dur:.2f}:offset={offset:.3f}[{out_label}]"
        )
        prev_label = out_label

    r = subprocess.run(
        [config.FFMPEG, "-y"] + inputs +
        ["-filter_complex", ";".join(filters),
         "-map", "[vout]",
         *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p", "-an", output],
        capture_output=True, timeout=3600,
    )
    if r.returncode != 0:
        _concat_clip_list(segment_files, output)


def _loop_video_to_duration(video_path: str, target_dur: float, output: str):
    """Лупить відео до потрібної тривалості через stream_loop."""
    subprocess.run(
        [config.FFMPEG, "-y",
         "-stream_loop", "-1", "-i", video_path,
         "-t", f"{target_dur:.3f}",
         *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p", "-an", output],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600,
    )


def _assign_clip_effects(n_clips: int, rng: random.Random) -> list:
    """
    Призначає ефекти і тривалості для кожного кліпу.
    Повертає список dict: {effect, speed, max_dur}.

    Розподіл (динамічний монтаж):
      - 45% — звичайний кліп (2.0–3.5с) — коротші = більше змін кадру
      - 30% — прискорений (1.5–3.0с, speed 1.15–1.40x) — динаміка
      - 25% — flash cut (0.5–1.5с) — різкі вставки для акценту
    """
    assignments = []
    for _ in range(n_clips):
        r = rng.random()
        if r < 0.45:
            assignments.append({
                "effect":   "none",
                "speed":    1.0,
                "max_dur":  rng.uniform(2.0, 3.5),
            })
        elif r < 0.75:
            assignments.append({
                "effect":   "none",
                "speed":    rng.uniform(1.15, 1.40),
                "max_dur":  rng.uniform(1.5, 3.0),
            })
        else:
            assignments.append({
                "effect":   "none",
                "speed":    1.0,
                "max_dur":  rng.uniform(0.5, 1.5),
            })
    return assignments


def _build_movie_video(clips: list, audio_path: str, output_path: str,
                       text_overlays: list = None, proj_id: str = "", emit=None):
    """
    Збирає готові (вже нормалізовані + унікалізовані) кліпи у фінальне відео.
    Складний монтаж: різні тривалості, Ken Burns, speed ramping, flash cuts,
    різноманітні переходи. Гарантує покриття повної тривалості аудіо.
    """
    from backend.montage import _fetch_bg_music, _add_audio

    # Детермінований RNG для монтажу (різний для кожного proj_id)
    seed = int.from_bytes(hashlib.md5(proj_id.encode()).digest()[:4], "big") if proj_id else None
    rng  = random.Random(seed)

    # Групи: розмір варіюється 2–5 (менші групи = частіші переходи)
    GROUP_SIZES = [2, 2, 3, 3, 3, 4, 4, 5]
    FADE_DUR    = 0.20  # швидкий fade — максимальна динаміка

    proj_dir   = os.path.dirname(output_path)
    raw_video  = os.path.join(proj_dir, "_raw_movie.mp4")
    with_audio = os.path.join(proj_dir, "_with_audio_movie.mp4")

    audio_dur = _get_duration(audio_path)

    # Групуємо кліпи
    groups = []
    i = 0
    while i < len(clips):
        size = rng.choice(GROUP_SIZES)
        groups.append(clips[i:i + size])
        i += size

    with tempfile.TemporaryDirectory() as tmp:
        seg_files = []
        n_groups = len(groups)

        for g_idx, group in enumerate(groups):
            seg = os.path.join(tmp, f"seg_{g_idx:04d}.mp4")
            _concat_clip_list(group, seg)
            if os.path.exists(seg) and os.path.getsize(seg) > 1000:
                seg_files.append(seg)
            if emit and n_groups > 5 and (g_idx + 1) % max(1, n_groups // 5) == 0:
                emit("montage", f"Building segments: {g_idx + 1}/{n_groups} ({int((g_idx+1)/n_groups*100)}%)")

        if not seg_files:
            raise RuntimeError("No segments created during assembly")

        if emit:
            emit("montage", f"Joining {len(seg_files)} segments...")

        assembled = os.path.join(tmp, "_assembled.mp4")
        if len(seg_files) == 1:
            shutil.copy2(seg_files[0], assembled)
        else:
            # Між деякими групами — різний fade_dur для динаміки
            _xfade_join(seg_files, assembled, FADE_DUR, rng=rng)

        # Перевіряємо чи відео покриває аудіо; якщо ні — лупимо
        assembled_dur = _get_duration(assembled)
        if assembled_dur < audio_dur - 0.5:
            print(f"[movie_pipeline] Video {assembled_dur:.1f}s < audio {audio_dur:.1f}s — looping", flush=True)
            if emit:
                emit("montage", "Extending video to match audio length...")
            _loop_video_to_duration(assembled, audio_dur + 1.0, raw_video)
        else:
            shutil.copy2(assembled, raw_video)

    # No background music for movie pipeline — reduces load on laptop CPU
    # Simple audio copy: just add voiceover to video, no music mixing
    if emit:
        emit("montage", "Adding voiceover...")
    r = subprocess.run(
        [config.FFMPEG, "-y", "-i", raw_video, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-t", str(audio_dur), "-movflags", "+faststart",
         with_audio],
        capture_output=True, timeout=3600,
    )
    if r.returncode != 0:
        print(f"[movie_pipeline] Audio copy failed: {r.stderr.decode(errors='replace')[-500:]}", flush=True)
        shutil.copy2(raw_video, with_audio)

    if text_overlays:
        if emit:
            emit("text", f"Adding {len(text_overlays)} text overlays...")
        apply_text_overlays(with_audio, text_overlays, output_path)
    else:
        shutil.copy2(with_audio, output_path)

    if emit:
        emit("montage", "Video assembly complete.")

    for p in [raw_video, with_audio]:
        if os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass


# ── Main pipeline ──────────────────────────────────────────────────────────────

def prepare(source_url: str, emit=None) -> dict:
    """Фаза 1: транскрипція джерельного відео."""
    def log(step, msg):
        print(f"[movie_pipeline:prepare:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    prepare_id  = f"movie_{int(time.time())}"
    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    os.makedirs(prepare_dir, exist_ok=True)

    log("transcribe", "Fetching transcript...")
    result     = get_transcript(source_url)
    transcript = result["text"]
    log("transcribe", f"Got {len(transcript)} chars via {result['source']}")

    from backend import channel_scanner
    meta = channel_scanner.get_video_metadata(source_url)

    state = {
        "prepare_id":         prepare_id,
        "prepare_dir":        prepare_dir,
        "source_url":         source_url,
        "source_title":       meta.get("title", ""),
        "source_description": meta.get("description", ""),
        "source_tags":        meta.get("tags", []),
        "transcript":         transcript,
    }
    with open(os.path.join(prepare_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    log("prepare", "Transcription done.")
    return {
        "prepare_id":     prepare_id,
        "source_url":     source_url,
        "source_title":   meta.get("title", ""),
        "transcript":     transcript[:2000],
        "transcript_len": len(transcript),
    }


def produce(prepare_id: str, movie_name: str, language: str, emit=None,
            global_used_ids: set = None) -> dict:
    """
    Фаза 2: рірайт → TTS → підбір кліпів + validation → текстові оверлеї → монтаж.

    global_used_ids — множина clip ID вже використаних в попередніх відео батчу.
    Передається ззовні щоб гарантувати різноманітність відеоряду між відео.
    """
    def log(step, msg):
        print(f"[movie_pipeline:produce:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    with open(os.path.join(prepare_dir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)

    transcript   = state["transcript"]
    source_title = state.get("source_title", "")

    # ── Resume: шукаємо існуючий проект з тим самим prepare_id + language ────
    _resume_proj_id = None
    _resume_candidates = []
    for _d in os.listdir(config.PROJECTS_DIR):
        _dpath = os.path.join(config.PROJECTS_DIR, _d)
        if not os.path.isdir(_dpath):
            continue
        _pid_file = os.path.join(_dpath, "_prepare_id.txt")
        if not os.path.exists(_pid_file):
            continue
        try:
            with open(_pid_file, encoding="utf-8") as _f:
                _stored = _f.read().strip()
        except Exception:
            continue
        if _stored != prepare_id:
            continue
        if f"_{language}_" not in _d and not _d.endswith(f"_{language}"):
            continue
        _vo = os.path.join(_dpath, "voiceover.mp3")
        _mp4 = os.path.join(_dpath, f"{_d}.mp4")
        if os.path.exists(_vo) and not os.path.exists(_mp4):
            _resume_candidates.append(_d)
    # Pick the newest candidate (highest timestamp in folder name)
    if _resume_candidates:
        _resume_candidates.sort(reverse=True)
        _resume_proj_id = _resume_candidates[0]
        log("resume", f"Resuming existing project: {_resume_proj_id}")

    if _resume_proj_id:
        proj_id  = _resume_proj_id
        proj_dir = os.path.join(config.PROJECTS_DIR, proj_id)
    else:
        proj_id  = f"{movie_name}_{language}_{int(time.time())}"
        proj_dir = os.path.join(config.PROJECTS_DIR, proj_id)
        os.makedirs(proj_dir, exist_ok=True)

    # Детермінований RNG для ефектів кліпів (різний для кожного proj_id)
    clip_seed = int.from_bytes(hashlib.md5(proj_id.encode()).digest()[:4], "big")
    clip_rng  = random.Random(clip_seed)

    # ── Рірайт ────────────────────────────────────────────────────────────────
    script_path = os.path.join(proj_dir, "script.txt")
    if os.path.exists(script_path):
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
        log("rewrite", f"Script cached ({len(script)} chars)")
    else:
        log("rewrite", "Rewriting script (with quality check)...")
        result = rewrite_all(
            transcript          = transcript,
            language            = language,
            source_title        = source_title,
            source_description  = state.get("source_description", ""),
            source_tags         = state.get("source_tags", []),
        )
        script = result["script"]
        if len(script.split()) < 100:
            raise RuntimeError(
                f"Script too short ({len(script.split())} words). "
                "Rewriter may have failed — check Claude API key and prompt."
            )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        meta_path = os.path.join(proj_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "script"},
                      f, ensure_ascii=False, indent=2)
        log("rewrite", f"Script done: {len(script)} chars, {len(script.split())} words")

    # ── TTS ───────────────────────────────────────────────────────────────────
    audio_path = os.path.join(proj_dir, "voiceover.mp3")
    if not os.path.exists(audio_path):
        log("tts", "Generating voiceover...")
        tts.generate(script, language, audio_path)
        log("tts", "Voiceover done.")
    else:
        log("tts", "Voiceover cached.")

    audio_dur = _get_duration(audio_path)
    log("tts", f"Audio duration: {audio_dur:.1f}s")

    # Перевірка мінімальної тривалості
    if audio_dur < MIN_AUDIO_DURATION:
        raise RuntimeError(
            f"Voiceover too short: {audio_dur:.1f}s (min {MIN_AUDIO_DURATION}s). "
            "Script may be too short or TTS failed."
        )

    # ── Сегменти з таймстампами (Whisper по паузах, 2-5s) ────────────────────
    segments_with_times = _segments_from_audio(audio_path, audio_dur)
    log("segments", f"{len(segments_with_times)} segments, "
        f"last ends at {segments_with_times[-1]['end']:.1f}s")

    # ── Планування текстових оверлеїв ────────────────────────────────────────
    log("overlays", "Planning text overlays...")
    overlay_plan  = _plan_text_overlays(segments_with_times, emit=emit)
    text_overlays = _build_text_overlays(overlay_plan, segments_with_times)
    log("overlays", f"Planned {len(text_overlays)} text overlays.")

    # ── Підбір кліпів: 3 кандидати на сегмент → Gemini → найкращий ───────────
    clips_cache = os.path.join(proj_dir, "clips.json")
    if os.path.exists(clips_cache):
        with open(clips_cache, encoding="utf-8") as f:
            clip_data = json.load(f)
        # Підтримка старого формату (список рядків)
        if clip_data and isinstance(clip_data[0], str):
            clip_data = [{"file": c, "duration": 3.0} for c in clip_data if os.path.exists(c)]
        else:
            clip_data = [c for c in clip_data if os.path.exists(c.get("file", ""))]
        log("clips", f"Clips cached: {len(clip_data)} clips loaded from clips.json")
    else:
        log("clips", f"Selecting clips from '{movie_name}' (3 candidates/seg → Gemini)...")
        clip_data = _select_clips_for_segments(
            segments_with_times, movie_name, audio_dur,
            global_used_ids=global_used_ids,
        )
        if not clip_data:
            raise RuntimeError(f"No clips found for movie '{movie_name}'. Is it indexed?")
        log("clips", f"Selected {len(clip_data)} clips.")
        with open(clips_cache, "w", encoding="utf-8") as f:
            json.dump(clip_data, f, ensure_ascii=False)

    # ── Нормалізація + унікалізація (тривалість = тривалість сегменту) ────────
    log("clips", "Preparing clips (normalize + uniqualize)...")
    uniq_params = make_uniq_params_for_language(language, proj_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Parallel clip processing (4 workers = ~4x speed on multi-core)
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

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_prepare_one, (i, cd)) for i, cd in enumerate(clip_data)]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    prepared.append(result)

        # Sort by index to preserve clip order
        prepared.sort(key=lambda x: x[0])
        prepared = [out for _, out in prepared]

        if not prepared:
            raise RuntimeError("No clips survived preparation.")

        log("montage", f"Assembling {len(prepared)} clips ({audio_dur:.1f}s audio)...")
        if emit:
            emit("montage", "Assembling video segments...")
        output_path = os.path.join(proj_dir, f"{proj_id}.mp4")
        _build_movie_video(
            clips         = prepared,
            audio_path    = audio_path,
            output_path   = output_path,
            text_overlays = text_overlays,
            proj_id       = proj_id,
            emit          = emit,
        )

    log("done", f"Video ready: {output_path}")

    meta_path = os.path.join(proj_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    # Collect used clip IDs for batch tracking
    used_ids_in_this_video = set()
    for cd in clip_data:
        if cd.get("id"):
            used_ids_in_this_video.add(cd["id"])
        else:
            used_ids_in_this_video.add(os.path.basename(cd.get("file", "")))

    return {
        "project_id":   proj_id,
        "project_dir":  proj_dir,
        "output_path":  output_path,
        "audio_dur":    round(audio_dur, 1),
        "clips_used":   len(prepared),
        "title":        meta.get("title", source_title),
        "all_titles":   meta.get("titles", []),
        "description":  meta.get("description", ""),
        "tags":         meta.get("tags", []),
        "used_ids":     list(used_ids_in_this_video),
    }


def produce_from_script(
    script: str,
    title: str,
    movie_name: str,
    language: str,
    metadata: dict = None,
    emit=None,
    global_used_ids: set = None,
) -> dict:
    """
    Produce a video from a pre-written script (Writer flow).
    Skips transcription and rewrite — goes straight to TTS → clips → montage.
    metadata: optional dict with keys title, titles, description, tags.
    """
    def log(step, msg):
        print(f"[movie_pipeline:from_script:{step}] {msg}", flush=True)
        if emit:
            emit(step, msg)

    proj_id  = f"writer_{language}_{int(time.time())}"
    proj_dir = os.path.join(config.PROJECTS_DIR, proj_id)
    os.makedirs(proj_dir, exist_ok=True)

    # Детермінований RNG для ефектів кліпів
    clip_seed = int.from_bytes(hashlib.md5(proj_id.encode()).digest()[:4], "big")
    clip_rng  = random.Random(clip_seed)

    # ── Save script ───────────────────────────────────────────────────────────
    script_path = os.path.join(proj_dir, "script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    if len(script.split()) < 100:
        raise RuntimeError(
            f"Script too short ({len(script.split())} words). "
            "Please provide a longer script."
        )

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = metadata or {}
    meta_path = os.path.join(proj_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log("script", f"Script: {len(script)} chars, {len(script.split())} words")

    # ── TTS ───────────────────────────────────────────────────────────────────
    audio_path = os.path.join(proj_dir, "voiceover.mp3")
    if not os.path.exists(audio_path):
        log("tts", "Generating voiceover...")
        tts.generate(script, language, audio_path)
        log("tts", "Voiceover done.")
    else:
        log("tts", "Voiceover cached.")

    audio_dur = _get_duration(audio_path)
    log("tts", f"Audio duration: {audio_dur:.1f}s")

    if audio_dur < MIN_AUDIO_DURATION:
        raise RuntimeError(
            f"Voiceover too short: {audio_dur:.1f}s (min {MIN_AUDIO_DURATION}s). "
            "Script may be too short or TTS failed."
        )

    # ── Segments with timestamps (Whisper by pauses, 2-5s) ───────────────────
    segments_with_times = _segments_from_audio(audio_path, audio_dur)
    log("segments", f"{len(segments_with_times)} segments, "
        f"last ends at {segments_with_times[-1]['end']:.1f}s")

    # ── Text overlays ─────────────────────────────────────────────────────────
    log("overlays", "Planning text overlays...")
    overlay_plan  = _plan_text_overlays(segments_with_times, emit=emit)
    text_overlays = _build_text_overlays(overlay_plan, segments_with_times)
    log("overlays", f"Planned {len(text_overlays)} text overlays.")

    # ── Clip selection: 3 candidates/seg → Gemini → best ─────────────────────
    log("clips", f"Selecting clips from '{movie_name}' (3 candidates/seg → Gemini)...")
    clip_data = _select_clips_for_segments(
        segments_with_times, movie_name, audio_dur,
        global_used_ids=global_used_ids,
    )
    if not clip_data:
        raise RuntimeError(f"No clips found for movie '{movie_name}'. Is it indexed?")
    log("clips", f"Selected {len(clip_data)} clips.")

    # ── Normalize + uniqualize (duration = segment duration) ──────────────────
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

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_prepare_one, (i, cd)) for i, cd in enumerate(clip_data)]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    prepared.append(result)

        prepared.sort(key=lambda x: x[0])
        prepared = [out for _, out in prepared]

        if not prepared:
            raise RuntimeError("No clips survived preparation.")

        log("montage", f"Assembling {len(prepared)} clips ({audio_dur:.1f}s audio)...")
        output_path = os.path.join(proj_dir, f"{proj_id}.mp4")
        _build_movie_video(
            clips         = prepared,
            audio_path    = audio_path,
            output_path   = output_path,
            text_overlays = text_overlays,
            proj_id       = proj_id,
        )

    log("done", f"Video ready: {output_path}")

    return {
        "project_id":  proj_id,
        "project_dir": proj_dir,
        "output_path": output_path,
        "audio_dur":   round(audio_dur, 1),
        "clips_used":  len(prepared),
        "title":       meta.get("title", title),
        "titles":      meta.get("titles", []),
        "description": meta.get("description", ""),
        "tags":        meta.get("tags", []),
    }



def produce_batch(prepare_id: str, movie_name: str, language: str,
                  count: int = 3, emit=None) -> list:
    """
    Виробляє count відео послідовно з одного prepare_id.

    Гарантії різноманітності:
    - Кожне відео має унікальний proj_id → різні uniq_params (zoom/brightness/flip)
    - global_used_ids передається між відео → різні кліпи в кожному відео
    - Різні ефекти (Ken Burns, speed, flash cuts) через різний clip_rng
    - Різні переходи між групами через різний rng в _build_movie_video

    Повертає список результатів produce() для кожного відео.
    """
    def log(msg):
        print(f"[movie_pipeline:batch] {msg}", flush=True)
        if emit:
            emit("batch", msg)

    count = max(1, min(count, 10))  # обмеження 1–10 відео
    log(f"Starting batch: {count} videos, movie='{movie_name}', lang='{language}'")

    results        = []
    global_used_ids = set()  # кліпи використані в попередніх відео

    for i in range(count):
        log(f"Video {i + 1}/{count} starting...")
        try:
            result = produce(
                prepare_id      = prepare_id,
                movie_name      = movie_name,
                language        = language,
                emit            = emit,
                global_used_ids = global_used_ids,
            )
            results.append({"index": i + 1, "status": "ok", **result})
            log(f"Video {i + 1}/{count} done: {result['output_path']}")

            # Оновлюємо глобальний пул використаних кліпів з повернутих used_ids
            returned_ids = result.get("used_ids", [])
            for cid in returned_ids:
                if cid:
                    global_used_ids.add(cid)
            log(f"Updated used clips: {len(global_used_ids)} total")

            time.sleep(1)

        except Exception as e:
            log(f"Video {i + 1}/{count} FAILED: {e}")
            results.append({"index": i + 1, "status": "error", "error": str(e)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    log(f"Batch done: {ok_count}/{count} videos successful")
    return results
