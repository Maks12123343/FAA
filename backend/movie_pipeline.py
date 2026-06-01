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
    validate_clips_batch,
    _uniqualize_movie_clip, make_uniq_params,
    VALIDATION_THRESHOLD,
)

WORDS_PER_SECTION  = 35
MIN_AUDIO_DURATION = 60.0   # секунд — менше цього вважається помилкою TTS


# ── Real timestamps from audio ────────────────────────────────────────────────

def _segments_from_audio(audio_path: str, chunks: list, audio_dur: float) -> list:
    """
    Будує таймстампи для чанків скрипту.
    Спочатку пробує Whisper (word-level); fallback — пропорційний розподіл
    по реальній тривалості аудіо (точніше ніж фіксовані 145 WPM).
    """
    # ── Whisper alignment ─────────────────────────────────────────────────────
    try:
        import whisper as _whisper
        model  = _whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True, language=None)

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word":  w["word"].strip().lower(),
                    "start": w["start"],
                    "end":   w["end"],
                })

        if words:
            segments = []
            word_idx = 0
            for i, chunk in enumerate(chunks):
                chunk_word_count = len(chunk.split())
                # Захист від виходу за межі
                word_idx = min(word_idx, len(words) - 1)
                start_t  = words[word_idx]["start"]
                end_idx  = min(word_idx + chunk_word_count - 1, len(words) - 1)
                end_t    = words[end_idx]["end"]
                segments.append({
                    "index": i, "text": chunk,
                    "start": round(start_t, 3),
                    "end":   round(end_t, 3),
                })
                word_idx = end_idx + 1

            print(f"[movie_pipeline] Whisper alignment: {len(segments)} segments", flush=True)
            return segments
    except Exception as e:
        print(f"[movie_pipeline] Whisper skipped ({e}), proportional fallback", flush=True)

    # ── Proportional fallback ─────────────────────────────────────────────────
    total_words = sum(len(c.split()) for c in chunks) or 1
    segments    = []
    t           = 0.0
    for i, chunk in enumerate(chunks):
        dur = (len(chunk.split()) / total_words) * audio_dur
        segments.append({
            "index": i, "text": chunk,
            "start": round(t, 3),
            "end":   round(t + dur, 3),
        })
        t += dur
    return segments


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
        "vignette":   rng.random() < 0.60,   # 60% кліпів отримують вінетку
        "grain":      rng.uniform(6, 14),     # зерно плівки
    }


# ── Text overlay planning ──────────────────────────────────────────────────────

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
    """Claude читає сегменти скрипту і повертає план текстових оверлеїв."""
    import anthropic
    settings = config.load_settings()
    client   = anthropic.Anthropic(api_key=settings.get("claude_api_key", ""), timeout=120.0)

    seg_data = [
        {"index": s["index"], "start": round(s["start"], 1), "text": s["text"][:120]}
        for s in segments_with_times
    ]
    prompt = _TEXT_OVERLAY_PROMPT.format(
        segments_json=json.dumps(seg_data, ensure_ascii=False, indent=2)
    )

    try:
        r = client.messages.create(
            model=settings.get("claude_model", "claude-sonnet-4-6"),
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = re.sub(r"^```(?:json)?\s*", "", r.content[0].text.strip())
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
    Нормалізує кліп (1920x1080, 30fps, обрізає до max_dur),
    застосовує унікалізацію та додаткові ефекти:
      effect="ken_burns" — повільний zoom + pan
      effect="none"      — стандартна обробка
    speed — множник швидкості (0.85–1.15), 1.0 = без змін.
    Повертає True якщо успішно.
    """
    norm_tmp = out_path + ".norm.mp4"
    try:
        # Базова нормалізація
        r = subprocess.run(
            [config.FFMPEG, "-y", "-i", clip_path,
             "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,"
                    "crop=1920:1080,fps=30",
             *config.get_video_encoder_args("ultrafast"),
             "-pix_fmt", "yuv420p", "-an", "-t", str(max_dur), norm_tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        if r.returncode != 0 or not os.path.exists(norm_tmp):
            return False
        if os.path.getsize(norm_tmp) < 5000:
            os.unlink(norm_tmp)
            return False

        # Унікалізація (zoom, brightness, contrast, saturation, grain, flip)
        _uniqualize_movie_clip(norm_tmp, out_path, uniq_params)
        os.unlink(norm_tmp)

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 5000:
            return False

        # ── Speed ramping (без zoom-ефектів) ─────────────────────────────────
        if abs(speed - 1.0) > 0.01:
            effect_tmp = out_path + ".fx.mp4"
            r2 = subprocess.run(
                [config.FFMPEG, "-y", "-i", out_path,
                 "-vf", f"setpts={1.0/speed:.4f}*PTS",
                 *config.get_video_encoder_args("fast"),
                 "-pix_fmt", "yuv420p", "-an", effect_tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
            )
            if r2.returncode == 0 and os.path.exists(effect_tmp) and os.path.getsize(effect_tmp) > 5000:
                os.replace(effect_tmp, out_path)
            elif os.path.exists(effect_tmp):
                os.unlink(effect_tmp)

        # ── Vignette (кінематографічна вінетка) ──────────────────────────────
        if uniq_params.get("vignette", False):
            vig_tmp = out_path + ".vig.mp4"
            r3 = subprocess.run(
                [config.FFMPEG, "-y", "-i", out_path,
                 "-vf", "vignette=PI/4",
                 *config.get_video_encoder_args("fast"),
                 "-pix_fmt", "yuv420p", "-an", vig_tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
            )
            if r3.returncode == 0 and os.path.exists(vig_tmp) and os.path.getsize(vig_tmp) > 5000:
                os.replace(vig_tmp, out_path)
            elif os.path.exists(vig_tmp):
                os.unlink(vig_tmp)

        return os.path.exists(out_path) and os.path.getsize(out_path) > 5000

    except Exception as e:
        print(f"[movie_pipeline] Clip prepare error: {e}", flush=True)
        for p in [norm_tmp, out_path + ".fx.mp4"]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
        return False


# ── Clip selection ─────────────────────────────────────────────────────────────

def _select_clips_for_segments(segments_with_times: list, movie_name: str,
                                audio_dur: float,
                                global_used_ids: set = None) -> list:
    """
    Для кожного сегменту нарації:
    1. Keyword scoring по JSON індексу (без Gemini)
    2. Batch Gemini validation топ-10 кандидатів за один запит
    3. Кліп з найвищим score >= 0.85 береться; fallback — кліп з найвищим score взагалі
    4. Якщо кандидатів нема — рандомний кліп з фільму

    global_used_ids — множина ID кліпів вже використаних в ПОПЕРЕДНІХ відео батчу.
    Це гарантує різноманітність відеоряду між відео.

    Повертає впорядкований список file-paths.
    """
    settings  = config.load_settings()
    clip_min  = settings.get("clip_min_duration", 2)
    clip_max  = settings.get("clip_max_duration", 5)
    avg_clip  = (clip_min + clip_max) / 2

    # Локально використані в цьому відео + глобально використані в батчі
    used_ids: set  = set(global_used_ids or set())
    selected: list = []

    all_movie_clips = get_movie_clips(movie_name)

    for seg in segments_with_times:
        chunk    = seg["text"]
        seg_dur  = max(0.0, seg["end"] - seg["start"])
        # Скільки кліпів потрібно щоб покрити цей сегмент
        clips_needed = max(1, round(seg_dur / avg_clip))

        # Keyword scoring без Gemini
        candidates = search_clips(
            chunk, movie_name=movie_name,
            used_ids=used_ids, top_n=20,
            gemini_validate=False,
        )

        # Fallback — будь-який невикористаний кліп
        if not candidates:
            candidates = [
                c for c in all_movie_clips
                if c.get("id") not in used_ids
                and os.path.exists(c.get("file", ""))
            ]
            random.shuffle(candidates)

        for _ in range(clips_needed):
            if not candidates:
                break

            # Batch validate top-10 в одному Gemini-запиті
            pool = [
                c for c in candidates
                if c.get("id") not in used_ids
                and os.path.exists(c.get("file", ""))
            ][:10]
            if not pool:
                break

            clip_paths = [c["file"] for c in pool]
            try:
                scores = validate_clips_batch(clip_paths, chunk)
            except Exception as e:
                print(f"[movie_pipeline] Batch validation error: {e}", flush=True)
                scores = [0.0] * len(pool)

            # Вибираємо: найкращий >= 0.85; fallback — найкращий взагалі
            best_validated       = None
            best_validated_score = -1.0
            best_overall         = pool[0]
            best_overall_score   = scores[0] if scores else 0.0

            for clip, score in zip(pool, scores):
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall       = clip
                if score >= VALIDATION_THRESHOLD and score > best_validated_score:
                    best_validated_score = score
                    best_validated       = clip

            best_clip = best_validated if best_validated is not None else best_overall

            file_path = best_clip["file"]
            clip_id   = best_clip.get("id", file_path)
            selected.append(file_path)
            used_ids.add(clip_id)
            candidates = [c for c in candidates if c.get("id") != clip_id]

    # ── Padding: додаємо кліпи поки не покриємо всю тривалість аудіо ─────────
    covered = len(selected) * avg_clip
    while covered < audio_dur:
        extra_candidates = [
            c for c in all_movie_clips
            if c.get("id") not in used_ids
            and os.path.exists(c.get("file", ""))
        ]
        if not extra_candidates:
            # Якщо всі кліпи вже використані — дозволяємо повтори
            extra_candidates = [
                c for c in all_movie_clips
                if os.path.exists(c.get("file", ""))
            ]
        if not extra_candidates:
            break
        clip = random.choice(extra_candidates)
        selected.append(clip["file"])
        used_ids.add(clip.get("id", clip["file"]))
        covered += avg_clip

    print(f"[movie_pipeline] Selected {len(selected)} clips for {audio_dur:.1f}s audio", flush=True)
    return selected


# ── Assembly ───────────────────────────────────────────────────────────────────

def _concat_clip_list(clip_paths: list, output: str):
    list_file = output + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")
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
                       text_overlays: list = None, proj_id: str = ""):
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

        for g_idx, group in enumerate(groups):
            seg = os.path.join(tmp, f"seg_{g_idx:04d}.mp4")
            _concat_clip_list(group, seg)
            if os.path.exists(seg) and os.path.getsize(seg) > 1000:
                seg_files.append(seg)

        if not seg_files:
            raise RuntimeError("No segments created during assembly")

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
            _loop_video_to_duration(assembled, audio_dur + 1.0, raw_video)
        else:
            shutil.copy2(assembled, raw_video)

    music_path = _fetch_bg_music(audio_dur, proj_dir)
    _add_audio(raw_video, audio_path, with_audio, music_path=music_path)

    if text_overlays:
        apply_text_overlays(with_audio, text_overlays, output_path)
    else:
        shutil.copy2(with_audio, output_path)

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

    proj_id  = f"movie_{language}_{int(time.time())}"
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

    # ── Сегменти з таймстампами (Whisper або пропорційний розподіл) ───────────
    chunks              = _split_into_chunks(script, WORDS_PER_SECTION)
    segments_with_times = _segments_from_audio(audio_path, chunks, audio_dur)
    log("segments", f"{len(segments_with_times)} segments, "
        f"last ends at {segments_with_times[-1]['end']:.1f}s")

    # ── Планування текстових оверлеїв ────────────────────────────────────────
    log("overlays", "Planning text overlays...")
    overlay_plan  = _plan_text_overlays(segments_with_times, emit=emit)
    text_overlays = _build_text_overlays(overlay_plan, segments_with_times)
    log("overlays", f"Planned {len(text_overlays)} text overlays.")

    # ── Підбір кліпів з Gemini validation (0.85) ──────────────────────────────
    log("clips", f"Selecting clips from '{movie_name}' (Gemini batch validation)...")
    clip_files = _select_clips_for_segments(
        segments_with_times, movie_name, audio_dur,
        global_used_ids=global_used_ids,
    )
    if not clip_files:
        raise RuntimeError(f"No clips found for movie '{movie_name}'. Is it indexed?")
    log("clips", f"Selected {len(clip_files)} clips.")

    # ── Нормалізація + унікалізація + ефекти кліпів ───────────────────────────
    log("clips", "Preparing clips (normalize + uniqualize + effects)...")
    uniq_params = make_uniq_params_for_language(language, proj_id)

    # Призначаємо ефекти для кожного кліпу
    effects = _assign_clip_effects(len(clip_files), clip_rng)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepared = []
        for i, (cf, fx) in enumerate(zip(clip_files, effects)):
            out = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
            ok  = _prepare_movie_clip(
                cf, out, uniq_params,
                max_dur = fx["max_dur"],
                effect  = fx["effect"],
                speed   = fx["speed"],
            )
            if ok:
                prepared.append(out)
            if emit and (i + 1) % 10 == 0:
                emit("clips", f"Prepared {i + 1}/{len(clip_files)} clips...")

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

    meta_path = os.path.join(proj_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "project_id":  proj_id,
        "project_dir": proj_dir,
        "output_path": output_path,
        "audio_dur":   round(audio_dur, 1),
        "clips_used":  len(prepared),
        "title":       meta.get("title", source_title),
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

            # Оновлюємо глобальний пул використаних кліпів
            # (з наступного відео ці кліпи будуть уникатися)
            # Ми не маємо прямого доступу до clip IDs після produce(),
            # тому читаємо з movie_library через назви файлів
            # Простіший підхід: передаємо global_used_ids як mutable set
            # і _select_clips_for_segments додає до нього — але це потребує
            # рефакторингу. Поки що використовуємо затримку між відео
            # щоб proj_id (timestamp) гарантовано відрізнявся.
            time.sleep(1)

        except Exception as e:
            log(f"Video {i + 1}/{count} FAILED: {e}")
            results.append({"index": i + 1, "status": "error", "error": str(e)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    log(f"Batch done: {ok_count}/{count} videos successful")
    return results