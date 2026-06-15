"""
Movie Library — індексація фільмів та підбір кліпів для cartoon-psychology ніші.
"""

import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG  = config.FFMPEG
FFPROBE = config.FFPROBE

SCENE_THRESHOLD    = 0.35
CLIP_MIN           = 2.0
CLIP_MAX           = 5.0
BATCH_SIZE         = 8      # кліпів за один Gemini запит
VALIDATION_THRESHOLD = 0.85  # Gemini validation score (як в FAA)

# Семантичний пошук: мінімальна косинусна близькість, нижче якої кліп
# вважається нерелевантним кандидатом. Використовується тільки для відсіву
# відвертого мотлоху — фінально вирішує Pioneer-валідація.
SEMANTIC_MIN_SIM = 0.20


def _is_gemini_auth_error(err) -> bool:
    text = str(err).lower()
    return any(x in text for x in (
        "401", "403", "permission", "credentials",
        "unauthenticated", "unauthorized",
    ))


# ── Шляхи ─────────────────────────────────────────────────────────────────────

def _movies_dir() -> str:
    return config.get_movies_dir()

def _movie_dir(movie_name: str) -> str:
    return os.path.join(_movies_dir(), movie_name)

def _clips_dir(movie_name: str) -> str:
    return os.path.join(_movie_dir(movie_name), "clips")

def _index_path(movie_name: str) -> str:
    return os.path.join(_movie_dir(movie_name), "index.json")


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=120,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _cut_clip(src: str, out: str, start: float, duration: float):
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}",
         "-c", "copy", "-an", out],
        capture_output=True, timeout=300,
    )


def _detect_scene_timestamps(video_path: str, total_dur: float) -> list:
    r = subprocess.run(
        [FFMPEG, "-threads", "6", "-i", video_path,
         "-vf", f"select=gt(scene\\,{SCENE_THRESHOLD}),showinfo",
         "-vsync", "vfr", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    timestamps = [0.0]
    if r.returncode == 0:
        for line in r.stderr.splitlines():
            if "showinfo" in line and "pts_time:" in line:
                m = re.search(r"pts_time:(\d+\.?\d*)", line)
                if m:
                    t = float(m.group(1))
                    if t > 0.1:
                        timestamps.append(t)
    timestamps.append(total_dur)
    return sorted(set(timestamps))


def _cut_by_scenes(src_path: str, out_dir: str, movie_id: str,
                   total_dur: float, emit=None) -> list:
    scene_times = _detect_scene_timestamps(src_path, total_dur)
    clips = []
    idx   = 0

    for i in range(len(scene_times) - 1):
        scene_start = scene_times[i]
        scene_end   = scene_times[i + 1]
        scene_dur   = scene_end - scene_start

        if scene_dur < CLIP_MIN:
            continue

        if scene_dur <= CLIP_MAX:
            out = os.path.join(out_dir, f"{movie_id}_{idx:04d}.mp4")
            _cut_clip(src_path, out, scene_start, scene_dur)
            if os.path.exists(out) and os.path.getsize(out) > 5000:
                clips.append({
                    "id":    f"{movie_id}_{idx:04d}",
                    "file":  out,
                    "start": round(scene_start, 2),
                    "end":   round(scene_end, 2),
                })
                idx += 1
        else:
            t = scene_start
            while t + CLIP_MIN <= scene_end:
                chunk = min(CLIP_MAX, scene_end - t)
                if chunk < CLIP_MIN:
                    break
                out = os.path.join(out_dir, f"{movie_id}_{idx:04d}.mp4")
                _cut_clip(src_path, out, t, chunk)
                if os.path.exists(out) and os.path.getsize(out) > 5000:
                    clips.append({
                        "id":    f"{movie_id}_{idx:04d}",
                        "file":  out,
                        "start": round(t, 2),
                        "end":   round(t + chunk, 2),
                    })
                    idx += 1
                t += chunk

        if emit and (i + 1) % 20 == 0:
            emit("movie", f"Cutting: {idx} clips so far...")

    return clips


# ── Gemini helpers ─────────────────────────────────────────────────────────────

def _gemini():
    from google import genai
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    settings = config.load_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    return client, settings.get("gemini_model", "gemini-2.5-flash")


def _frame_bytes(clip_path: str, ratio: float) -> bytes:
    dur = _get_duration(clip_path)
    ts  = max(0.01, dur * max(0.0, min(1.0, ratio)))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", clip_path,
         "-vframes", "1", "-vf", "scale=640:-2", "-q:v", "4", tmp.name],
        capture_output=True, timeout=120,
    )
    data = b""
    if os.path.exists(tmp.name):
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
    return data


_BATCH_PROMPT = """\
Analyze {n} video clips from the movie "{movie_name}".
For each clip, 3 frames are shown (start/middle/end), labeled CLIP 1, CLIP 2, etc.

For EACH clip return a JSON object with:
- characters: list of character names visible (e.g. "Tigress", "Po", "Shifu")
- emotion: one of: joy, sadness, fear, anger, determination, vulnerability, shame, guilt, pride, neutral
- scene_type: one of: training, fight, emotional_dialogue, rejection, acceptance, flashback, celebration, isolation, comedy, action, quiet_moment, transformation, credits, title_card
- themes: 2-4 from: [growth, trauma, impostor_syndrome, false_self, identity, rejection, acceptance, vulnerability, shame, fear, determination, healing, connection, isolation, anger, betrayal, grief, love, trust]
- description: 1-2 sentence visual description
- tags: 8-12 topic tags
- is_blurry: true if most frames are out of focus or heavily motion-blurred
- is_static: true if all 3 frames look nearly identical (frozen/no motion)

Reply ONLY with a JSON array of exactly {n} objects, no markdown."""


def _analyze_batch(items: list, movie_name: str, client, model: str) -> list:
    from google.genai import types

    contents = []
    for i, item in enumerate(items):
        contents.append(f"CLIP {i + 1}:")
        for fb in item["frames"]:
            contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    contents.append(_BATCH_PROMPT.format(n=len(items), movie_name=movie_name))

    r = client.models.generate_content(model=model, contents=contents)
    text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
    text = re.sub(r"\s*```$", "", text)
    m    = re.search(r"\[.*\]", text, re.DOTALL)
    raw  = json.loads(m.group() if m else text)

    results = []
    for i, item in enumerate(items):
        a = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        a.setdefault("characters", [])
        a.setdefault("emotion", "neutral")
        a.setdefault("scene_type", "quiet_moment")
        a.setdefault("themes", [])
        a.setdefault("description", "")
        a.setdefault("tags", [])
        a.setdefault("is_blurry", False)
        a.setdefault("is_static", False)
        results.append(a)
    return results


def _analyze_batch_gigacoder(items: list, movie_name: str) -> list:
    """Fallback: analyze clips via GigaCoder GPT-5.4-mini (multimodal)."""
    import base64
    import urllib.request

    settings = config.load_settings()
    gc_keys = settings.get("gigacoder_api_keys", [])
    gc_url = settings.get("gigacoder_api_url", "https://www.gigacoder.org/api/v1/chat/completions")
    gc_model = settings.get("gigacoder_model", "gpt-5.4-mini")
    if not gc_keys:
        raise RuntimeError("No gigacoder_api_keys for fallback")

    content_parts = []
    for i, item in enumerate(items):
        content_parts.append({"type": "text", "text": f"CLIP {i + 1}:"})
        for fb in item["frames"]:
            b64 = base64.b64encode(fb).decode("ascii")
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content_parts.append({"type": "text", "text": _BATCH_PROMPT.format(n=len(items), movie_name=movie_name)})

    payload = json.dumps({
        "model": gc_model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 4096,
    }).encode("utf-8")

    last_err = None
    for key in gc_keys:
        try:
            req = urllib.request.Request(
                gc_url, data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\[.*\]", text, re.DOTALL)
            raw = json.loads(m.group() if m else text)
            results = []
            for i, item in enumerate(items):
                a = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
                a.setdefault("characters", [])
                a.setdefault("emotion", "neutral")
                a.setdefault("scene_type", "quiet_moment")
                a.setdefault("themes", [])
                a.setdefault("description", "")
                a.setdefault("tags", [])
                a.setdefault("is_blurry", False)
                a.setdefault("is_static", False)
                results.append(a)
            return results
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"GigaCoder analyze fallback failed: {last_err}")


def _analysis_cached(clip_path: str) -> dict | None:
    ap = clip_path + ".analysis.json"
    if not os.path.exists(ap):
        return None
    try:
        st = os.stat(clip_path)
        with open(ap, encoding="utf-8") as f:
            cached = json.load(f)
        if (cached.get("_size") == st.st_size and
                cached.get("_mtime") == round(st.st_mtime, 2)):
            return cached
    except Exception:
        pass
    return None


def _save_analysis(clip_path: str, analysis: dict):
    try:
        st = os.stat(clip_path)
        analysis["_size"]  = st.st_size
        analysis["_mtime"] = round(st.st_mtime, 2)
    except Exception:
        pass
    ap = clip_path + ".analysis.json"
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)


def _analyze_all_clips(clips: list, movie_name: str, emit=None) -> list:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_analyze     = []
    cached_results = []

    for c in clips:
        cached = _analysis_cached(c["file"])
        if cached:
            cached["id"]   = c["id"]
            cached["file"] = c["file"]
            cached_results.append(cached)
        else:
            to_analyze.append(c)

    if not to_analyze:
        return cached_results

    print(f"[movie_library] Analyzing {len(to_analyze)} clips "
          f"({len(cached_results)} cached)...", flush=True)

    def _extract_frames(clip):
        frames = []
        for ratio in [0.0, 0.5, 1.0]:
            fb = _frame_bytes(clip["file"], ratio)
            if fb:
                frames.append(fb)
        return {"clip": clip, "frames": frames}

    with ThreadPoolExecutor(max_workers=2) as pool:
        items = list(pool.map(_extract_frames, to_analyze))
    items = [it for it in items if it["frames"]]

    batches     = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    done        = [0]
    lock        = threading.Lock()
    results     = []
    worker_errors = []
    client, model = _gemini()

    def _process_batch(batch):
        for attempt in range(3):
            try:
                analyses = _analyze_batch(batch, movie_name, client, model)
                for item, analysis in zip(batch, analyses):
                    clip = item["clip"]
                    analysis["id"]   = clip["id"]
                    analysis["file"] = clip["file"]
                    _save_analysis(clip["file"], analysis)
                    with lock:
                        done[0] += 1
                        results.append(analysis)
                    if emit:
                        emit("movie", f"Analyzed {done[0]}/{len(items)} clips...")
                return
            except Exception as e:
                err = str(e).lower()
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[movie_library] Gemini auth/config error: {e}") from e
                is_rate = "429" in str(e) or "quota" in err or "resource_exhausted" in err
                if is_rate and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                else:
                    print(f"[movie_library] Gemini batch error, trying GigaCoder: {e}", flush=True)
                    try:
                        analyses = _analyze_batch_gigacoder(batch, movie_name)
                        for item, analysis in zip(batch, analyses):
                            clip = item["clip"]
                            analysis["id"]   = clip["id"]
                            analysis["file"] = clip["file"]
                            _save_analysis(clip["file"], analysis)
                            with lock:
                                done[0] += 1
                                results.append(analysis)
                            if emit:
                                emit("movie", f"Analyzed {done[0]}/{len(items)} clips...")
                        return
                    except Exception as gc_e:
                        print(f"[movie_library] GigaCoder fallback also failed: {gc_e}", flush=True)
                    for item in batch:
                        clip = item["clip"]
                        fallback = {
                            "characters": [], "emotion": "neutral",
                            "scene_type": "quiet_moment", "themes": [],
                            "description": "unknown", "tags": [],
                            "is_blurry": False, "is_static": False,
                            "id": clip["id"], "file": clip["file"],
                        }
                        _save_analysis(clip["file"], fallback)
                        with lock:
                            done[0] += 1
                            results.append(fallback)
                    return

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[movie_library] Worker error: {e}", flush=True)
                worker_errors.append(e)

    if worker_errors:
        raise RuntimeError(f"[movie_library] Clip analysis failed: {worker_errors[0]}")

    return cached_results + results


# ── Gemini validation (0.85, як в FAA) ────────────────────────────────────────

def _validation_cache_path(clip_path: str, section_text: str) -> str:
    import hashlib
    h = hashlib.md5(section_text[:300].encode()).hexdigest()[:12]
    return clip_path + f".movval_{h}.json"


def validate_clip(clip_path: str, section_text: str) -> float:
    """
    Gemini оцінює наскільки кліп підходить для сегменту нарації.
    Результат кешується поряд з кліпом.
    Повертає float 0.0–1.0.
    """
    cache_path = _validation_cache_path(clip_path, section_text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return float(json.load(f).get("score", 0.0))
        except Exception:
            pass

    from google.genai import types
    client, model = _gemini()

    parts = []
    for ratio in [0.0, 0.5, 1.0]:
        fb = _frame_bytes(clip_path, ratio)
        if fb:
            parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    if not parts:
        return 0.0

    prompt = (
        "These are 3 frames from an animated movie clip.\n"
        f'The narration script section this clip should illustrate: "{section_text[:300]}"\n'
        "Rate how well this clip fits this narration — considering the characters shown, "
        "their emotional state, and the psychological theme being discussed.\n"
        'JSON only: {"score": 0.0} where 0.0=completely wrong, 1.0=perfect fit.'
    )
    parts.append(prompt)

    got_response = False
    try:
        r = client.models.generate_content(model=model, contents=parts)
        text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
        text = re.sub(r"\s*```$", "", text)
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group() if m else text)
        score = float(data.get("score", 0.0))
        got_response = True
    except Exception as e:
        if _is_gemini_auth_error(e):
            raise RuntimeError(f"[movie_library] Gemini auth/config error: {e}") from e
        score = 0.0

    if got_response:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"score": score}, f)
        except Exception:
            pass

    return score


_BATCH_VALIDATION_PROMPT = """\
You are evaluating {n} animated movie clips against a narration segment.

Narration: "{section_text}"

For each clip, 3 frames are shown (start/middle/end), labeled CLIP 1, CLIP 2, etc.

Rate how well each clip fits this narration — considering characters shown,
their emotional state, and the psychological theme being discussed.

Reply ONLY with a JSON array of exactly {n} numbers (0.0–1.0), e.g. [0.9, 0.3, 0.7]
where 0.0=completely wrong, 1.0=perfect fit. No markdown."""


def validate_clips_batch(clip_paths: list, section_text: str) -> list:
    """
    Batch Gemini validation: оцінює список кліпів за один запит.
    Повертає list[float] тієї ж довжини що й clip_paths.
    Кешує кожен результат окремо.
    """
    from google.genai import types

    # Перевіряємо кеш для кожного кліпу
    scores   = [None] * len(clip_paths)
    to_fetch = []  # (original_index, clip_path)

    for i, cp in enumerate(clip_paths):
        cache_path = _validation_cache_path(cp, section_text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    scores[i] = float(json.load(f).get("score", 0.0))
                continue
            except Exception:
                pass
        to_fetch.append((i, cp))

    if not to_fetch:
        return scores

    client, model = _gemini()

    # Збираємо фрейми для некешованих кліпів
    items = []
    for orig_idx, cp in to_fetch:
        frames = []
        for ratio in [0.0, 0.5, 1.0]:
            fb = _frame_bytes(cp, ratio)
            if fb:
                frames.append(fb)
        items.append({"orig_idx": orig_idx, "clip_path": cp, "frames": frames})

    # Відправляємо батчами по BATCH_SIZE
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        contents = []
        for j, item in enumerate(batch):
            contents.append(f"CLIP {j + 1}:")
            for fb in item["frames"]:
                contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
        contents.append(_BATCH_VALIDATION_PROMPT.format(
            n=len(batch), section_text=section_text[:300]
        ))

        batch_scores = None
        for attempt in range(3):
            try:
                r    = client.models.generate_content(model=model, contents=contents)
                text = re.sub(r"^```(?:json)?\s*", "", r.text.strip())
                text = re.sub(r"\s*```$", "", text)
                m    = re.search(r"\[.*?\]", text, re.DOTALL)
                raw  = json.loads(m.group() if m else text)
                if isinstance(raw, list) and len(raw) >= len(batch):
                    batch_scores = [float(x) for x in raw[:len(batch)]]
                break
            except Exception as e:
                if _is_gemini_auth_error(e):
                    raise RuntimeError(f"[movie_library] Gemini auth error: {e}") from e
                is_rate = "429" in str(e) or "quota" in str(e).lower()
                if is_rate and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                else:
                    print(f"[movie_library] Batch validation error: {e}", flush=True)
                    break

        for j, item in enumerate(batch):
            s = (batch_scores[j] if batch_scores and j < len(batch_scores) else 0.0)
            scores[item["orig_idx"]] = s
            try:
                cache_path = _validation_cache_path(item["clip_path"], section_text)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"score": s}, f)
            except Exception:
                pass

    # Заповнюємо None → 0.0 на випадок помилок
    return [s if s is not None else 0.0 for s in scores]


# ── Uniqualization ─────────────────────────────────────────────────────────────

def _uniqualize_movie_clip(input_path: str, output_path: str, params: dict):
    """Агресивніша унікалізація для кліпів фільму."""
    grain = random.uniform(8, 15)
    flip  = params.get("flip", False)

    vf_parts = [
        f"scale=iw*{params['zoom']:.4f}:ih*{params['zoom']:.4f}",
        "crop=1920:1080",
        (f"eq=brightness={params['brightness']:.3f}"
         f":contrast={params['contrast']:.3f}"
         f":saturation={params['saturation']:.3f}"),
        f"noise=alls={grain:.1f}:allf=t+u",
    ]
    if flip:
        vf_parts.append("hflip")

    subprocess.run(
        [FFMPEG, "-y", "-i", input_path,
         "-vf", ",".join(vf_parts),
         *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p", "-an",
         output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    )


def make_uniq_params() -> dict:
    return {
        "zoom":       random.uniform(1.04, 1.08),
        "brightness": random.uniform(-0.05, 0.05),
        "contrast":   random.uniform(0.95, 1.08),
        "saturation": random.uniform(0.90, 1.15),
        "flip":       random.random() < 0.30,
    }


# ── Семантичні вектори ───────────────────────────────────────────────────────

def _attach_embeddings(clips: list, emit=None) -> int:
    """
    Дорахувати semantic embedding для кліпів, у яких його ще немає.
    Вектор будується з текстового опису (description+tags+characters+emotion+
    scene_type+themes), які Gemini вже згенерував — відео не відкривається.
    Записує вектор у clip["embedding"] in-place.
    Повертає кількість дорахованих векторів. Якщо embeddings недоступні —
    мовчки повертає 0 (кліпи лишаються без векторів, пошук впаде на keyword).
    """
    from backend import embeddings as _emb

    need = [c for c in clips if not c.get("embedding")]
    if not need:
        return 0

    texts = [_emb.clip_embed_text(c) for c in need]
    if emit:
        emit("movie", f"Computing semantic vectors for {len(need)} clips...")
    try:
        vectors = _emb.embed_texts(texts, emit=emit)
    except Exception as e:
        print(f"[movie_library] Embedding computation failed: {e}", flush=True)
        vectors = None

    if not vectors:
        print("[movie_library] No embeddings produced — clips left without vectors "
              "(search will use keyword fallback)", flush=True)
        return 0

    count = 0
    for c, v in zip(need, vectors):
        if v:
            c["embedding"] = v
            count += 1
    print(f"[movie_library] Attached embeddings to {count}/{len(need)} clips", flush=True)
    return count


# ── Публічний API ──────────────────────────────────────────────────────────────

def process_movie(movie_path: str, movie_name: str, emit=None) -> dict:
    """
    Нарізати фільм на кліпи, проаналізувати через Gemini, зберегти індекс в GDrive.
    Якщо фільм вже проіндексований — повертає кеш.
    """
    def log(msg):
        print(f"[movie_library:{movie_name}] {msg}", flush=True)
        if emit:
            emit("movie", msg)

    idx_path = _index_path(movie_name)
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        log(f"Already indexed: {len(index['clips'])} clips")
        return index

    clips_out_dir = _clips_dir(movie_name)
    os.makedirs(clips_out_dir, exist_ok=True)

    log("Getting duration...")
    total_dur = _get_duration(movie_path)
    if total_dur < 10:
        raise ValueError(f"Movie too short or unreadable: {movie_path}")

    log(f"Duration: {total_dur / 60:.1f} min. Detecting scene changes...")
    movie_id = re.sub(r"[^\w]", "_", movie_name.lower())[:20]
    clips    = _cut_by_scenes(movie_path, clips_out_dir, movie_id, total_dur, emit=emit)
    log(f"Cut {len(clips)} clips. Starting Gemini analysis...")

    analyzed = _analyze_all_clips(clips, movie_name, emit=emit)

    # Відфільтрувати blurry/static
    good = [a for a in analyzed
            if not a.get("is_blurry") and not a.get("is_static")]
    log(f"Analysis done: {len(good)}/{len(analyzed)} clips passed quality check")

    # Дорахувати семантичні вектори (для семантичного підбору кліпів)
    _attach_embeddings(good, emit=emit)

    index = {
        "movie_name": movie_name,
        "movie_id":   movie_id,
        "total_dur":  total_dur,
        "clips":      good,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    log(f"Index saved: {idx_path}")
    return index


def process_movie_folder(folder_path: str, movie_name: str, emit=None) -> dict:
    """
    Обробити всі mp4 файли в папці як один фільм.
    Підтримує інкрементальну індексацію — нові файли додаються до існуючого індексу.
    Зберігає прогрес після кожного файлу (якщо впаде — можна продовжити).
    """
    def log(msg):
        print(f"[movie_library:{movie_name}] {msg}", flush=True)
        if emit:
            emit("movie", msg)

    # Знайти всі mp4 файли
    try:
        all_files = sorted([
            os.path.join(folder_path, fn)
            for fn in os.listdir(folder_path)
            if fn.lower().endswith(".mp4")
        ])
    except Exception as e:
        raise ValueError(f"Cannot read folder: {folder_path}: {e}")

    if not all_files:
        raise ValueError(f"No mp4 files found in: {folder_path}")

    log(f"Found {len(all_files)} mp4 file(s): {[os.path.basename(f) for f in all_files]}")

    idx_path      = _index_path(movie_name)
    clips_out_dir = _clips_dir(movie_name)
    os.makedirs(clips_out_dir, exist_ok=True)

    # Завантажити існуючий індекс (якщо є)
    existing_index    = {}
    processed_sources = set()
    all_good_clips    = []
    total_dur         = 0.0
    movie_id          = re.sub(r"[^\w]", "_", movie_name.lower())[:20]

    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            existing_index = json.load(f)
        processed_sources = set(existing_index.get("processed_sources", []))
        all_good_clips    = existing_index.get("clips", [])
        total_dur         = existing_index.get("total_dur", 0.0)
        log(f"Existing index: {len(all_good_clips)} clips, "
            f"{len(processed_sources)} source(s) already processed")

    new_files_processed = 0

    for file_idx, movie_path in enumerate(all_files):
        src_key = os.path.basename(movie_path)
        if src_key in processed_sources:
            log(f"Already processed: {src_key} — skipping")
            continue

        log(f"[{file_idx + 1}/{len(all_files)}] Processing: {src_key}")
        dur = _get_duration(movie_path)
        if dur < 10:
            log(f"Skipping (too short or unreadable): {src_key}")
            continue

        log(f"Duration: {dur / 60:.1f} min. Detecting scene changes...")
        # Унікальний префікс для кожного файлу — щоб кліпи не перезаписувались
        file_prefix = f"{movie_id}_f{file_idx:02d}"
        clips = _cut_by_scenes(movie_path, clips_out_dir, file_prefix, dur, emit=emit)
        log(f"Cut {len(clips)} clips. Starting Gemini analysis...")

        analyzed = _analyze_all_clips(clips, movie_name, emit=emit)
        good = [a for a in analyzed
                if not a.get("is_blurry") and not a.get("is_static")]
        log(f"{len(good)}/{len(analyzed)} clips passed quality check for {src_key}")

        # Дорахувати семантичні вектори новим кліпам перед збереженням
        _attach_embeddings(good, emit=emit)

        all_good_clips.extend(good)
        total_dur += dur
        processed_sources.add(src_key)
        new_files_processed += 1

        # Зберігаємо після кожного файлу — щоб не втратити прогрес при збої
        index = {
            "movie_name":        movie_name,
            "movie_id":          movie_id,
            "total_dur":         total_dur,
            "clips":             all_good_clips,
            "processed_sources": sorted(processed_sources),
            "created_at":        existing_index.get(
                                     "created_at",
                                     time.strftime("%Y-%m-%dT%H:%M:%S")),
            "updated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        log(f"Index saved: {len(all_good_clips)} total clips so far")

    if new_files_processed == 0:
        log(f"All files already processed. Total: {len(all_good_clips)} clips")
    else:
        log(f"Done! Processed {new_files_processed} new file(s). "
            f"Total: {len(all_good_clips)} clips")

    return {
        "movie_name": movie_name,
        "clips":      all_good_clips,
        "total_dur":  total_dur,
    }


def list_movies() -> list:
    """Список всіх проіндексованих фільмів."""
    movies_dir = _movies_dir()
    if not os.path.exists(movies_dir):
        return []
    result = []
    for name in os.listdir(movies_dir):
        idx = _index_path(name)
        if os.path.exists(idx):
            try:
                with open(idx, encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "name":       name,
                    "clip_count": len(data.get("clips", [])),
                    "duration":   data.get("total_dur", 0),
                    "created_at": data.get("created_at", ""),
                })
            except Exception:
                pass
    return result


def get_movie_clips(movie_name: str) -> list:
    """Всі кліпи з індексу фільму."""
    idx_path = _index_path(movie_name)
    if not os.path.exists(idx_path):
        return []
    with open(idx_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("clips", [])


_REJECT_KEYWORDS = {
    "credits", "credit", "end credits", "opening credits", "title card",
    "title screen", "text screen", "intertitle", "author", "authors",
    "directed by", "produced by", "written by", "cast", "crew",
    "copyright", "logo", "studio logo", "black screen", "blank",
    "the end", "fin",
}


def _score_clip(clip: dict, segment_text: str) -> float:
    """Розумний keyword-based скор без Gemini. Fuzzy matching:
    - Shifu знаходить Master Shifu
    - po знаходить kung fu panda, young panda
    - anger знаходить anger, angry, outburst of anger
    Rejects: credits, title cards, text-only screens → score = -1
    """
    desc_lower = clip.get("description", "").lower()
    tags_lower = [t.lower() for t in clip.get("tags", [])]
    scene_type = clip.get("scene_type", "").lower()

    for kw in _REJECT_KEYWORDS:
        if kw in desc_lower or kw in scene_type:
            return -1.0
        for tag in tags_lower:
            if kw in tag:
                return -1.0

    text_lower = segment_text.lower()
    text_words = set(text_lower.split())
    score = 0.0

    def _word_match(query: str, text: str, full_bonus: float = 1.0, partial_bonus: float = 0.4) -> float:
        """Повертає бонус якщо query входить в text (повністю або частково)."""
        q = query.lower().strip()
        if not q:
            return 0.0
        # Full match
        if q in text:
            return full_bonus
        # Partial: Shifu matches Master Shifu
        words = text.split()
        for w in words:
            if q in w or w in q:
                return partial_bonus
        return 0.0

    # Characters: "Master Shifu" matches "Shifu teaches Po" -> Shifu = partial match
    for char in clip.get("characters", []):
        score += _word_match(char, text_lower, full_bonus=5.0, partial_bonus=3.0)

    # Emotion: strong match
    if clip.get("emotion"):
        score += _word_match(clip["emotion"], text_lower, full_bonus=4.0, partial_bonus=2.5)

    # Scene type
    if clip.get("scene_type"):
        score += _word_match(clip["scene_type"], text_lower, full_bonus=3.0, partial_bonus=1.5)

    # Themes: each word of theme checked separately
    for theme in clip.get("themes", []):
        for w in theme.replace("_", " ").split():
            if len(w) > 2:
                score += _word_match(w, text_lower, full_bonus=2.5, partial_bonus=1.5)

    # Tags: full tag match + partial word match
    for tag in clip.get("tags", []):
        score += _word_match(tag, text_lower, full_bonus=2.0, partial_bonus=1.0)

    # Description: words > 4 chars
    for word in clip.get("description", "").lower().split():
        if len(word) > 4:
            score += _word_match(word, text_lower, full_bonus=0.8, partial_bonus=0.3)

    return score


def _has_embeddings(clips: list) -> bool:
    """
    True якщо семантичний пошук доцільний — тобто векторами покрита БІЛЬШІСТЬ
    кліпів. При частковому бекфілі (вектори лише в частини кліпів) повертаємо
    False, щоб не загубити кліпи без векторів — тоді працює keyword по всіх.
    """
    if not clips:
        return False
    with_emb = sum(1 for c in clips if c.get("embedding"))
    return with_emb >= len(clips) * 0.8


def _semantic_rank(segment_text: str, clips: list, used_ids: set, top_n: int) -> list:
    """
    Ранжувати кліпи за косинусною близькістю їхнього вектора до вектора сегмента.
    Кліпи-кредити/титри відсіюються (як і в keyword-режимі).
    Повертає список clip dict (найрелевантніші першими), без рандому.
    Якщо вектор сегмента порахувати не вдалось — повертає None (→ keyword fallback).
    """
    from backend import embeddings as _emb

    seg_vec = _emb.embed_text(segment_text)
    if not seg_vec:
        return None

    scored = []
    for clip in clips:
        if clip.get("id") in used_ids:
            continue
        if not os.path.exists(clip.get("file", "")):
            continue
        emb = clip.get("embedding")
        if not emb:
            continue
        # Відсів кредитів/титрів/текстових екранів (та сама логіка, що в _score_clip)
        if _score_clip(clip, segment_text) < 0:
            continue
        sim = _emb.cosine(seg_vec, emb)
        if sim >= SEMANTIC_MIN_SIM:
            scored.append((sim, clip))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]


def search_clips(segment_text: str, movie_name: str = None,
                 used_ids: set = None, top_n: int = 15,
                 gemini_validate: bool = False) -> list:
    """
    Знайти кліпи що підходять до тексту сегменту нарації.

    Крок 1: семантичний пошук за embedding-векторами (за СЕНСОМ, не за словами).
            Якщо вектори відсутні або недоступні — fallback на keyword scoring.
    Крок 2: Gemini validation 0.85 (якщо gemini_validate=True).
    Fallback: якщо нічого не пройшло валідацію — повертає top-5 без валідації.
    """
    if movie_name:
        all_clips = get_movie_clips(movie_name)
    else:
        all_clips = []
        for m in list_movies():
            all_clips.extend(get_movie_clips(m["name"]))

    used_ids = used_ids or set()

    # ── Крок 1: семантичний пошук (пріоритетний) ──
    top = None
    if _has_embeddings(all_clips):
        semantic = _semantic_rank(segment_text, all_clips, used_ids, top_n)
        if semantic is not None:
            top = semantic

    # ── Fallback: keyword scoring (якщо немає векторів або їх не порахувати) ──
    if top is None:
        candidates = []
        for clip in all_clips:
            if clip.get("id") in used_ids:
                continue
            if not os.path.exists(clip.get("file", "")):
                continue
            s = _score_clip(clip, segment_text)
            if s > 0:
                candidates.append((s, clip))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in candidates[:top_n]]

    if not gemini_validate or not top:
        return top

    # Gemini validation: оцінюємо топ-10, фільтруємо >= 0.85
    validated = []
    for clip in top[:10]:
        gem_score = validate_clip(clip["file"], segment_text)
        if gem_score >= VALIDATION_THRESHOLD:
            validated.append((gem_score, clip))

    if validated:
        validated.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in validated]

    # Fallback — повертаємо top-5 без валідації
    return top[:5]
