"""
CLIP-based clip classifier for the "library" niche pipeline.

Workflow
--------
1. User puts source mp4 files in <library_root>/_sources/
2. process_library(niche_name) is called.
3. Each unprocessed source file is scene-cut into 2-5s clips via FFmpeg
   (same scene-detection logic as movie_library.py).
4. For each clip, 3 frames are extracted (start/middle/end) and embedded with
   open_clip (ViT-B/32, downloaded once, runs locally — no API).
5. Frame embeddings are averaged into one vector per clip.
6. Each clip is classified into one of the niche's categories by cosine
   similarity to category description embeddings.
7. Clip is moved into <library_root>/<category>/. Clips with confidence below
   the niche's min_classification_confidence go to <library_root>/_unsorted/.
8. Progress, error counts, and per-category counts are returned/streamed.

The classifier model is cached in memory so the second run for the same niche
is much faster — no model reloading.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG = config.FFMPEG
FFPROBE = config.FFPROBE

# Same defaults as movie_library — niche may override via JSON
_DEFAULT_SCENE_THRESHOLD = 0.30
_DEFAULT_CLIP_MIN = 2.0
_DEFAULT_CLIP_MAX = 5.0
_DEFAULT_MIN_CONFIDENCE = 0.18

# Cache the loaded CLIP model so we don't pay the cold-start cost twice.
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None
_CLIP_LOCK = threading.Lock()


# ── Paths ─────────────────────────────────────────────────────────────────────

def _library_root(niche_name: str) -> str:
    """
    Each library niche stores its data under <stocks_dir>/../library/<niche>/
    Sources go into _sources/, sorted clips into the category folders.
    """
    stocks_dir = config.get_stocks_dir()
    parent = os.path.dirname(stocks_dir.rstrip("/\\"))
    return os.path.join(parent, "library", niche_name)


def _sources_dir(niche_name: str) -> str:
    return os.path.join(_library_root(niche_name), "_sources")


def _unsorted_dir(niche_name: str) -> str:
    return os.path.join(_library_root(niche_name), "_unsorted")


def _state_path(niche_name: str) -> str:
    return os.path.join(_library_root(niche_name), "_state.json")


def _category_dir(niche_name: str, category: str) -> str:
    return os.path.join(_library_root(niche_name), category)


# ── State ─────────────────────────────────────────────────────────────────────

def _load_state(niche_name: str) -> dict:
    p = _state_path(niche_name)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_sources": [], "categorized_clips": 0, "unsorted_clips": 0}


def _save_state(niche_name: str, state: dict):
    p = _state_path(niche_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=120,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _cut_clip(src: str, out: str, start: float, duration: float):
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src,
         "-t", f"{duration:.3f}", "-c", "copy", "-an", out],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=300,
    )


def _detect_scene_timestamps(video_path: str, total_dur: float, scene_threshold: float, emit=None) -> list:
    """Stream FFmpeg scene-detection and report progress every 30s of video time."""
    proc = subprocess.Popen(
        [FFMPEG, "-threads", "0", "-i", video_path,
         "-vf", f"select=gt(scene\\,{scene_threshold}),showinfo",
         "-vsync", "vfr", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    timestamps = [0.0]
    last_emit = 0.0
    progress_re = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")
    pts_re = re.compile(r"pts_time:(\d+\.?\d*)")
    start_time = time.time()

    if emit:
        emit("library", f"Scene detection started ({total_dur/60:.1f} min)")

    try:
        for line in proc.stderr:
            if "showinfo" in line:
                m = pts_re.search(line)
                if m:
                    t = float(m.group(1))
                    if t > 0.1:
                        timestamps.append(t)
            m = progress_re.search(line)
            if m:
                hh, mm, ss = m.groups()
                cur = int(hh) * 3600 + int(mm) * 60 + float(ss)
                if cur - last_emit >= 30 or cur >= total_dur - 1:
                    last_emit = cur
                    pct = min(100, int(cur / max(1, total_dur) * 100))
                    msg = (
                        f"Scene detection: {cur/60:.1f}/{total_dur/60:.1f} min "
                        f"({pct}%) — found {len(timestamps)-1} scenes"
                    )
                    print(f"[clip_classifier] {msg}", flush=True)
                    if emit:
                        emit("library", msg)
        proc.wait()
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"Scene detection failed: {e}") from e

    timestamps.append(total_dur)
    return sorted(set(timestamps))


def _frame_bytes(clip_path: str, ratio: float) -> bytes:
    """Extract one JPEG frame at the given relative position (0-1)."""
    dur = _get_duration(clip_path)
    ts = max(0.01, dur * max(0.0, min(1.0, ratio)))
    fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    data = b""
    try:
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", clip_path,
             "-vframes", "1", "-vf", "scale=336:-2", "-q:v", "3", out_path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                data = f.read()
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass
    return data


# ── CLIP model ────────────────────────────────────────────────────────────────

def _ensure_clip_model():
    """
    Load open_clip ViT-B/32 once. Uses GPU if available (CUDA), else CPU.
    Returns (model, preprocess, tokenizer, device).
    """
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    with _CLIP_LOCK:
        if _CLIP_MODEL is not None:
            return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[clip_classifier] Loading open_clip ViT-B-32 on {device}...", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
        _CLIP_TOKENIZER = tokenizer
        _CLIP_DEVICE = device
        print(f"[clip_classifier] Model ready ({device})", flush=True)
        return model, preprocess, tokenizer, device


def _embed_text_categories(category_descriptions: dict) -> dict:
    """
    Embed each category description into a unit-normalized vector.
    Returns {category_name: tensor_on_device}.
    """
    import torch
    model, _, tokenizer, device = _ensure_clip_model()
    names = list(category_descriptions.keys())
    texts = [category_descriptions[n] for n in names]
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return {n: feats[i] for i, n in enumerate(names)}


def _embed_clip_frames(clip_path: str) -> "torch.Tensor | None":
    """
    Extract 3 frames (start/middle/end), encode each with CLIP image encoder,
    average the embeddings into one unit-normalized vector. Returns None if
    no frames could be extracted.
    """
    import io
    import torch
    from PIL import Image
    model, preprocess, _, device = _ensure_clip_model()

    images = []
    for ratio in (0.0, 0.5, 1.0):
        fb = _frame_bytes(clip_path, ratio)
        if not fb:
            continue
        try:
            img = Image.open(io.BytesIO(fb)).convert("RGB")
            images.append(preprocess(img))
        except Exception:
            continue
    if not images:
        return None

    with torch.no_grad():
        batch = torch.stack(images).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        avg = feats.mean(dim=0)
        avg = avg / avg.norm()
    return avg


def _classify_clip(clip_path: str, cat_vectors: dict, min_conf: float) -> tuple:
    """
    Classify one clip. Returns (category_name, confidence).
    Category is "_unsorted" if the top score is below min_conf.
    """
    import torch
    avg = _embed_clip_frames(clip_path)
    if avg is None:
        return "_unsorted", 0.0
    best_name = None
    best_score = -1.0
    for name, vec in cat_vectors.items():
        score = float(torch.dot(avg, vec).item())
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < min_conf:
        return "_unsorted", best_score
    return best_name, best_score


# ── Cutting and classifying a single source file ──────────────────────────────

def _cut_source_into_clips(src_path: str, out_dir: str, src_id: str,
                           total_dur: float, scene_threshold: float,
                           clip_min: float, clip_max: float, emit=None) -> list:
    """Scene-cut one source mp4 into many short clips."""
    scene_times = _detect_scene_timestamps(src_path, total_dur, scene_threshold, emit=emit)

    clips = []
    idx = 0
    total_segs = len(scene_times) - 1
    last_emit = time.time()

    for i in range(total_segs):
        scene_start = scene_times[i]
        scene_end = scene_times[i + 1]
        scene_dur = scene_end - scene_start

        if scene_dur < clip_min:
            continue

        if scene_dur <= clip_max:
            out = os.path.join(out_dir, f"{src_id}_{idx:04d}.mp4")
            _cut_clip(src_path, out, scene_start, scene_dur)
            if os.path.exists(out) and os.path.getsize(out) > 5000:
                clips.append(out)
                idx += 1
        else:
            t = scene_start
            while t + clip_min <= scene_end:
                chunk = min(clip_max, scene_end - t)
                if chunk < clip_min:
                    break
                out = os.path.join(out_dir, f"{src_id}_{idx:04d}.mp4")
                _cut_clip(src_path, out, t, chunk)
                if os.path.exists(out) and os.path.getsize(out) > 5000:
                    clips.append(out)
                    idx += 1
                t += chunk

        now = time.time()
        if now - last_emit >= 2.0 or (i + 1) % 25 == 0:
            last_emit = now
            pct = int((i + 1) / max(1, total_segs) * 100)
            msg = f"Cutting: {idx} clips so far ({i+1}/{total_segs} segments, {pct}%)"
            print(f"[clip_classifier] {msg}", flush=True)
            if emit:
                emit("library", msg)

    return clips


# ── Public API ────────────────────────────────────────────────────────────────

def process_library(niche_name: str, niche_cfg: dict, emit=None) -> dict:
    """
    Run the full pipeline for one library-mode niche:
    cut scenes from every new source mp4, classify clips, move to category folders.

    niche_cfg must contain at least:
      - categories: dict[str, str]   # name → text description for CLIP
    Optional:
      - scene_threshold (default 0.30)
      - clip_min_duration (default 2)
      - clip_max_duration (default 5)
      - min_classification_confidence (default 0.18)
    """
    def log(msg: str):
        print(f"[clip_classifier:{niche_name}] {msg}", flush=True)
        if emit:
            emit("library", msg)

    categories = niche_cfg.get("categories") or {}
    if not categories:
        raise ValueError(f"Niche '{niche_name}' has no 'categories' defined")

    scene_threshold = float(niche_cfg.get("scene_threshold", _DEFAULT_SCENE_THRESHOLD))
    clip_min = float(niche_cfg.get("clip_min_duration", _DEFAULT_CLIP_MIN))
    clip_max = float(niche_cfg.get("clip_max_duration", _DEFAULT_CLIP_MAX))
    min_conf = float(niche_cfg.get("min_classification_confidence", _DEFAULT_MIN_CONFIDENCE))

    root = _library_root(niche_name)
    sources = _sources_dir(niche_name)
    unsorted = _unsorted_dir(niche_name)

    os.makedirs(root, exist_ok=True)
    os.makedirs(sources, exist_ok=True)
    os.makedirs(unsorted, exist_ok=True)
    for cat in categories.keys():
        os.makedirs(_category_dir(niche_name, cat), exist_ok=True)

    # Discover source mp4 files we haven't processed yet
    state = _load_state(niche_name)
    processed = set(state.get("processed_sources", []))
    try:
        all_sources = sorted([
            fn for fn in os.listdir(sources) if fn.lower().endswith(".mp4")
        ])
    except FileNotFoundError:
        raise ValueError(f"Sources directory not found: {sources}. Put your mp4 files there.")

    new_sources = [fn for fn in all_sources if fn not in processed]
    log(f"Found {len(all_sources)} source file(s); {len(new_sources)} new to process")

    if not new_sources:
        return {
            "niche": niche_name,
            "new_sources": 0,
            "categorized_clips": state.get("categorized_clips", 0),
            "unsorted_clips": state.get("unsorted_clips", 0),
            "categories": _count_per_category(niche_name, categories.keys()),
        }

    # Embed category descriptions once
    log("Embedding category descriptions with CLIP...")
    cat_vectors = _embed_text_categories(categories)

    total_categorized = state.get("categorized_clips", 0)
    total_unsorted = state.get("unsorted_clips", 0)

    for src_idx, src_fn in enumerate(new_sources):
        src_path = os.path.join(sources, src_fn)
        src_id = re.sub(r"[^\w]", "_", os.path.splitext(src_fn)[0])[:40]
        log(f"[{src_idx+1}/{len(new_sources)}] Processing: {src_fn}")

        dur = _get_duration(src_path)
        if dur < 10:
            log(f"Skipping {src_fn}: too short ({dur:.1f}s)")
            processed.add(src_fn)
            state["processed_sources"] = sorted(processed)
            _save_state(niche_name, state)
            continue

        log(f"Duration: {dur/60:.1f} min. Detecting scenes...")
        # Cut into a temp staging dir so we can move files atomically per category
        staging = os.path.join(root, "_staging", src_id)
        os.makedirs(staging, exist_ok=True)
        try:
            clips = _cut_source_into_clips(
                src_path, staging, src_id, dur,
                scene_threshold, clip_min, clip_max, emit=emit,
            )
        except Exception as e:
            log(f"ERROR cutting {src_fn}: {e}")
            shutil.rmtree(staging, ignore_errors=True)
            continue

        log(f"Cut {len(clips)} clips. Classifying with CLIP...")

        last_log = time.time()
        per_cat = {c: 0 for c in categories}
        per_cat["_unsorted"] = 0

        for ci, clip_path in enumerate(clips):
            try:
                cat, score = _classify_clip(clip_path, cat_vectors, min_conf)
            except Exception as e:
                log(f"Classify error on {os.path.basename(clip_path)}: {e}")
                cat, score = "_unsorted", 0.0

            dst_dir = unsorted if cat == "_unsorted" else _category_dir(niche_name, cat)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(clip_path))
            try:
                shutil.move(clip_path, dst)
                per_cat[cat] = per_cat.get(cat, 0) + 1
                if cat == "_unsorted":
                    total_unsorted += 1
                else:
                    total_categorized += 1
            except Exception as e:
                log(f"Move error: {e}")

            now = time.time()
            if now - last_log >= 3.0 or ci + 1 == len(clips):
                last_log = now
                msg = (
                    f"Classifying: {ci+1}/{len(clips)} "
                    f"({int((ci+1)/len(clips)*100)}%) — "
                    + ", ".join(f"{c}:{per_cat.get(c,0)}" for c in list(categories.keys())[:4])
                    + f", _unsorted:{per_cat.get('_unsorted',0)}"
                )
                log(msg)

        # Per-source done — drop empty staging dir
        shutil.rmtree(staging, ignore_errors=True)
        processed.add(src_fn)
        state["processed_sources"] = sorted(processed)
        state["categorized_clips"] = total_categorized
        state["unsorted_clips"] = total_unsorted
        _save_state(niche_name, state)
        log(f"Done {src_fn}: " + ", ".join(f"{c}:{per_cat.get(c,0)}" for c in per_cat))

    counts = _count_per_category(niche_name, list(categories.keys()) + ["_unsorted"])
    log(f"All done. Final counts: {counts}")
    return {
        "niche": niche_name,
        "new_sources": len(new_sources),
        "categorized_clips": total_categorized,
        "unsorted_clips": total_unsorted,
        "categories": counts,
    }


def _count_per_category(niche_name: str, categories) -> dict:
    out = {}
    for c in categories:
        d = _unsorted_dir(niche_name) if c == "_unsorted" else _category_dir(niche_name, c)
        try:
            out[c] = sum(1 for fn in os.listdir(d) if fn.lower().endswith(".mp4"))
        except FileNotFoundError:
            out[c] = 0
    return out


def list_clips_in_category(niche_name: str, category: str) -> list:
    """List full paths of clips in one category (for clip pool building)."""
    d = _unsorted_dir(niche_name) if category == "_unsorted" else _category_dir(niche_name, category)
    try:
        return sorted([
            os.path.join(d, fn) for fn in os.listdir(d)
            if fn.lower().endswith(".mp4")
        ])
    except FileNotFoundError:
        return []
