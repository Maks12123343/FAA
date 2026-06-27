"""
Semantic embeddings for clip matching.

Why this exists
---------------
The old clip selection matched script segments to clips by literal word overlap
(_score_clip). Narration like "he was left all alone" never matched a clip tagged
"isolation" because no words are shared, so the candidate pool was full of noise and
the LLM validator could only pick "the best of the noise". This module replaces that
first step with semantic vectors: text is turned into a vector of meaning, and clips
are ranked by cosine similarity, so meaning matches even when words differ.

Backends (auto-detected at runtime, in order):
  1. Pioneer.ai  /v1/embeddings   — OpenAI-compatible, uses the existing pioneer keys
                                     in parallel (one key per worker) for speed.
  2. Vertex AI   text-embedding-004 — fallback, uses the existing gcloud credentials.

If neither backend is available, embed_texts() returns None and callers fall back to
the legacy keyword search — nothing crashes.
"""

import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# Remember which backend works so we don't re-probe a dead one every call.
# Values: None (unknown), "pioneer", "vertex", "none"
_BACKEND_CACHE = {"kind": None}
_BACKEND_LOCK = threading.Lock()

# In-process cache of single-text embeddings (segment texts repeat across retries).
_TEXT_CACHE = {}
_TEXT_CACHE_LOCK = threading.Lock()

# OpenAI-compatible embeddings batch size per request.
_PIONEER_BATCH = 96
# Vertex text-embedding-004 accepts up to 250 instances; keep margin.
_VERTEX_BATCH = 100


# ── Math ────────────────────────────────────────────────────────────────────────

def cosine(a: list, b: list) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 on bad input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ── Pioneer backend ───────────────────────────────────────────────────────────

def _pioneer_keys() -> list:
    settings = config.load_settings()
    keys = settings.get("pioneer_api_keys", [])
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    return [k for k in keys if k]


def _pioneer_embed_url() -> str:
    settings = config.load_settings()
    chat_url = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")
    # Derive the embeddings endpoint from the chat endpoint.
    return settings.get("pioneer_embed_url", chat_url.replace("chat/completions", "embeddings"))


def _pioneer_embed_model() -> str:
    settings = config.load_settings()
    return settings.get("pioneer_embed_model", "text-embedding-004")


def _pioneer_embed_batch(texts: list, api_key: str, timeout: int = 120) -> list:
    """Embed one batch via Pioneer (OpenAI-compatible /v1/embeddings). Raises on failure."""
    payload = json.dumps({
        "model": _pioneer_embed_model(),
        "input": texts,
    }).encode("utf-8")
    req = urllib.request.Request(
        _pioneer_embed_url(),
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # OpenAI shape: {"data": [{"index": 0, "embedding": [...]}, ...]}
    data = body.get("data", [])
    data_sorted = sorted(data, key=lambda d: d.get("index", 0))
    vectors = [d.get("embedding") for d in data_sorted]
    if len(vectors) != len(texts) or any(v is None for v in vectors):
        raise RuntimeError(
            f"Pioneer embeddings returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors


def _pioneer_embed_all(texts: list, emit=None) -> list:
    """Embed all texts using every Pioneer key in parallel. Raises if any batch fails."""
    keys = _pioneer_keys()
    if not keys:
        raise RuntimeError("No Pioneer keys configured")

    batches = [texts[i:i + _PIONEER_BATCH] for i in range(0, len(texts), _PIONEER_BATCH)]
    results = [None] * len(batches)
    n_workers = min(len(keys), len(batches)) or 1

    def _work(batch_idx: int):
        key = keys[batch_idx % len(keys)]
        results[batch_idx] = _pioneer_embed_batch(batches[batch_idx], key)

    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_work, i): i for i in range(len(batches))}
        done = 0
        for f in as_completed(futures):
            try:
                f.result()
                done += 1
                if emit and len(batches) > 1:
                    emit("embed", f"Embedded batch {done}/{len(batches)} (Pioneer)")
            except Exception as e:
                errors.append(e)

    if errors:
        raise errors[0]

    out = []
    for r in results:
        out.extend(r)
    return out


# ── Vertex backend ──────────────────────────────────────────────────────────────

def _vertex_embed_all(texts: list, emit=None) -> list:
    """Embed all texts via Vertex AI text-embedding-004. Raises on failure."""
    from google import genai

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
    if not os.path.exists(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")):
        raise RuntimeError(
            f"Vertex credentials file not found: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')}"
        )
    settings = config.load_settings()
    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project_id", ""),
        location=settings.get("vertex_location", "us-central1"),
    )
    model = settings.get("vertex_embed_model", "text-embedding-004")

    out = []
    batches = [texts[i:i + _VERTEX_BATCH] for i in range(0, len(texts), _VERTEX_BATCH)]
    for bi, batch in enumerate(batches):
        last_err = None
        for attempt in range(4):
            try:
                resp = client.models.embed_content(model=model, contents=batch)
                break
            except Exception as e:
                last_err = e
                wait = 3 * (attempt + 1)
                print(f"[embeddings] Vertex batch {bi+1} attempt {attempt+1} failed: {e}, retry in {wait}s", flush=True)
                import time as _time
                _time.sleep(wait)
        else:
            raise RuntimeError(f"Vertex embeddings failed after 4 attempts: {last_err}")
        embs = getattr(resp, "embeddings", None) or []
        vectors = [list(getattr(e, "values", e)) for e in embs]
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Vertex embeddings returned {len(vectors)} vectors for {len(batch)} inputs"
            )
        out.extend(vectors)
        if emit and len(batches) > 1:
            emit("embed", f"Embedded batch {bi + 1}/{len(batches)} (Vertex)")
    return out


# ── Public API ──────────────────────────────────────────────────────────────────

def embed_texts(texts: list, emit=None) -> list:
    """
    Turn a list of strings into a list of vectors (list[float]).

    Tries Pioneer first, then Vertex. Caches which backend works so a dead backend
    is not re-probed on every call. Returns None if no backend is available — callers
    must handle None by falling back to keyword matching.
    """
    if not texts:
        return []

    # Sanitize: embedding APIs reject empty strings.
    clean = [(t if (t and t.strip()) else "untitled clip") for t in texts]

    order = _backend_order()
    last_err = None
    for kind in order:
        try:
            if kind == "pioneer":
                vecs = _pioneer_embed_all(clean, emit=emit)
            elif kind == "vertex":
                vecs = _vertex_embed_all(clean, emit=emit)
            else:
                continue
            _set_backend(kind)
            return vecs
        except Exception as e:
            last_err = e
            print(f"[embeddings] backend '{kind}' failed: {e}", flush=True)
            continue

    print(f"[embeddings] all backends failed ({last_err}); callers will use keyword fallback", flush=True)
    _set_backend("none")
    return None


def embed_text(text: str, emit=None) -> list:
    """Embed a single string. Returns a vector or None. Cached in-memory per process."""
    key = (text or "").strip()
    if not key:
        return None
    with _TEXT_CACHE_LOCK:
        if key in _TEXT_CACHE:
            return _TEXT_CACHE[key]
    vecs = embed_texts([key], emit=emit)
    vec = vecs[0] if vecs else None
    if vec:
        with _TEXT_CACHE_LOCK:
            # Bound the cache so a very long run can't grow it without limit.
            if len(_TEXT_CACHE) < 5000:
                _TEXT_CACHE[key] = vec
    return vec


def _backend_order() -> list:
    """Try Pioneer first, then Vertex AI fallback."""
    return ["pioneer", "vertex"]


def _set_backend(kind: str):
    with _BACKEND_LOCK:
        _BACKEND_CACHE["kind"] = kind


# ── Clip embedding text builder ───────────────────────────────────────────────

def clip_embed_text(clip: dict) -> str:
    """
    Build the text that represents a movie clip for embedding.
    Combines the semantic fields produced by Gemini at index time so the vector
    captures characters, emotion, scene and themes — not just the raw description.
    """
    parts = []
    desc = clip.get("description", "")
    if desc:
        parts.append(desc)
    chars = clip.get("characters", []) or []
    if chars:
        parts.append("Characters: " + ", ".join(chars))
    emotion = clip.get("emotion", "")
    if emotion:
        parts.append("Emotion: " + emotion)
    scene = clip.get("scene_type", "")
    if scene:
        parts.append("Scene: " + scene.replace("_", " "))
    themes = clip.get("themes", []) or []
    if themes:
        parts.append("Themes: " + ", ".join(t.replace("_", " ") for t in themes))
    tags = clip.get("tags", []) or []
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    text = ". ".join(parts).strip()
    return text or "untitled clip"
