"""Швидка перевірка живих API ключів. Робить короткий PING-запит до кожного."""
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(__file__))


def _short(k: str) -> str:
    if not k:
        return "(empty)"
    return f"{k[:12]}...{k[-6:]}"


def _ping_openai_compat(url: str, key: str, model: str, label: str) -> tuple:
    """Send a tiny PONG request to an OpenAI-compatible chat/completions endpoint."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly one word: PONG"},
        ],
        "max_tokens": 20,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        elapsed = time.time() - t0
        msg = body.get("choices", [{}])[0].get("message", {}) or {}
        content = (msg.get("content") or msg.get("reasoning") or "").strip()[:40]
        return True, f"OK ({elapsed:.1f}s): {content!r}"
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            body = ""
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def main():
    settings_path = os.path.join(os.path.dirname(__file__), "data", "settings.json")
    settings = json.load(open(settings_path, encoding="utf-8"))

    print("=" * 70)
    print("Pioneer API keys")
    print("=" * 70)
    pio_url = settings.get("pioneer_api_url", "")
    pio_model = settings.get("pioneer_model", "gemini-3.5-flash")
    pio_keys = settings.get("pioneer_api_keys", [])
    for i, k in enumerate(pio_keys, 1):
        ok, msg = _ping_openai_compat(pio_url, k, pio_model, "pioneer")
        print(f"  Key #{i} ({_short(k)}): {msg}")

    pio_rewrite = settings.get("pioneer_rewrite_key", "").strip()
    pio_rewrite_model = settings.get("pioneer_rewrite_model", "claude-opus-4-7")
    if pio_rewrite:
        ok, msg = _ping_openai_compat(pio_url, pio_rewrite, pio_rewrite_model, "pioneer-rewrite")
        print(f"  Rewrite ({_short(pio_rewrite)}, model={pio_rewrite_model}): {msg}")

    print()
    print("=" * 70)
    print("GigaCoder API keys")
    print("=" * 70)
    gc_url = settings.get("gigacoder_api_url", "")
    gc_model = settings.get("gigacoder_model", "gpt-5.4-mini")
    gc_keys = settings.get("gigacoder_api_keys", [])
    for i, k in enumerate(gc_keys, 1):
        ok, msg = _ping_openai_compat(gc_url, k, gc_model, "gigacoder")
        print(f"  Key #{i} ({_short(k)}): {msg}")

    gc_rewrite = settings.get("gigacoder_rewrite_key", "").strip()
    gc_rewrite_model = settings.get("gigacoder_rewrite_model", "claude-opus-4-8")
    if gc_rewrite:
        ok, msg = _ping_openai_compat(gc_url, gc_rewrite, gc_rewrite_model, "gigacoder-rewrite")
        print(f"  Rewrite ({_short(gc_rewrite)}, model={gc_rewrite_model}): {msg}")

    print()
    print("=" * 70)
    print("YouTube Data API")
    print("=" * 70)
    for key_name in ("youtube_api_key", "youtube_api_key_2", "youtube_api_key_3"):
        k = settings.get(key_name, "").strip()
        if not k:
            print(f"  {key_name}: (empty, skipped)")
            continue
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=test&maxResults=1&key={k}"
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
            elapsed = time.time() - t0
            print(f"  {key_name} ({_short(k)}): OK ({elapsed:.1f}s) — YouTube responded")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            print(f"  {key_name} ({_short(k)}): HTTP {e.code}: {body}")
        except Exception as e:
            print(f"  {key_name} ({_short(k)}): {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 70)
    print("Pexels API")
    print("=" * 70)
    pexels_keys = settings.get("pexels_api_keys", [])
    if isinstance(pexels_keys, str):
        pexels_keys = [pexels_keys]
    for i, k in enumerate(pexels_keys, 1):
        if not k:
            continue
        url = "https://api.pexels.com/videos/search?query=nature&per_page=1"
        req = urllib.request.Request(url, headers={"Authorization": k})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
            elapsed = time.time() - t0
            n = len(data.get("videos", []))
            print(f"  Key #{i} ({_short(k)}): OK ({elapsed:.1f}s), {n} video(s)")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            print(f"  Key #{i} ({_short(k)}): HTTP {e.code}: {body}")
        except Exception as e:
            print(f"  Key #{i} ({_short(k)}): {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 70)
    print("Vertex embeddings (text-embedding-004)")
    print("=" * 70)
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"
        from backend import embeddings as _emb
        t0 = time.time()
        vec = _emb.embed_text("hello world")
        elapsed = time.time() - t0
        if vec and len(vec) > 100:
            print(f"  Vertex: OK ({elapsed:.1f}s), vector dim={len(vec)}")
        else:
            print(f"  Vertex: RETURNED EMPTY (elapsed={elapsed:.1f}s)")
    except Exception as e:
        print(f"  Vertex: {type(e).__name__}: {str(e)[:300]}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
