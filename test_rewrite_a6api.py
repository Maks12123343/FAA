"""Standalone A6API rewrite test.

Checks the real production rewrite path end to end: transcript -> 3 chunks ->
rewrite -> length/quality checks -> metadata. Uses A6API only, ignores whatever
provider is selected in Settings, and never writes into data/settings.json or
the projects folder.

EDIT ONE LINE: put your A6API key into API_KEY below, then run:

    cd C:\\Users\\Ukraine\\FAA
    python test_rewrite_a6api.py

Optional overrides:
    python test_rewrite_a6api.py --language pl --chunks 3
    python test_rewrite_a6api.py --url https://www.youtube.com/watch?v=...
"""

# ─────────────────────────────────────────────────────────────────────────────
API_KEY = ""          # <-- paste the A6API key here (sk-...)
MODEL = "gpt-5.5"
API_URL = "https://a6api.com/v1/chat/completions"
REASONING_EFFORT = "high"
MAX_TOKENS = "12000"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

import config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the A6API rewrite path.")
    p.add_argument("--language", default="pl",
                   help="Any language except ja/ko (they use the two-stage path). Default: pl")
    p.add_argument("--chunks", type=int, default=3, help="Chunk count for the rewrite. Default: 3")
    p.add_argument("--url", default="", help="YouTube URL. Default: reuse a cached transcript.")
    p.add_argument("--keep", action="store_true", help="Keep the output folder instead of a temp dir.")
    p.add_argument("--timeout", type=int, default=600, help="Per-request timeout in seconds.")
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"\n  FAILED: {msg}")
    print("=" * 72)
    raise SystemExit(1)


def _resolve_key() -> str:
    key = (API_KEY or os.environ.get("A6API_KEY", "")).strip()
    if not key:
        print("Paste your A6API key into API_KEY at the top of this file")
        print("(or set the A6API_KEY environment variable), then run it again.")
        raise SystemExit(2)
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        _fail("the key contains non-ASCII characters - it is not a real API key")
    return key


def _newest_cached_transcript() -> tuple[str, dict]:
    """Reuse a transcript from an existing prepare state: no yt-dlp needed."""
    roots = [
        Path(config.PROJECTS_DIR),
        Path(r"G:\My Drive\workspace\FAA\projects"),
        Path(r"E:\Мій диск\workspace\FAA\projects"),
    ]
    best = None
    for root in roots:
        if not root.is_dir():
            continue
        for state_path in root.glob("_prepare_war_*/state.json"):
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, state_path)
    if not best:
        return "", {}
    with open(best[1], encoding="utf-8") as fh:
        state = json.load(fh)
    if not (state.get("transcript") or "").strip():
        return "", {}
    print(f"Transcript source : {best[1]}")
    return state["transcript"], state


def main() -> int:
    args = _parse_args()
    key = _resolve_key()

    if args.language.strip().lower() in {"ja", "japanese", "ko", "korean"}:
        _fail("ja/ko go through the two-stage translate-first path; pick another language")

    # Drop env vars that would override the forced config below.
    for stale in (
        "FAA_REWRITE_CHUNKS", "REWRITE_PROVIDER", "REWRITE_API_KEY", "REWRITE_API_URL",
        "REWRITE_MODEL", "REWRITE_REASONING_EFFORT", "REWRITE_MAX_TOKENS",
    ):
        os.environ.pop(stale, None)

    # Force A6API for this process only. The real settings file is untouched.
    base = config.load_settings()

    def patched_settings() -> dict:
        s = dict(base)
        s["rewrite_active_provider"] = "a6api"
        s["rewrite_fallback_provider"] = ""      # no silent switch to byesu
        s["rewrite_providers"] = {
            "a6api": {
                "name": "A6API",
                "api_key": key,
                "model": MODEL,
                "api_url": API_URL,
                "reasoning_effort": REASONING_EFFORT,
                "max_tokens": MAX_TOKENS,
            }
        }
        s["rewrite_api_key"] = key
        s["rewrite_api_url"] = API_URL
        s["rewrite_model"] = MODEL
        s["rewrite_chunks"] = max(1, min(10, args.chunks))
        s["rewrite_script_enabled"] = True
        s["rewrite_metadata_enabled"] = True
        s["rewrite_thumbnail_enabled"] = False   # no Flow, no image spend
        return s

    # Both backend modules do `import config`, so they share this one object.
    config.load_settings = patched_settings

    from backend import api_client
    from backend import languages as lang_utils
    from backend.rewriter import rewrite_all, _length_bounds, _rewrite_chunk_count

    language = args.language.strip().lower()
    language_name = lang_utils.configured_language_name(language)

    print("=" * 72)
    print("A6API REWRITE TEST")
    print("=" * 72)
    print(f"Provider          : A6API (forced; Settings not modified)")
    print(f"URL               : {API_URL}")
    print(f"Model             : {MODEL}   reasoning_effort={REASONING_EFFORT or '(none)'}")
    print(f"Key               : {key[:6]}...{key[-4:]}  ({len(key)} chars)")
    print(f"Language          : {language} -> {language_name}")
    print(f"Chunks            : {_rewrite_chunk_count()}")
    print(f"Fallback provider : (disabled for this test)")
    print()

    # ---- Step 1: connectivity -----------------------------------------------
    print("[1/3] Connectivity check (one tiny request)...")
    t0 = time.time()
    try:
        text, finish = api_client.call_rewrite_api(
            "You reply with exactly one word and nothing else.",
            [{"role": "user", "content": "Reply with the single word: READY"}],
            timeout=120,
            max_retries=2,
            step_label="ping",
        )
    except Exception as exc:
        _fail(f"the API did not answer: {exc}")
    reply = (text or "").strip()
    print(f"      answered in {time.time()-t0:.1f}s, finish={finish}, reply={reply[:60]!r}")
    if not reply:
        _fail("the API returned an empty body (key/quota/model problem)")
    print("      OK\n")

    # ---- Step 2: transcript -------------------------------------------------
    if args.url:
        print(f"[2/3] Fetching transcript from {args.url} ...")
        from backend.transcriber import get_transcript
        try:
            result = get_transcript(args.url)
        except Exception as exc:
            _fail(f"could not get the transcript: {exc}")
        transcript = result["text"]
        state = {"source_title": "", "source_description": "", "source_tags": []}
        print(f"      {len(transcript)} chars via {result.get('source')}")
    else:
        print("[2/3] Reusing a cached transcript from a previous prepare...")
        transcript, state = _newest_cached_transcript()
        if not transcript:
            _fail("no cached transcript found - rerun with --url <youtube link>")
        print(f"      {len(transcript)} chars")
        if state.get("source_title"):
            print(f"      source: {state['source_title'][:70]}")
    if len(transcript) < 3000:
        _fail(f"transcript too short for a real test: {len(transcript)} chars")
    print()

    # ---- Step 3: the real rewrite -------------------------------------------
    import tempfile
    out_dir = (
        str(APP_DIR / f"_test_rewrite_{language}_{int(time.time())}")
        if args.keep else tempfile.mkdtemp(prefix="faa_rewrite_test_")
    )
    os.makedirs(out_dir, exist_ok=True)
    min_chars, max_chars = _length_bounds(len(transcript))
    print(f"[3/3] Rewriting for real (this costs tokens and takes a few minutes)...")
    print(f"      target length: {min_chars}-{max_chars} chars of {len(transcript)}")
    print(f"      cache dir    : {out_dir}")
    print("-" * 72)

    t0 = time.time()
    try:
        result = rewrite_all(
            transcript=transcript,
            language=language,
            source_title=state.get("source_title", ""),
            source_description=state.get("source_description", ""),
            source_tags=state.get("source_tags", []),
            test_mode=False,
            cache_dir=out_dir,
        )
    except Exception as exc:
        import traceback
        print("-" * 72)
        traceback.print_exc()
        _fail(f"rewrite raised: {exc}")
    elapsed = time.time() - t0
    print("-" * 72)

    script = result.get("script", "")
    chunks_file = Path(out_dir) / "rewrite_chunks.json"
    chunk_entries = []
    if chunks_file.exists():
        with open(chunks_file, encoding="utf-8") as fh:
            chunk_entries = json.load(fh).get("entries", [])

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Elapsed           : {elapsed/60:.1f} min")
    print(f"Script length     : {len(script):,} chars "
          f"({round(len(script)/len(transcript)*100)}% of source)")
    print(f"Allowed range     : {min_chars:,}-{max_chars:,} chars")
    print(f"Chunks rewritten  : {len(chunk_entries)}  "
          f"{[len(e.get('part','')) for e in chunk_entries]}")
    print(f"Title             : {result.get('title','')[:70]}")
    print(f"Title options     : {len(result.get('titles', []))}")
    print(f"Description       : {len(result.get('description',''))} chars")
    print(f"Tags              : {len(result.get('tags', []))}")
    print()
    print("First 400 chars of the script:")
    print("  " + script[:400].replace("\n", "\n  "))
    print()

    problems = []
    if len(script) < min_chars:
        problems.append(f"script shorter than the minimum ({len(script)} < {min_chars})")
    if not result.get("title"):
        problems.append("no title produced")
    if not result.get("description"):
        problems.append("no description produced")
    if not result.get("tags"):
        problems.append("no tags produced")
    if len(chunk_entries) != _rewrite_chunk_count():
        problems.append(
            f"expected {_rewrite_chunk_count()} chunks, cached {len(chunk_entries)}"
        )

    if problems:
        print("VERDICT: PROBLEMS FOUND")
        for item in problems:
            print(f"  - {item}")
        print("=" * 72)
        return 1

    print("VERDICT: PASS - A6API rewrites correctly, chunking and metadata are fine.")
    if not args.keep:
        print(f"(temp folder {out_dir} can be deleted)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
