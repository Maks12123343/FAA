"""Повноцінний тест rewrite pipeline — з реальним payload'ом ~17k символів.
Це імітує ту саму нагрузку, що падала в проді з HTTP 0.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

# Force reload settings (не через кеш)
import config
if hasattr(config, "_settings_cache"):
    config._settings_cache = None

from backend import api_client, rewriter


# Синтетичний "транскрипт" ~17k символів (як реальне відео)
SEED = (
    "The war in Ukraine has entered a decisive phase. "
    "Russian forces continue to press along the eastern front while Ukrainian troops "
    "hold defensive positions near key logistical hubs. "
    "Analysts believe that the coming weeks will determine the trajectory of the entire campaign. "
    "Drones now dominate the battlefield, replacing much of what was once artillery-driven combat. "
    "Both sides deploy FPV drones daily, targeting armored vehicles, supply routes, and command posts. "
    "The Ukrainian defense industry has scaled drone production to over a million units per year. "
    "Meanwhile, Western partners continue debating the next tranche of military aid. "
    "Long-range strikes into Russian territory have become more frequent and more precise. "
    "Refineries, air bases, and ammunition depots have all been hit in the past month. "
    "The Kremlin publicly downplays these losses but internally has reorganized air defense priorities. "
    "In the Black Sea, Ukraine's naval drone campaign has pushed the Russian fleet further from Crimea. "
    "This has opened critical grain corridors and restored a share of pre-war export volumes. "
)


def build_transcript(target_chars: int = 17000) -> str:
    buf = []
    total = 0
    i = 0
    while total < target_chars:
        buf.append(f"Section {i+1}. " + SEED)
        total += len(buf[-1])
        i += 1
    return "\n\n".join(buf)[:target_chars]


def test_ping_curl(rewrite_key: str, model: str):
    """Тест 1: маленький PONG-запит через curl (як робить сайт Pioneer)."""
    print("=" * 70)
    print(f"TEST 1: Small curl PING (model={model})")
    print("=" * 70)

    import subprocess
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: PONG"}],
        "stream": False,
    })
    cmd = [
        "curl", "-sS", "-X", "POST",
        "https://api.pioneer.ai/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {rewrite_key}",
        "-d", payload,
        "-w", "\n---HTTP:%{http_code} TIME:%{time_total}s---",
        "--max-time", "60",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    elapsed = time.time() - t0
    out = result.stdout[-800:]
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {out}")
    print()
    return "HTTP:200" in out


def test_ping_python(rewrite_key: str, model: str):
    """Тест 2: PONG через requests (точно як пайплайн у прод)."""
    print("=" * 70)
    print(f"TEST 2: Small PING via requests (model={model})")
    print("=" * 70)

    t0 = time.time()
    try:
        text, stop = api_client.call_pioneer(
            "You are a helpful assistant.",
            [{"role": "user", "content": "Reply with exactly one word: PONG"}],
            timeout=60,
            max_retries=1,
        )
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s): text={text!r}, stop={stop}")
        print()
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {e}")
        print()
        return False


def test_medium_payload(rewrite_key: str, model: str):
    """Тест 3: середній payload ~3k символів."""
    print("=" * 70)
    print(f"TEST 3: Medium payload ~3k chars (model={model})")
    print("=" * 70)

    transcript = build_transcript(3000)
    print(f"Transcript length: {len(transcript)} chars")

    t0 = time.time()
    try:
        text, stop = api_client.call_pioneer(
            "You are a professional voiceover script rewriter. Rewrite the following transcript, keeping approximately the same length. Respond with rewritten text only, wrapped in ```.",
            [{"role": "user", "content": f"Target language: en\n\n{transcript}"}],
            timeout=180,
            max_retries=1,
        )
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s): got {len(text)} chars, stop={stop}")
        print(f"First 200 chars: {text[:200]!r}")
        print()
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {str(e)[:400]}")
        print()
        return False


def test_full_rewrite(rewrite_key: str, model: str):
    """Тест 4: повний rewrite через rewriter._rewrite_script (~17k chars — як у прод)."""
    print("=" * 70)
    print(f"TEST 4: FULL production-style rewrite ~17k chars (model={model})")
    print("=" * 70)

    transcript = build_transcript(17000)
    print(f"Transcript length: {len(transcript)} chars (matches production video)")

    t0 = time.time()
    try:
        script = rewriter._rewrite_script(
            transcript=transcript,
            language="en",
            video_title="The Coming Turning Point in Ukraine",
            feedback="",
            test_mode=True,  # використовує коротший prompt
        )
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s): script={len(script)} chars")
        print(f"First 300 chars: {script[:300]!r}")
        print()
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {str(e)[:400]}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    settings = config.load_settings()
    rewrite_key = settings.get("pioneer_rewrite_key", "").strip()
    model = settings.get("pioneer_rewrite_model", "pioneer/auto")

    if not rewrite_key:
        print("ERROR: no pioneer_rewrite_key in settings.json")
        sys.exit(1)

    print(f"Rewrite key: {rewrite_key[:20]}...{rewrite_key[-8:]}")
    print(f"Model:       {model}")
    print(f"URL:         {settings.get('pioneer_api_url')}")
    print()

    r1 = test_ping_curl(rewrite_key, model)
    r2 = test_ping_python(rewrite_key, model)
    r3 = test_medium_payload(rewrite_key, model) if r2 else False
    r4 = test_full_rewrite(rewrite_key, model) if r3 else False

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  1. Small curl PING:       {'PASS' if r1 else 'FAIL'}")
    print(f"  2. Small requests PING:   {'PASS' if r2 else 'FAIL'}")
    print(f"  3. Medium 3k payload:     {'PASS' if r3 else 'FAIL' if r2 else 'SKIP'}")
    print(f"  4. FULL 17k rewrite:      {'PASS' if r4 else 'FAIL' if r3 else 'SKIP'}")
    print()

    if r4:
        print("Rewrite pipeline is HEALTHY. Проблема HTTP 0 виправлена.")
    elif r3 and not r4:
        print("Small requests OK, but large payload FAILS. Це саме та проблема з HTTP 0.")
        print("Скоріш за все треба:")
        print("  - Або зменшити payload (chunks)")
        print("  - Або перейти на прямий Claude API")
    elif r2 and not r3:
        print("Small requests OK, but medium fails. Pioneer може блокувати навіть 3k payload.")
    else:
        print("Ключ або модель не працюють. Перевір pioneer_rewrite_model.")


if __name__ == "__main__":
    main()
