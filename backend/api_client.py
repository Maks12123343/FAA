"""Shared Pioneer.ai API client with key rotation, retry, and timeout handling."""
import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def call_pioneer(system: str, messages: list, timeout: int = 180, max_retries: int = 3,
                 emit=None, step_label: str = "api", use_rewrite_model: bool = True) -> tuple:
    """Call Pioneer.ai with automatic key rotation and retry.

    use_rewrite_model=True (default): uses pioneer_rewrite_key + pioneer_rewrite_model (Opus)
                                      with fallback to regular pioneer_api_keys.
    use_rewrite_model=False: uses regular pioneer_api_keys + pioneer_model.

    Returns: (text, stop_reason)
    Raises: RuntimeError if all keys fail.
    """
    import requests as _req
    settings = config.load_settings()

    # Build key list: rewrite key first (if available and requested), then regular keys
    rewrite_key = settings.get("pioneer_rewrite_key", "").strip()
    regular_keys = settings.get("pioneer_api_keys", [])
    if isinstance(regular_keys, str):
        regular_keys = [k.strip() for k in regular_keys.split(",") if k.strip()]

    if use_rewrite_model and rewrite_key:
        api_keys = [rewrite_key] + regular_keys
        model = settings.get("pioneer_rewrite_model", "claude-opus-4-8")
    else:
        api_keys = regular_keys
        model = settings.get("pioneer_model", "gemini-3.5-flash")

    if not api_keys:
        raise RuntimeError("No pioneer_api_keys configured in Settings.")

    api_url = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")

    payload = {
        "model":    model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream":   False,
    }

    last_err = None
    for key_idx, api_key in enumerate(api_keys):
        for attempt in range(max_retries):
            if emit and (key_idx > 0 or attempt > 0):
                emit(step_label, f"Pioneer API call — key {key_idx+1}/{len(api_keys)}, attempt {attempt+1}/{max_retries}")

            try:
                resp = _req.post(
                    api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                finish = data["choices"][0]["finish_reason"]
                stop_reason = "max_tokens" if finish == "length" else finish
                return text, stop_reason

            except _req.exceptions.Timeout:
                last_err = f"Timeout ({timeout}s) on key {key_idx+1}, attempt {attempt+1}"
                print(f"[api_client] {last_err}", flush=True)
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[api_client] Retry in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    break  # Try next key

            except _req.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                # ✅ Don't include response body — may contain sensitive data
                last_err = f"HTTP {status} on key {key_idx+1}"
                print(f"[api_client] {last_err}", flush=True)
                # Don't retry 4xx errors (bad request, auth)
                if status in (400, 401, 403, 404):
                    break  # Try next key
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"[api_client] Retry in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    break  # Try next key

            except Exception as e:
                # ✅ Don't include exception message — may contain API keys or sensitive data
                last_err = f"{type(e).__name__} on key {key_idx+1}, attempt {attempt+1}"
                print(f"[api_client] {last_err}", flush=True)
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[api_client] Retry in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    break  # Try next key

    raise RuntimeError(f"All Pioneer API keys failed. Last error: {last_err}")


def call_gigacoder_opus(system: str, messages: list, timeout: int = 180, max_retries: int = 2,
                        emit=None, step_label: str = "api") -> tuple:
    """Call GigaCoder Opus 4.8 for rewrite. Returns (text, stop_reason). Raises on failure."""
    import requests as _req
    settings = config.load_settings()

    api_key = settings.get("gigacoder_rewrite_key", "").strip()
    if not api_key:
        raise RuntimeError("No gigacoder_rewrite_key configured")

    api_url = settings.get("gigacoder_api_url", "https://www.gigacoder.org/api/v1/chat/completions")
    model = settings.get("gigacoder_rewrite_model", "claude-opus-4-8")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _req.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            finish = data["choices"][0]["finish_reason"]
            stop_reason = "max_tokens" if finish == "length" else finish
            return text, stop_reason
        except Exception as e:
            last_err = f"{type(e).__name__} attempt {attempt+1}"
            print(f"[api_client] GigaCoder Opus: {last_err}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5)

    raise RuntimeError(f"GigaCoder Opus failed: {last_err}")


def call_gigacoder(system: str, messages: list, timeout: int = 180, max_retries: int = 2,
                   emit=None, step_label: str = "api") -> tuple:
    """Call GigaCoder GPT-5.4-mini. Returns (text, stop_reason). Raises on failure."""
    import requests as _req
    settings = config.load_settings()

    api_keys = settings.get("gigacoder_api_keys", [])
    if isinstance(api_keys, str):
        api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
    if not api_keys:
        raise RuntimeError("No gigacoder_api_keys configured")

    api_url = settings.get("gigacoder_api_url", "https://www.gigacoder.org/api/v1/chat/completions")
    model = settings.get("gigacoder_model", "gpt-5.4-mini")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }

    last_err = None
    for api_key in api_keys:
        for attempt in range(max_retries):
            try:
                resp = _req.post(
                    api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                finish = data["choices"][0]["finish_reason"]
                stop_reason = "max_tokens" if finish == "length" else finish
                return text, stop_reason
            except Exception as e:
                last_err = f"{type(e).__name__} key {api_keys.index(api_key)+1} attempt {attempt+1}"
                print(f"[api_client] GigaCoder: {last_err}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(3)

    raise RuntimeError(f"GigaCoder GPT failed: {last_err}")
