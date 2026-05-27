import sys
import os
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _get_keys() -> list:
    s = config.load_settings()
    keys = [
        s.get("youtube_api_key",   ""),
        s.get("youtube_api_key_2", ""),
        s.get("youtube_api_key_3", ""),
    ]
    return [k for k in keys if k.strip()]


def _is_quota_error(response: requests.Response) -> bool:
    if response.status_code != 403:
        return False
    try:
        errors = response.json().get("error", {}).get("errors", [])
        return any(e.get("reason") in ("quotaExceeded", "dailyLimitExceeded") for e in errors)
    except Exception:
        return False


def yt_request(endpoint: str, params: dict, _rate_retries: int = 4) -> dict:
    """
    Make a YouTube Data API v3 request with automatic key rotation and rate-limit retry.
    - 403 quotaExceeded → rotate to next key
    - 429 rate limit    → wait and retry same key (up to _rate_retries times)
    """
    import time

    keys = _get_keys()
    if not keys:
        raise RuntimeError(
            "No YouTube API key configured. Add at least one key in Settings."
        )

    for i, key in enumerate(keys):
        p = {**params, "key": key}

        for attempt in range(_rate_retries):
            r = requests.get(
                f"https://www.googleapis.com/youtube/v3/{endpoint}",
                params=p,
                timeout=20,
            )

            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"[youtube_api] Rate limit (429), waiting {wait}s (attempt {attempt+1}/{_rate_retries})...", flush=True)
                time.sleep(wait)
                continue

            if _is_quota_error(r):
                print(f"[youtube_api] Key {i+1} daily quota exceeded, trying next...", flush=True)
                break

            if r.status_code != 200:
                raise RuntimeError(
                    f"YouTube API '{endpoint}' error {r.status_code}: {r.text[:200]}"
                )

            return r.json()

        else:
            # exhausted rate-limit retries on this key, try next
            print(f"[youtube_api] Key {i+1} still rate-limited after {_rate_retries} retries, trying next...", flush=True)
            continue

    raise RuntimeError(
        "All YouTube API keys have exceeded their quota or rate limit. Try again later."
    )
