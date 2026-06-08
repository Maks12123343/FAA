import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

POLL_INTERVAL = 5    # seconds between status checks
MAX_WAIT      = 1800 # max seconds to wait for task (30 min — long scripts need time)


def _get_voice_profile(language: str) -> dict:
    settings = config.load_settings()
    profiles = settings.get("voice_profiles", {})
    profile = profiles.get(language)
    if not profile:
        raise ValueError(f"No voice profile configured for language: {language}")
    if not profile.get("voice_id"):
        raise ValueError(f"voice_id not set for language: {language}")
    return profile


def generate(text: str, language: str, output_path: str) -> str:
    """
    Generate TTS audio for text in the given language.
    Saves MP3 to output_path and returns the path.
    """
    settings  = config.load_settings()
    api_key   = settings.get("tts_api_key", "")
    base_url  = settings.get("tts_api_url", "https://voiceapi.csv666.ru").rstrip("/")
    profile   = _get_voice_profile(language)

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    payload = {
        "text": text,
        "template": {
            "model_id": "eleven_multilingual_v2",
            "voice_id": profile["voice_id"],
            "stability": profile.get("stability", 0.85),
            "similarity_boost": profile.get("similarity_boost", 0.75),
            "speed": profile.get("speed", 1.0),
            "style": 0.0,
            "use_speaker_boost": True,
        }
    }

    print(f"[tts] Creating task for language={language}, chars={len(text)}", flush=True)
    r = requests.post(f"{base_url}/tasks", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    task_id = r.json()["task_id"]
    print(f"[tts] Task created: {task_id}", flush=True)

    # Poll until done
    _DONE_STATUSES = {"ending", "completed", "done", "finished", "success"}
    _FAIL_STATUSES = {"error", "failed", "cancelled"}
    waited = 0
    status = ""
    while waited < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL

        sr = requests.get(f"{base_url}/tasks/{task_id}/status", headers=headers, timeout=15)
        sr.raise_for_status()
        status = sr.json().get("status", "")
        print(f"[tts] Status: {status}", flush=True)

        if status in _DONE_STATUSES:
            break
        if status in _FAIL_STATUSES:
            raise RuntimeError(f"TTS task {task_id} failed (status: {status})")

    if status not in _DONE_STATUSES:
        raise RuntimeError(f"TTS task {task_id} timed out after {MAX_WAIT}s (last status: {status})")

    # Download result
    dr = requests.get(f"{base_url}/tasks/{task_id}/result", headers=headers, timeout=60, stream=True)
    dr.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in dr.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[tts] Saved to {output_path}", flush=True)
    return output_path


def get_balance() -> dict:
    settings = config.load_settings()
    api_key  = settings.get("tts_api_key", "")
    base_url = settings.get("tts_api_url", "https://voiceapi.csv666.ru").rstrip("/")
    headers  = {"X-API-Key": api_key}
    r = requests.get(f"{base_url}/balance", headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def list_templates() -> list:
    settings = config.load_settings()
    api_key  = settings.get("tts_api_key", "")
    base_url = settings.get("tts_api_url", "https://voiceapi.csv666.ru").rstrip("/")
    headers  = {"X-API-Key": api_key}
    r = requests.get(f"{base_url}/templates", headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()
