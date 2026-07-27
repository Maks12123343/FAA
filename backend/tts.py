import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

VOICEGEN_DEFAULT_URL = "https://qw1voicegencore.pro"
LEGACY_TTS_URLS = {"https://voiceapi.csv666.ru"}

POLL_INTERVAL = 5
MAX_WAIT = 1800


def _get_voice_profile(language: str) -> dict:
    settings = config.load_settings()
    profiles = settings.get("voice_profiles", {})
    profile = profiles.get(language)
    if not profile and language == "sv":
        legacy = profiles.get("sw")
        if legacy and "swedish" in legacy.get("name", "").lower():
            profile = legacy
    if not profile:
        raise ValueError(f"No voice profile configured for language: {language}")
    if not profile.get("voice_id"):
        raise ValueError(f"voice_id not set for language: {language}")
    return profile


def _voicegen_settings() -> tuple[str, str, dict]:
    settings = config.load_settings()
    api_key = (
        os.environ.get("VOICEGEN_API_KEY")
        or os.environ.get("TTS_API_KEY")
        or settings.get("tts_api_key", "")
    ).strip()
    if not api_key:
        raise RuntimeError("No VoiceGen API key configured. Set VOICEGEN_API_KEY or tts_api_key in Settings.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("VoiceGen API key must be ASCII, for example vg_live_...") from exc
    if not api_key.startswith("vg_"):
        raise RuntimeError("VoiceGen API key must start with 'vg_'.")

    base_url = (
        os.environ.get("VOICEGEN_API_URL")
        or os.environ.get("TTS_API_URL")
        or settings.get("tts_api_url")
        or VOICEGEN_DEFAULT_URL
    ).rstrip("/")
    if base_url in LEGACY_TTS_URLS:
        base_url = VOICEGEN_DEFAULT_URL

    headers = {"Authorization": f"Bearer {api_key}"}
    return base_url, api_key, headers


def _response_error(resp: requests.Response) -> str:
    body = ""
    try:
        body = (resp.text or "")[:500]
    except Exception:
        pass
    return f"HTTP {resp.status_code}: {body}"


def _get_with_retries(url: str, headers: dict, timeout: int, attempts: int = 4, **kwargs) -> requests.Response:
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return requests.get(url, headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_err = exc
            if attempt < attempts:
                wait = min(20, attempt * 5)
                print(f"[tts] VoiceGen GET failed attempt {attempt}/{attempts}: {exc}; retry in {wait}s", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"VoiceGen GET failed after {attempts} attempts: {last_err}")


def _voicegen_payload(text: str, language: str, output_path: str) -> dict:
    settings = config.load_settings()
    profile = _get_voice_profile(language)
    voice_engine = profile.get("voice_engine") or settings.get("tts_voice_engine") or "elevenLabsV3"
    payload = {
        "text": text,
        "filename": os.path.basename(output_path) or "voiceover.mp3",
        "voice_id": profile["voice_id"],
        "voice_engine": voice_engine,
        "settings_preset": profile.get("settings_preset") or settings.get("tts_settings_preset") or "standard",
        "chunk_size": int(profile.get("chunk_size") or settings.get("tts_chunk_size") or 1200),
        "delay_between_chunks": float(
            profile.get("delay_between_chunks") or settings.get("tts_delay_between_chunks") or 0.5
        ),
        "thread_count": int(profile.get("thread_count") or settings.get("tts_thread_count") or 10),
        "auto_start": True,
    }
    return payload


def generate(text: str, language: str, output_path: str) -> str:
    """
    Generate TTS audio through VoiceGen and save MP3 to output_path.
    """
    base_url, _api_key, headers = _voicegen_settings()
    payload = _voicegen_payload(text, language, output_path)

    print(f"[tts] VoiceGen: creating task language={language}, chars={len(text)}", flush=True)
    r = requests.post(
        f"{base_url}/api/v1/client/tasks",
        json=payload,
        headers={**headers, "Content-Type": "application/json"},
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"VoiceGen task create failed: {_response_error(r)}")
    data = r.json()
    task = data.get("task") or data
    task_id = task.get("task_id") or task.get("id")
    if not task_id:
        raise RuntimeError(f"VoiceGen task create response has no task_id: {data}")
    print(f"[tts] VoiceGen task created: {task_id}", flush=True)

    _DONE_STATUSES = {"done", "completed", "finished", "success"}
    _FAIL_STATUSES = {"error", "failed", "cancelled", "canceled"}
    waited = 0
    status = ""
    progress = None
    while waited < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL

        sr = _get_with_retries(f"{base_url}/api/v1/client/tasks/{task_id}", headers=headers, timeout=30)
        if not sr.ok:
            raise RuntimeError(f"VoiceGen task status failed: {_response_error(sr)}")
        state = sr.json()
        task = state.get("task") or state
        status = (task.get("status") or "").lower()
        progress = task.get("progress")
        if progress is None:
            print(f"[tts] VoiceGen status: {status}", flush=True)
        else:
            print(f"[tts] VoiceGen status: {status} ({progress}%)", flush=True)

        if status in _DONE_STATUSES:
            break
        if status in _FAIL_STATUSES:
            detail = task.get("error") or task.get("message") or state.get("message") or ""
            raise RuntimeError(f"VoiceGen task {task_id} failed (status: {status}) {detail}".strip())

    if status not in _DONE_STATUSES:
        raise RuntimeError(f"VoiceGen task {task_id} timed out after {MAX_WAIT}s (last status: {status})")

    dr = _get_with_retries(
        f"{base_url}/api/v1/client/tasks/{task_id}/download",
        headers=headers,
        timeout=300,
        stream=True,
        allow_redirects=True,
    )
    if not dr.ok:
        raise RuntimeError(f"VoiceGen download failed: {_response_error(dr)}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    part_path = output_path + ".part"
    written = 0
    try:
        with open(part_path, "wb") as f:
            for chunk in dr.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        if written < 1024:
            raise RuntimeError(f"VoiceGen result for task {task_id} is too small ({written} bytes)")
        os.replace(part_path, output_path)
    except BaseException:
        try:
            os.unlink(part_path)
        except OSError:
            pass
        raise

    print(f"[tts] Saved to {output_path} ({written} bytes)", flush=True)
    return output_path


def get_balance() -> dict:
    base_url, _api_key, headers = _voicegen_settings()
    r = _get_with_retries(f"{base_url}/api/v1/client/me", headers=headers, timeout=10)
    if not r.ok:
        raise RuntimeError(f"VoiceGen profile failed: {_response_error(r)}")
    return r.json()


def list_templates() -> list:
    base_url, _api_key, headers = _voicegen_settings()
    r = _get_with_retries(f"{base_url}/api/v1/client/templates", headers=headers, timeout=10)
    if not r.ok:
        raise RuntimeError(f"VoiceGen templates failed: {_response_error(r)}")
    data = r.json()
    if isinstance(data, list):
        return data
    return data.get("templates") or data.get("items") or []
