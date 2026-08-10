"""FAA client for the local Gemini Web image bridge."""

import base64
import binascii
from io import BytesIO
import os
import tempfile

import requests
from PIL import Image

import config


MAX_IMAGE_BYTES = 50 * 1024 * 1024


def _emit(emit, message: str):
    if emit:
        emit("thumbnail_image", message)


def _settings():
    settings = config.load_settings()
    return {
        "enabled": bool(settings.get("gemini_image_enabled", False)),
        "url": str(settings.get("gemini_image_bridge_url", "http://127.0.0.1:4981")).rstrip("/"),
        "api_key": str(settings.get("gemini_image_api_key", "")).strip(),
        "model": str(settings.get("gemini_image_model", "gemini-3-pro-image")).strip(),
        "timeout": max(30, min(900, int(settings.get("gemini_image_timeout", 360) or 360))),
    }


def _decode_image(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Gemini bridge returned an empty image")
    raw = value.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeError("Gemini bridge returned invalid base64 image data") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError("Gemini bridge returned an image with an invalid size")
    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff") or data.startswith(b"RIFF")):
        raise RuntimeError("Gemini bridge returned a non-image payload")
    return data


def _normalize_png(data: bytes) -> bytes:
    """Store every provider image as a real PNG, regardless of returned MIME."""
    try:
        with Image.open(BytesIO(data)) as image:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA")
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except Exception as exc:
        raise RuntimeError("Gemini bridge returned an unreadable image") from exc


def generate_thumbnail(prompt: str, output_path: str, emit=None) -> bool:
    """Generate and atomically save a thumbnail. Returns False when disabled."""
    cfg = _settings()
    if not cfg["enabled"]:
        return False
    if not cfg["api_key"]:
        raise RuntimeError("Gemini image bridge is enabled but its local API key is empty")
    if not prompt.strip():
        raise RuntimeError("Cannot generate a thumbnail from an empty prompt")

    endpoint = cfg["url"] + "/v1/images/generations"
    payload = {
        "model": cfg["model"],
        "prompt": prompt.strip(),
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json",
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "FAA/1.0",
    }
    _emit(emit, "Generating thumbnail through Gemini Web...")
    response = requests.post(endpoint, json=payload, headers=headers, timeout=cfg["timeout"])
    if response.status_code >= 400:
        detail = (response.text or "")[:500]
        raise RuntimeError(f"Gemini image bridge HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
        encoded = body["data"][0]["b64_json"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini image bridge returned an unexpected response") from exc

    data = _normalize_png(_decode_image(encoded))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="thumbnail_generated.", suffix=".part", dir=os.path.dirname(output_path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    _emit(emit, f"Gemini thumbnail saved: {output_path}")
    return True
