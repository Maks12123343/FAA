"""Local OpenAI-compatible image bridge backed by Google Flow in Chrome."""

from __future__ import annotations

import base64
import hmac
import os
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, request

try:
    from .flow_client import FLOW_MODELS, FlowCliClient
except ImportError:
    from flow_client import FLOW_MODELS, FlowCliClient


ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4981"))
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "").strip()
FLOW_MODEL = os.environ.get("FLOW_MODEL", "flow-nano-pro").strip() or "flow-nano-pro"
FLOW_PROFILE = os.environ.get("FLOW_PROFILE", "faa").strip() or "faa"
FLOW_HOME = os.path.abspath(os.path.expandvars(os.environ.get(
    "FLOW_HOME",
    os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "FAA", "flow_browser"),
)))
REQUEST_TIMEOUT = max(600, min(1800, int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))))
FLOW_RETRIES = max(1, min(5, int(os.environ.get("FLOW_RETRIES", "3"))))
MAX_IMAGE_BYTES = max(1_000_000, int(os.environ.get("MAX_IMAGE_BYTES", str(50 * 1024 * 1024))))


app = Flask(__name__)
client = FlowCliClient(
    profile=FLOW_PROFILE,
    home=FLOW_HOME,
    default_model=FLOW_MODEL,
    timeout=REQUEST_TIMEOUT,
    retries=FLOW_RETRIES,
    max_image_bytes=MAX_IMAGE_BYTES,
)


def _authorized() -> bool:
    if not LOCAL_API_KEY:
        return False
    supplied = request.headers.get("X-API-Key", "")
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    return hmac.compare_digest(supplied, LOCAL_API_KEY)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "backend": "google-flow-chrome",
        "installed": client.installed,
        "configured": client.configured,
        "profile": FLOW_PROFILE,
        "profile_dir": str(client.profile_dir),
        "model": FLOW_MODEL,
        "aspect": "16:9",
        "last_success_at": int(client.last_success_at) if client.last_success_at else None,
        "last_error": client.last_error,
        "bind": HOST,
        "port": PORT,
    })


@app.get("/auth/status")
def auth_status():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    try:
        ok, detail = client.auth_status()
        return jsonify({"ok": ok, "detail": detail}), 200 if ok else 503
    except Exception as exc:
        return jsonify({"ok": False, "detail": str(exc)}), 503


@app.get("/v1/models")
def models():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    ids = sorted({key for key in FLOW_MODELS if key.startswith("flow-")})
    return jsonify({
        "data": [
            {"id": model_id, "object": "model", "owned_by": "google-flow"}
            for model_id in ids
        ]
    })


@app.post("/v1/images/generations")
def generate_image():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "prompt is required"}), 400
    try:
        # Check the saved browser session before spending a Flow generation.
        session_ok, session_detail = client.auth_status()
        if not session_ok:
            return jsonify({
                "error": {
                    "message": "Google Flow session is signed out. Run gemini_bridge/setup_browser_profile.ps1 to sign in again.",
                    "type": "google_flow_auth_required",
                    "detail": session_detail,
                }
            }), 401
        image, mime = client.generate(prompt, str(body.get("model", "")))
        return jsonify({
            "created": int(time.time()),
            "data": [{
                "b64_json": base64.b64encode(image).decode("ascii"),
                "revised_prompt": prompt.strip(),
                "mime_type": mime,
            }],
        })
    except Exception as exc:
        print(f"[flow-bridge] generation failed: {type(exc).__name__}: {exc}", flush=True)
        return jsonify({"error": {"message": str(exc), "type": "google_flow_error"}}), 502


if __name__ == "__main__":
    if not LOCAL_API_KEY:
        raise SystemExit("LOCAL_API_KEY is required in gemini_bridge/.env")
    print(f"[flow-bridge] listening on http://{HOST}:{PORT}", flush=True)
    print(f"[flow-bridge] profile: {client.profile_dir}", flush=True)
    print(f"[flow-bridge] model={FLOW_MODEL} aspect=16:9", flush=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
