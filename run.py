import os
import socket
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)
os.environ['FAA_DEV'] = '1'
os.environ.setdefault('FAA_CORS_ORIGIN', '*')
import eventlet
eventlet.monkey_patch()
from app import app, socketio


def _bridge_is_running(host: str = "127.0.0.1", port: int = 4981) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _start_gemini_bridge_if_configured() -> None:
    """Start the local Gemini bridge once when its private .env is present."""
    try:
        settings = __import__("config").load_settings()
        if not bool(settings.get("gemini_image_enabled", False)):
            return
        bridge_dir = APP_DIR / "gemini_bridge"
        env_file = bridge_dir / ".env"
        server_file = bridge_dir / "server.py"
        if not env_file.exists() or not server_file.exists():
            print("Gemini bridge not started: gemini_bridge/.env is missing.", flush=True)
            return
        if _bridge_is_running():
            print("Gemini bridge already running on 127.0.0.1:4981.", flush=True)
            return
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        subprocess.Popen(
            [sys.executable, str(server_file)],
            cwd=str(bridge_dir),
            creationflags=flags,
        )
        time.sleep(1)
        print("Gemini bridge started on 127.0.0.1:4981.", flush=True)
    except Exception as exc:
        # Image generation is optional; never prevent the FAA site from starting.
        print(f"Gemini bridge auto-start skipped: {exc}", flush=True)


_start_gemini_bridge_if_configured()
port = int(os.environ.get('FAA_PORT', '5050'))
host = os.environ.get('FAA_HOST', '127.0.0.1')
print(f'Starting on {host}:{port}...')
socketio.run(app, host=host, port=port)
