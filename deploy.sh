#!/bin/bash
# ============================================================
# FAA — Full Auto Deploy for Vast.ai
# One command in Jupyter terminal and everything works.
# Usage: curl -sL https://raw.githubusercontent.com/Maks12123343/FAA/master/deploy.sh | bash
#   OR:  paste this entire script into the Jupyter terminal
# ============================================================
set -e

FAA_DIR="/workspace/FAA"
FAA_PORT=16006
NGROK_TOKEN="2oFXcF3ErF74nLdfF0vUk9ObCBq_4K2Pj98oZzptmf8NchEPC"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     FAA — Automated Server Deploy        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────
# 1. Clone repo
# ──────────────────────────────────────────────
echo "=== [1/8] Cloning FAA repository ==="
if [ -d "$FAA_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$FAA_DIR" && git pull origin master
else
    rm -rf "$FAA_DIR"
    git clone https://github.com/Maks12123343/FAA.git "$FAA_DIR"
fi
cd "$FAA_DIR"

# ──────────────────────────────────────────────
# 2. System packages
# ──────────────────────────────────────────────
echo "=== [2/8] System packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg python3-pip git curl unzip 2>&1 | tail -3

# ──────────────────────────────────────────────
# 3. Python packages
# ──────────────────────────────────────────────
echo "=== [3/8] Python packages ==="
pip install "setuptools<70" --force-reinstall --break-system-packages -q 2>&1 | tail -2
pip install -r requirements.txt --break-system-packages -q 2>&1 | tail -5
pip install -U yt-dlp --break-system-packages -q 2>&1 | tail -2

# ──────────────────────────────────────────────
# 4. Verify GPU
# ──────────────────────────────────────────────
echo "=== [4/8] GPU Check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" 2>/dev/null || echo "WARNING: torch/CUDA check failed"

if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    echo "NVENC: available"
else
    echo "NVENC: NOT available (fallback to libx264)"
fi

# ──────────────────────────────────────────────
# 5. Settings.json (API keys)
# ──────────────────────────────────────────────
echo "=== [5/8] Writing settings.json ==="
mkdir -p "$FAA_DIR/data"
if [ ! -f "$FAA_DIR/data/settings.json" ]; then
python3 << 'PYEOF'
import json, os
settings = {
  "stocks_dir": "/workspace/FAA/stocks",
  "vertex_project_id": "project-b1dbf3fd-b163-4fab-9e6",
  "vertex_location": "us-central1",
  "gemini_model": "gemini-2.5-flash",
  "claude_api_key": os.environ.get("FAA_CLAUDE_KEY", ""),
  "claude_model": "claude-sonnet-4-6",
  "pioneer_api_keys": os.environ.get("FAA_PIONEER_KEYS", "").split(",") if os.environ.get("FAA_PIONEER_KEYS") else [],
  "pioneer_model": "gemini-3.5-flash",
  "pioneer_api_url": "https://api.pioneer.ai/v1/chat/completions",
  "pioneer_timeout": 90,
  "pioneer_retries": 1,
  "tts_api_key": os.environ.get("FAA_TTS_KEY", ""),
  "tts_api_url": "https://voiceapi.csv666.ru",
  "youtube_api_key": os.environ.get("FAA_YT_KEY", ""),
  "youtube_api_key_2": os.environ.get("FAA_YT_KEY_2", ""),
  "youtube_api_key_3": "",
  "voice_profiles": {
    "en": {"name": "English Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "pl": {"name": "Polish Voice", "voice_id": "1nUkvoDFCcCTjJk9U8mL", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "de": {"name": "German Voice", "voice_id": "Cqbq4nsuUe1we6J45miU", "stability": 0.8, "similarity_boost": 0.75, "speed": 1},
    "fr": {"name": "French Voice", "voice_id": "DGTOOUoGpoP6UZ9uSWfA", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "es": {"name": "Spanish Voice", "voice_id": "GFA5VGODW0iSJ4ob3Vn7", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "it": {"name": "Italian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "pt": {"name": "Portuguese Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "uk": {"name": "Ukrainian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "ru": {"name": "Russian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1},
    "tr": {"name": "Turkish Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1}
  },
  "clip_score_threshold": 0.85,
  "clip_frames_positions": [0.01, 0.1, 0.5, 0.9],
  "clip_min_duration": 2,
  "clip_max_duration": 5,
  "stock_max_duration": 6,
  "competitor_ratio": 0.6,
  "output_width": 1920,
  "output_height": 1080,
  "fps": 30,
  "pexels_api_key": os.environ.get("FAA_PEXELS_KEY", ""),
  "pexels_api_keys": os.environ.get("FAA_PEXELS_KEYS", "").split(",") if os.environ.get("FAA_PEXELS_KEYS") else [],
  "pixabay_api_key": "",
  "envato_api_key": ""
}
with open("/workspace/FAA/data/settings.json", "w") as f:
    json.dump(settings, f, indent=2)
print("settings.json written (keys from env vars)")
PYEOF
else
    echo "settings.json already exists, skipping"
fi

# ──────────────────────────────────────────────
# 6. Vertex AI credentials
# ──────────────────────────────────────────────
echo "=== [6/8] Vertex AI credentials ==="
CREDS_DIR="$FAA_DIR/.config/gcloud"
mkdir -p "$CREDS_DIR"
if [ ! -f "$CREDS_DIR/application_default_credentials.json" ]; then
python3 << 'PYEOF'
import json, os
creds = {
  "account": "",
  "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
  "client_secret": os.environ.get("FAA_GCLOUD_SECRET", ""),
  "refresh_token": os.environ.get("FAA_GCLOUD_REFRESH", ""),
  "type": "authorized_user",
  "universe_domain": "googleapis.com"
}
out_path = "/workspace/FAA/.config/gcloud/application_default_credentials.json"
with open(out_path, "w") as f:
    json.dump(creds, f, indent=2)
print(f"Credentials written to {out_path}")
PYEOF
else
    echo "Credentials already exist, skipping"
fi

# ──────────────────────────────────────────────
# 7. Create directories
# ──────────────────────────────────────────────
echo "=== [7/8] Creating directories ==="
mkdir -p "$FAA_DIR/stocks"
mkdir -p "$FAA_DIR/projects"
mkdir -p "$FAA_DIR/data/library"
mkdir -p "$FAA_DIR/data/niches"
echo "Done"

# ──────────────────────────────────────────────
# 8. Install & start ngrok
# ──────────────────────────────────────────────
echo "=== [8/8] Ngrok setup ==="
if ! command -v ngrok &>/dev/null; then
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin
fi
ngrok config add-authtoken "$NGROK_TOKEN" 2>/dev/null || true

# ──────────────────────────────────────────────
# LAUNCH
# ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           SETUP COMPLETE!                ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
echo "║  Start FAA:                              ║"
echo "║  cd /workspace/FAA                       ║"
echo "║  FAA_DEV=1 python3 app.py &              ║"
echo "║                                          ║"
echo "║  Start ngrok (new terminal):             ║"
echo "║  ngrok http $FAA_PORT                    ║"
echo "║                                          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "NOTE: Set API keys via environment variables BEFORE running this script:"
echo "  export FAA_CLAUDE_KEY='sk-ant-...'"
echo "  export FAA_PIONEER_KEYS='key1,key2,key3'"
echo "  export FAA_TTS_KEY='...'"
echo "  export FAA_YT_KEY='...'"
echo "  export FAA_YT_KEY_2='...'"
echo "  export FAA_PEXELS_KEY='...'"
echo "  export FAA_PEXELS_KEYS='key1,key2,key3'"
echo "  export FAA_GCLOUD_SECRET='...'"
echo "  export FAA_GCLOUD_REFRESH='...'"
echo ""
echo "Or edit /workspace/FAA/data/settings.json manually after deploy."
