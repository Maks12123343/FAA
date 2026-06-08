#!/bin/bash
# Vast.ai GPU setup — run as root after connecting
# Usage: bash vast_setup.sh
# Expects: NVIDIA GPU with CUDA (Vast.ai default images have this)
set -e

FAA_DIR="/workspace/FAA"

echo "=== [1/7] System packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg python3-pip git curl unzip fuse3 2>&1 | tail -3

echo "=== [2/7] Check GPU ==="
if nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "WARNING: No GPU detected! Whisper will run on CPU (slow)."
fi

echo "=== [3/7] Python packages ==="
cd "$FAA_DIR"
pip install -r requirements.txt --break-system-packages --no-build-isolation -q 2>&1 | tail -5
# Whisper with GPU support (torch with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages -q 2>&1 | tail -3
pip install openai-whisper==20231117 --no-build-isolation --break-system-packages -q 2>&1 | tail -3

echo "=== [4/7] Verify Whisper GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo "=== [5/7] Verify FFmpeg NVENC ==="
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    echo "h264_nvenc: available (GPU video encoding)"
else
    echo "h264_nvenc: NOT available (will use libx264 CPU)"
fi

echo "=== [6/7] Node.js 20 (for yt-dlp) ==="
if ! node --version 2>/dev/null | grep -q "v2"; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1 | tail -2
    apt-get install -y nodejs 2>&1 | tail -2
fi
echo "Node: $(node --version 2>/dev/null || echo 'not installed')"

echo "=== [7/7] rclone (for Google Drive mount) ==="
if ! command -v rclone &>/dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
echo "rclone: $(rclone --version 2>/dev/null | head -1)"

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy rclone config (from your PC):"
echo "   mkdir -p ~/.config/rclone"
echo "   # paste your rclone.conf content"
echo ""
echo "2. Mount Google Drive (stocks + movies):"
echo "   mkdir -p /mnt/gdrive"
echo "   rclone mount gdrive:FAA /mnt/gdrive --daemon --allow-other --vfs-cache-mode full --vfs-cache-max-size 10G"
echo ""
echo "   OR copy movies locally (faster):"
echo "   rclone copy gdrive:FAA/movies /workspace/FAA/movies --progress"
echo "   rclone copy gdrive:FAA/stocks /workspace/FAA/stocks --progress"
echo ""
echo "3. Put settings.json with API keys:"
echo "   nano $FAA_DIR/data/settings.json"
echo ""
echo "4. Put Vertex AI credentials:"
echo "   mkdir -p ~/.config/gcloud"
echo "   nano ~/.config/gcloud/application_default_credentials.json"
echo ""
echo "5. Start FAA:"
echo "   cd $FAA_DIR"
echo "   export FAA_DEV=1"
echo "   python3 app.py"
echo "   # Access via Vast.ai port forwarding on port 5050"
echo ""
echo "========================================="
