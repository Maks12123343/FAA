#!/bin/bash
# RunPod fresh setup — run as root in /workspace
# Usage: bash runpod_setup.sh
set -e

GITHUB_TOKEN="${1:-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}"
FAA_DIR="/workspace/FAA"

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg python3-pip git curl unzip 2>&1 | tail -3

echo "=== [2/6] Clone FAA ==="
if [ ! -d "$FAA_DIR" ]; then
    git clone "https://Maks12123343:${GITHUB_TOKEN}@github.com/Maks12123343/FAA.git" "$FAA_DIR"
    echo "Cloned FAA"
else
    cd "$FAA_DIR" && git pull
    echo "FAA already exists, pulled latest"
fi

echo "=== [3/6] Python packages ==="
cd "$FAA_DIR"
pip install -r requirements.txt --break-system-packages --no-build-isolation -q 2>&1 | tail -5
pip install openai-whisper==20231117 --no-build-isolation --break-system-packages -q 2>&1 | tail -3

echo "=== [4/6] Node.js 20 ==="
if ! node --version 2>/dev/null | grep -q "v20"; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
    nvm install 20
    ln -sf "$NVM_DIR/versions/node/v20.20.2/bin/node" /usr/local/bin/node 2>/dev/null || true
    ln -sf "$NVM_DIR/versions/node/v20.20.2/bin/node" /usr/bin/node 2>/dev/null || true
    echo "Node $(node --version) installed"
else
    echo "Node already OK: $(node --version)"
fi

echo "=== [5/6] rclone ==="
if ! command -v rclone &>/dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
echo "rclone $(rclone --version | head -1)"

echo ""
echo "========================================="
echo "Setup complete! Next steps:"
echo ""
echo "1. Configure rclone (if not already):"
echo "   rclone config"
echo "   (create remote named 'gdrive', type: drive)"
echo ""
echo "2. Copy stocks from Google Drive:"
echo "   rclone copy gdrive:FAA/stocks /workspace/FAA/stocks --progress"
echo ""
echo "3. Upload cookies.txt from your PC:"
echo "   scp -P PORT -i KEY cookies.txt root@IP:/workspace/FAA/cookies.txt"
echo ""
echo "4. Start FAA:"
echo "   cd /workspace/FAA"
echo "   export FAA_DEV=1"
echo "   export FAA_CORS_ORIGIN=https://PODID-5050.proxy.runpod.net"
echo "   python3 -c \"import app; app.socketio.run(app.app, host='0.0.0.0', port=5050, debug=False)\""
echo "========================================="
