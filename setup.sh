#!/bin/bash
# Run once on the server as root: bash setup.sh
set -e

APP_DIR=/opt/faa
GDRIVE_MOUNT=/mnt/gdrive

# 1. Install system packages
apt-get update -qq
apt-get install -y -qq ffmpeg curl fuse3
echo "[setup] system packages: OK"

# 2. Install rclone (latest)
if ! command -v rclone &>/dev/null; then
    curl -s https://rclone.org/install.sh | bash
    echo "[setup] rclone installed"
else
    echo "[setup] rclone already installed"
fi

# 3. Create system user (no login, no home)
if ! id faa &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin faa
    echo "[setup] Created user: faa"
fi

# 4. Create required directories
mkdir -p "$APP_DIR/data/library" "$APP_DIR/data/niches" "$APP_DIR/projects"
mkdir -p "$GDRIVE_MOUNT"
echo "[setup] directories: OK"

# 5. Set ownership and permissions
chown -R faa:faa "$APP_DIR"
chmod +x "$APP_DIR/start.sh"
echo "[setup] permissions: OK"

# 6. Install nginx config (HTTP only — certbot adds HTTPS later)
cp "$APP_DIR/nginx.conf" /etc/nginx/sites-available/faa
ln -sf /etc/nginx/sites-available/faa /etc/nginx/sites-enabled/faa
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
echo "[setup] nginx: OK"

# 7. Install systemd units (NOT started yet — deps must be installed first)
cp "$APP_DIR/faa.service" /etc/systemd/system/faa.service
cp "$APP_DIR/faa-cleanup.service" /etc/systemd/system/faa-cleanup.service
cp "$APP_DIR/faa-cleanup.timer" /etc/systemd/system/faa-cleanup.timer
cp "$APP_DIR/faa-gdrive.service" /etc/systemd/system/faa-gdrive.service
systemctl daemon-reload
systemctl enable faa
systemctl enable faa-cleanup.timer
systemctl enable faa-gdrive
echo "[setup] systemd units: enabled (not started)"

echo ""
echo "=== Next steps (in order) ==="
echo "  1. Configure rclone Google Drive:"
echo "       mkdir -p /opt/faa/.config/rclone"
echo "       RCLONE_CONFIG=/opt/faa/.config/rclone/rclone.conf rclone config"
echo "       (create remote named 'gdrive', type: drive, follow OAuth)"
echo "       chown -R faa:faa /opt/faa/.config"
echo "  2. Test rclone mount:"
echo "       rclone --config /opt/faa/.config/rclone/rclone.conf ls gdrive:FAA/stocks"
echo "  3. Create .env:"
echo "       cp $APP_DIR/.env.example $APP_DIR/.env && nano $APP_DIR/.env"
echo "       (set FAA_USER, FAA_PASS, FAA_CORS_ORIGIN)"
echo "  4. Put Vertex AI credentials:"
echo "       mkdir -p /opt/faa/.config/gcloud"
echo "       # upload your service account JSON to /opt/faa/.config/gcloud/application_default_credentials.json"
echo "       chown faa:faa /opt/faa/.config/gcloud/application_default_credentials.json"
echo "  5. Install Python deps:"
echo "       cd $APP_DIR && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "  6. Edit nginx.conf — replace YOUR_DOMAIN_OR_IP with real domain:"
echo "       nano $APP_DIR/nginx.conf && nginx -t && systemctl reload nginx"
echo "  7. Start Google Drive mount:"
echo "       systemctl start faa-gdrive"
echo "  8. Start app:"
echo "       systemctl start faa && systemctl start faa-cleanup.timer"
echo "  9. Get HTTPS cert (replace YOUR_DOMAIN):"
echo "       certbot --nginx -d YOUR_DOMAIN"
echo " 10. Watch logs:"
echo "       journalctl -u faa -f"
