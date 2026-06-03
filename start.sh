#!/bin/bash
set -e

cd "$(dirname "$0")"

# Load .env file (strips surrounding quotes from values)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Setup virtual environment if not exists
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install/update requirements
"$VENV_DIR/bin/pip" install -q -r requirements.txt

# Generate persistent secret key on first run
SECRET_FILE=".secret_key"
if [ ! -f "$SECRET_FILE" ]; then
    python3 -c "import secrets; print(secrets.token_hex(32))" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
fi
export FAA_SECRET_KEY="${FAA_SECRET_KEY:-$(cat "$SECRET_FILE")}"

export FAA_HOST="${FAA_HOST:-127.0.0.1}"
export FAA_PORT="${FAA_PORT:-5050}"

echo "Starting FAA on $FAA_HOST:$FAA_PORT"
exec "$VENV_DIR/bin/gunicorn" --worker-class eventlet -w 1 \
    --bind "$FAA_HOST:$FAA_PORT" \
    --timeout 0 \
    --keep-alive 5 \
    app:app
