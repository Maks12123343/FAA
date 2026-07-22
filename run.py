import sys
sys.path.insert(0, '/workspace/FAA')
import os
os.chdir('/workspace/FAA')
os.environ['FAA_DEV'] = '1'
os.environ.setdefault('FAA_CORS_ORIGIN', '*')
import eventlet
eventlet.monkey_patch()
from app import app, socketio
port = int(os.environ.get('FAA_PORT', '5050'))
host = os.environ.get('FAA_HOST', '127.0.0.1')
print(f'Starting on {host}:{port}...')
socketio.run(app, host=host, port=port)
