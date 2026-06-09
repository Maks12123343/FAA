import sys
sys.path.insert(0, '/workspace/FAA')
import os
os.chdir('/workspace/FAA')
os.environ['FAA_DEV'] = '1'
os.environ.setdefault('FAA_CORS_ORIGIN', '*')
import eventlet
eventlet.monkey_patch()
from app import app, socketio
socketio.server.eio.allow_upgrades = False
print('Starting on port 16006 (polling mode)...')
socketio.run(app, host='0.0.0.0', port=16006)
