import paramiko
import json

host = '206.189.103.5'
user = 'root'
password = 'FAA_Deploy_2026!'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

def run(cmd, timeout=30):
    s, o, e = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='ignore') + e.read().decode('utf-8', errors='ignore')

print("=== processed_sources in index.json ===")
script = (
    "import json\n"
    "d = json.load(open('/mnt/gdrive/FAA/movies/Kung Fu Panda/index.json'))\n"
    "ps = d.get('processed_sources', [])\n"
    "print('processed_sources:', ps)\n"
    "print('total clips in index:', len(d.get('clips', [])))\n"
)
print(run(f"python3 -c \"{script}\""))

print("\n=== source files in folder ===")
script2 = (
    "import os\n"
    "files = [f for f in os.listdir('/mnt/gdrive/FAA/movies/Kung Fu Panda') if f.endswith('.mp4')]\n"
    "for f in sorted(files): print(f)\n"
)
print(run(f"python3 -c \"{script2}\""))

print("\n=== Pioneer.ai keys in settings ===")
script3 = (
    "import json\n"
    "s = json.load(open('/opt/faa/data/settings.json'))\n"
    "keys = [k for k in s if 'pioneer' in k.lower() or 'piapi' in k.lower()]\n"
    "for k in keys: print(k, '=', str(s[k])[:60])\n"
)
print(run(f"python3 -c \"{script3}\""))

print("\n=== clips without analysis.json ===")
script4 = (
    "import os\n"
    "d = '/mnt/gdrive/FAA/movies/Kung Fu Panda/clips'\n"
    "mp4s = set(f for f in os.listdir(d) if f.endswith('.mp4'))\n"
    "analyzed = set(f.replace('.analysis.json','') for f in os.listdir(d) if f.endswith('.analysis.json'))\n"
    "missing = [f for f in mp4s if f not in analyzed]\n"
    "print('clips without analysis:', len(missing))\n"
    "print('first 5:', missing[:5])\n"
)
print(run(f"python3 -c \"{script4}\""))

client.close()