import paramiko, time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('206.189.103.5', username='root', password='FAA_Deploy_2026!')

# Upload a helper script
script = r"""
import subprocess
from collections import Counter

def rclone_list(pattern):
    r = subprocess.run(
        ["rclone", "lsf", "gdrive:FAA/movies/Kung Fu Panda/clips",
         "--config", "/opt/faa/.config/rclone/rclone.conf",
         "--include", pattern, "--max-depth", "1"],
        capture_output=True, text=True, timeout=120
    )
    return set(f.strip() for f in r.stdout.splitlines() if f.strip())

print("Listing mp4s...")
mp4s = rclone_list("*.mp4")
print(f"Total mp4: {len(mp4s)}")

print("Listing analyzed...")
analyzed = rclone_list("*.analysis.json")
analyzed_bases = set(f.replace(".analysis.json", "") for f in analyzed)
print(f"Analyzed: {len(analyzed_bases)}")

missing = sorted(f for f in mp4s if f.replace(".mp4","") not in analyzed_bases)
print(f"Missing: {len(missing)}")

prefixes = Counter()
for f in missing:
    parts = f.split("_")
    prefix = "_".join(parts[:4]) if len(parts) >= 4 else f[:20]
    prefixes[prefix] += 1
for p, c in prefixes.most_common(10):
    print(f"  {p}: {c}")

print("First 5:", missing[:5])
"""

sftp = client.open_sftp()
with sftp.open('/tmp/check_missing.py', 'w') as f:
    f.write(script)
sftp.close()

s, o, e = client.exec_command('python3 /tmp/check_missing.py')
o.channel.settimeout(180)
out = o.read().decode('utf-8', errors='ignore')
err = e.read().decode('utf-8', errors='ignore')
print(out)
if err:
    print("ERR:", err[:300])

client.close()