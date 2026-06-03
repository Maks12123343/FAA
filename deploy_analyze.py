import paramiko, time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('206.189.103.5', username='root', password='FAA_Deploy_2026!')

def run(cmd, timeout=60):
    s, o, e = client.exec_command(cmd)
    o.channel.settimeout(timeout)
    out = o.read().decode('utf-8', errors='ignore')
    err = e.read().decode('utf-8', errors='ignore')
    return (out + err).strip()

# Kill old process
print("Killing old process...")
run("pkill -f analyze_f03.py; pkill -f analyze_all.py; sleep 2")
print("OK")

# Remount GDrive if needed
print("Checking GDrive mount...")
mount_ok = run("mountpoint -q /mnt/gdrive && echo OK || echo NOTMOUNTED")
if "NOTMOUNTED" in mount_ok:
    print("Remounting GDrive...")
    run("systemctl restart faa-gdrive.service; sleep 5")
    mount_ok = run("mountpoint -q /mnt/gdrive && echo OK || echo NOTMOUNTED")
print("Mount status:", run("mountpoint /mnt/gdrive"))

# Test rclone
test = run('rclone lsf "gdrive:FAA/movies/Kung Fu Panda/clips" --config /opt/faa/.config/rclone/rclone.conf --include "*.mp4" --max-depth 1 2>/dev/null | head -3')
print("Test rclone:", test[:100])

# Upload script
print("Uploading analyze_all.py...")
sftp = client.open_sftp()
sftp.put('analyze_all.py', '/opt/faa/analyze_all.py')
sftp.close()
print("Uploaded")

# Start
print("Starting...")
run("nohup python3 /opt/faa/analyze_all.py > /opt/faa/analyze_all.log 2>&1 &")
time.sleep(3)
pid = run("pgrep -f analyze_all.py")
print(f"PID: {pid}")

# Show first log lines
time.sleep(5)
log = run("head -10 /opt/faa/analyze_all.log")
print("Log:\n" + log)

client.close()