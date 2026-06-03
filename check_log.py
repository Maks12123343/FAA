import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('206.189.103.5', username='root', password='FAA_Deploy_2026!')

def run(cmd, timeout=30):
    s, o, e = client.exec_command(cmd)
    o.channel.settimeout(timeout)
    try:
        return o.read().decode('utf-8', errors='ignore').strip()
    except Exception:
        return ""

pid = run("pgrep -f analyze_all.py")
print(f"PID: {pid or 'DONE'}")

cache = run("ls /opt/faa/kfp_cache/*.analysis.json 2>/dev/null | wc -l")
print(f"Cache files: {cache}")

log = run("tail -20 /opt/faa/analyze_all.log")
print(f"\nLog tail:\n{log}")

client.close()