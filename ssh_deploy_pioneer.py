import paramiko
import json
import time

host = '206.189.103.5'
user = 'root'
password = 'FAA_Deploy_2026!'

PIONEER_KEYS = [
    "pio_sk_6dd4ba50-4f7e-4c1a-9072-1f134f8ef190_squ1d_DropltO2aMNiQEc1",
    "pio_sk_6dd4ba50-4f7e-4c1a-9072-1f134f8ef190_cr4b_NjEzI94RX7Itu9QX",
    "pio_sk_6dd4ba50-4f7e-4c1a-9072-1f134f8ef190_r4v3n_W_R_UXeYrBa_PyPR",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

def run(cmd, timeout=60):
    s, o, e = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='ignore') + e.read().decode('utf-8', errors='ignore')

# 1. git pull
print(">>> git pull")
print(run("cd /opt/faa && git pull origin master"))

# 2. Update settings.json with pioneer key
print(">>> updating settings.json")
settings_raw = run("cat /opt/faa/data/settings.json")
try:
    settings = json.loads(settings_raw)
except Exception:
    settings = {}

settings["pioneer_api_keys"] = PIONEER_KEYS
settings["pioneer_model"] = "a87f8985-e7d8-4012-adac-6d5c66287213"
settings["pioneer_api_url"] = "https://api.pioneer.ai/v1/chat/completions"

new_json = json.dumps(settings, indent=2, ensure_ascii=False)

# Write via heredoc
escaped = new_json.replace("'", "'\\''")
print(run(f"echo '{escaped}' > /opt/faa/data/settings.json"))
print("Settings written.")

# Verify
verify = run("python3 -c \"import json; s=json.load(open('/opt/faa/data/settings.json')); print('pioneer_api_keys:', s.get('pioneer_api_keys'))\"")
print("Verify:", verify)

# 3. Restart
print(">>> restart faa")
print(run("systemctl restart faa"))
time.sleep(4)
print(run("systemctl status faa --no-pager -n 3"))

client.close()
print("Done")