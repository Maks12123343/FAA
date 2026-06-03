import paramiko

host = '206.189.103.5'
user = 'root'
password = 'FAA_Deploy_2026!'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

def exec_cmd(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    return out + err

# Upload start.sh only (no restart - file 4 is processing)
sftp = client.open_sftp()
sftp.put('start.sh', '/opt/faa/start.sh')
sftp.close()
print("=== start.sh uploaded ===")

# Check current status
print("=== current status ===")
print(exec_cmd("journalctl -u faa -n 5 --no-pager 2>&1", timeout=10))

print("=== ffmpeg running? ===")
print(exec_cmd("ps aux | grep ffmpeg | grep -v grep", timeout=10))

client.close()