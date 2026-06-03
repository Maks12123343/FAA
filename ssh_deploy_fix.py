import paramiko
import time

host = '206.189.103.5'
user = 'root'
password = 'FAA_Deploy_2026!'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

def exec_cmd(cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    return out + err

print('>>> git stash')
print(exec_cmd('cd /opt/faa && git stash'))

print('>>> git pull')
print(exec_cmd('cd /opt/faa && git pull origin master'))

print('>>> restart faa')
print(exec_cmd('systemctl restart faa'))

time.sleep(3)

print('>>> verify -c copy in new code')
print(exec_cmd('grep -n "c copy" /opt/faa/backend/movie_library.py | head -5'))

print('>>> status')
print(exec_cmd('systemctl status faa --no-pager -n 3'))

client.close()
print('Done')