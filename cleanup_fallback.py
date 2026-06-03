import os, json

d = '/mnt/gdrive/FAA/movies/Kung Fu Panda/clips'
removed = 0
for fn in os.listdir(d):
    if not fn.endswith('.analysis.json'):
        continue
    fp = os.path.join(d, fn)
    try:
        a = json.load(open(fp))
        if a.get('description') == 'unknown':
            os.remove(fp)
            removed += 1
    except Exception:
        pass

# Count remaining
total = sum(1 for f in os.listdir(d) if 'f03' in f and f.endswith('.analysis.json'))
print(f'Removed {removed} fallback files')
print(f'Remaining f03 analysis files: {total}')