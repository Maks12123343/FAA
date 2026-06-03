"""
Аналіз кліпів f03 Kung Fu Panda через Pioneer.ai.
Зберігає аналіз ЛОКАЛЬНО (/opt/faa/f03_cache/), копіює в GDrive в кінці.
Використовує rclone lsf для listing (обходить FUSE зависання).
"""
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

CLIPS_DIR    = "/mnt/gdrive/FAA/movies/Kung Fu Panda/clips"
RCLONE_CLIPS  = "gdrive:FAA/movies/Kung Fu Panda/clips"
RCLONE_CONFIG = "/opt/faa/.config/rclone/rclone.conf"
INDEX_PATH   = "/mnt/gdrive/FAA/movies/Kung Fu Panda/index.json"
LOCAL_CACHE  = "/opt/faa/f03_cache"
MOVIE_NAME   = "Kung Fu Panda"
F03_PREFIX   = "kung_fu_panda_f03_"
F03_SOURCE   = "252643--ca78ff29-b147-4075-a9b1-3d5fc7efc404--usew--2215528-streamwish.mp4"

FFMPEG = "ffmpeg"
BATCH_SIZE = 4
NUM_WORKERS = 3

BATCH_PROMPT = """\
Analyze {n} video clips from the movie "{movie_name}".
For each clip, 1 frame is shown, labeled CLIP 1, CLIP 2, etc.

For EACH clip return a JSON object with:
- characters: list of character names visible (e.g. "Tigress", "Po", "Shifu")
- emotion: one of: joy, sadness, fear, anger, determination, vulnerability, shame, guilt, pride, neutral
- scene_type: one of: training, fight, emotional_dialogue, rejection, acceptance, flashback, celebration, isolation, comedy, action, quiet_moment, transformation
- themes: 2-4 from: [growth, trauma, impostor_syndrome, false_self, identity, rejection, acceptance, vulnerability, shame, fear, determination, healing, connection, isolation, anger, betrayal, grief, love, trust]
- description: 1-2 sentence visual description
- tags: 8-12 topic tags
- is_blurry: true if frame is out of focus or heavily motion-blurred
- is_static: false (default)

Reply ONLY with a JSON array of exactly {n} objects, no markdown, no comments."""


def _load_settings():
    return json.load(open("/opt/faa/data/settings.json"))


def _rclone_list(rclone_path, pattern, timeout=120):
    """Use rclone lsf to list files — bypasses FUSE, uses GDrive API directly."""
    result = subprocess.run(
        ["rclone", "lsf", rclone_path,
         "--config", RCLONE_CONFIG,
         "--include", pattern,
         "--max-depth", "1"],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone lsf failed: {result.stderr[:200]}")
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    return files


def _frame_b64(clip_path, worker_name):
    """Extract 1st frame (no seek — avoids GDrive FUSE slow seek)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", clip_path,
             "-vframes", "1", "-vf", "scale=320:-2", "-q:v", "6", tmp.name],
            capture_output=True, timeout=90,
        )
        if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
            with open(tmp.name, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"  [{worker_name}] ffmpeg err: {e}", flush=True)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return ""


def _save_local(clip_name, analysis):
    """Save to local cache (fast, no GDrive hang)."""
    path = os.path.join(LOCAL_CACHE, clip_name + ".analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)


def _parse_json_robust(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    idx = text.find("[")
    if idx >= 0:
        candidate = text[idx:]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        last_brace = candidate.rfind("},")
        if last_brace > 0:
            try:
                return json.loads(candidate[:last_brace+1] + "]")
            except Exception:
                pass

    objects = []
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    objects.append(json.loads(text[start:i+1]))
                except Exception:
                    pass
                start = -1
    if objects:
        return objects

    raise ValueError(f"Cannot parse JSON: {text[:200]}")


def _parse_raw(raw, batch_clips):
    results = []
    for i, clip_path in enumerate(batch_clips):
        a = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        a.setdefault("characters", [])
        a.setdefault("emotion", "neutral")
        a.setdefault("scene_type", "quiet_moment")
        a.setdefault("themes", [])
        a.setdefault("description", "")
        a.setdefault("tags", [])
        a.setdefault("is_blurry", False)
        a.setdefault("is_static", False)
        clip_name = os.path.basename(clip_path).replace(".mp4", "")
        a["id"]   = clip_name
        a["file"] = clip_path
        results.append((clip_name, a))
    return results


def _analyze_pioneer(batch_clips, api_key, api_url, model, worker_name, batch_idx):
    import urllib.request

    content = []
    for i, clip_path in enumerate(batch_clips):
        b64 = _frame_b64(clip_path, worker_name)
        content.append({"type": "text", "text": f"CLIP {i+1}:"})
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content.append({"type": "text",
                    "text": BATCH_PROMPT.format(n=len(batch_clips), movie_name=MOVIE_NAME)})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2500,
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        api_url, data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return _parse_json_robust(data["choices"][0]["message"]["content"])


def main():
    os.makedirs(LOCAL_CACHE, exist_ok=True)

    settings = _load_settings()
    pioneer_keys  = settings.get("pioneer_api_keys", [])
    pioneer_model = settings.get("pioneer_model", "a87f8985-e7d8-4012-adac-6d5c66287213")
    pioneer_url   = settings.get("pioneer_api_url", "https://api.pioneer.ai/v1/chat/completions")

    print(f"Pioneer keys: {len(pioneer_keys)}, model: {pioneer_model}", flush=True)

    # List f03 mp4 clips via rclone (bypasses FUSE)
    print("Listing f03 clips via rclone...", flush=True)
    all_mp4 = sorted(_rclone_list(RCLONE_CLIPS, f"{F03_PREFIX}*.mp4"))
    print(f"Total f03 clips: {len(all_mp4)}", flush=True)

    # List existing analysis files via rclone (bypasses FUSE)
    print("Listing existing analysis files via rclone...", flush=True)
    gdrive_analysis = _rclone_list(RCLONE_CLIPS, f"{F03_PREFIX}*.analysis.json")
    gdrive_done = set()
    for fn in gdrive_analysis:
        # fn = "kung_fu_panda_f03_XXXX.mp4.analysis.json"
        gdrive_done.add(fn.replace(".mp4.analysis.json", ""))
    print(f"Already on GDrive: {len(gdrive_done)}", flush=True)

    # Local cache
    local_done = set()
    if os.path.exists(LOCAL_CACHE):
        for fn in os.listdir(LOCAL_CACHE):
            if fn.endswith(".analysis.json"):
                try:
                    a = json.load(open(os.path.join(LOCAL_CACHE, fn)))
                    if a.get("description") not in ("unknown", "", None):
                        local_done.add(fn.replace(".analysis.json", ""))
                except Exception:
                    pass
    print(f"Already in local cache: {len(local_done)}", flush=True)

    done_set = local_done | gdrive_done
    missing = [
        os.path.join(CLIPS_DIR, f) for f in all_mp4
        if f.replace(".mp4", "") not in done_set
    ]
    print(f"Need analysis: {len(missing)}", flush=True)

    if not missing:
        print("All f03 clips already analyzed!", flush=True)
        _copy_to_gdrive_and_update_index()
        return

    batches = [missing[i:i+BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    print(f"Batches: {len(batches)} (batch_size={BATCH_SIZE}, workers={NUM_WORKERS})", flush=True)

    batch_q = queue.Queue()
    for i, b in enumerate(batches):
        batch_q.put((i, b))

    done_count = [0]
    total = len(missing)
    lock = threading.Lock()
    start_time = [time.time()]

    def process_batch(batch_idx, batch_clips, worker_name, analyze_fn):
        for attempt in range(3):
            try:
                raw = analyze_fn(batch_clips)
                results = _parse_raw(raw, batch_clips)
                saved = 0
                for clip_name, a in results:
                    if a.get("description") not in ("", None):
                        _save_local(clip_name, a)
                        saved += 1
                with lock:
                    done_count[0] += saved
                    elapsed = time.time() - start_time[0]
                    rate = done_count[0] / elapsed * 60 if elapsed > 0 else 0
                    eta = (total - done_count[0]) / (rate / 60) / 60 if rate > 0 else 0
                    print(f"[{worker_name}] batch {batch_idx}: {done_count[0]}/{total} | {rate:.1f}/min | ETA {eta:.0f}min", flush=True)
                return
            except Exception as e:
                print(f"[{worker_name}] batch {batch_idx} attempt {attempt+1} err: {str(e)[:100]}", flush=True)
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        print(f"[{worker_name}] batch {batch_idx}: SKIPPED", flush=True)

    def worker_pioneer(worker_id, api_key):
        name = f"P{worker_id}"
        while True:
            try:
                idx, batch = batch_q.get_nowait()
            except queue.Empty:
                break
            analyze_fn = lambda b, i=idx: _analyze_pioneer(b, api_key, pioneer_url, pioneer_model, name, i)
            process_batch(idx, batch, name, analyze_fn)
            batch_q.task_done()
        print(f"[{name}] done", flush=True)

    threads = []
    for worker_id, key in enumerate(pioneer_keys[:NUM_WORKERS]):
        t = threading.Thread(target=worker_pioneer, args=(worker_id, key), daemon=True)
        threads.append(t)
        t.start()
        print(f"Started P{worker_id}", flush=True)

    for t in threads:
        t.join()

    print(f"\nAnalysis done! {done_count[0]}/{total} clips saved locally", flush=True)
    _copy_to_gdrive_and_update_index()


def _copy_to_gdrive_and_update_index():
    print("\nCopying analysis files to GDrive via rclone...", flush=True)

    if not os.path.exists(LOCAL_CACHE):
        print("No local cache found.", flush=True)
        return

    # Use rclone copy to upload all analysis files at once
    result = subprocess.run(
        ["rclone", "copy", LOCAL_CACHE, RCLONE_CLIPS,
         "--config", RCLONE_CONFIG,
         "--include", "*.analysis.json",
         "--transfers", "8"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        count = len([f for f in os.listdir(LOCAL_CACHE) if f.endswith(".analysis.json")])
        print(f"Copied {count} analysis files to GDrive", flush=True)
    else:
        print(f"rclone copy failed: {result.stderr[:300]}", flush=True)

    print("Updating index.json...", flush=True)
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            index = json.load(f)
    except Exception as e:
        print(f"Cannot read index: {e}", flush=True)
        return

    existing_ids = {c["id"] for c in index.get("clips", [])}
    new_clips = []

    for fn in sorted(os.listdir(LOCAL_CACHE)):
        if not fn.endswith(".analysis.json"):
            continue
        clip_name = fn.replace(".analysis.json", "")
        if clip_name in existing_ids:
            continue
        try:
            with open(os.path.join(LOCAL_CACHE, fn), encoding="utf-8") as f:
                a = json.load(f)
        except Exception:
            continue
        if a.get("description") in ("unknown", "", None):
            continue
        if a.get("is_blurry") or a.get("is_static"):
            continue
        a["id"]   = clip_name
        a["file"] = os.path.join(CLIPS_DIR, clip_name + ".mp4")
        new_clips.append(a)

    print(f"New good clips to add: {len(new_clips)}", flush=True)
    index["clips"].extend(new_clips)

    ps = set(index.get("processed_sources", []))
    ps.add(F03_SOURCE)
    index["processed_sources"] = sorted(ps)
    index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Index updated: {len(index['clips'])} total clips", flush=True)
    print(f"processed_sources: {index['processed_sources']}", flush=True)


if __name__ == "__main__":
    main()