import eventlet
eventlet.monkey_patch()

import json
import os
import re
import shutil
import threading
import time

from flask import Flask, render_template, request, jsonify, send_file, Response, session
from flask_socketio import SocketIO, emit

import config
from backend import pipeline, media_library, clip_sourcer
from backend import stocks_library, competitor_finder
from backend import movie_library, movie_pipeline
from backend import writer as writer_backend

app = Flask(__name__)
app.secret_key = os.environ.get("FAA_SECRET_KEY") or os.urandom(32).hex()

_cors_env = os.environ.get("FAA_CORS_ORIGIN", "")
if _cors_env.strip() == "*":
    _cors_origins = "*"
else:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else ["http://localhost:5050", "http://127.0.0.1:5050"]
socketio = SocketIO(app, cors_allowed_origins=_cors_origins, async_mode="eventlet")

# ── Basic auth ────────────────────────────────────────────────────────────────
_AUTH_USER = os.environ.get("FAA_USER", "")
_AUTH_PASS = os.environ.get("FAA_PASS", "")
_DEV_MODE  = os.environ.get("FAA_DEV", "") == "1"

if not _DEV_MODE and (not _AUTH_USER or not _AUTH_PASS):
    import sys
    print("[app] FATAL: FAA_USER and FAA_PASS must be set. Set FAA_DEV=1 to run without auth (local dev only).", flush=True)
    sys.exit(1)

@app.before_request
def _require_auth():
    if not _AUTH_USER or not _AUTH_PASS:
        return
    if request.path.startswith("/static"):
        return
    # Only trust remote_addr (set by the OS/nginx), never trust X-Real-IP from client
    if request.remote_addr in ("127.0.0.1", "::1"):
        return
    auth = request.authorization
    if auth and auth.username == _AUTH_USER and auth.password == _AUTH_PASS:
        return
    return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="FAA"'})

# ── Job concurrency limit (max 1 active generation) ──────────────────────────
_job_lock = threading.Lock()
_job_active = False
_job_last_msg = ""
_job_started_at = 0.0

_ID_RE = re.compile(r'^[a-z0-9_\-]{1,64}$')

def _safe_id(value: str, field: str = "id") -> str:
    v = (value or "").strip().lower()
    if not _ID_RE.match(v):
        raise ValueError(f"Invalid {field}: use only lowercase letters, digits, _ and -")
    return v


def _niches() -> list:
    niches = []
    for f in os.listdir(config.NICHES_DIR):
        if f.endswith(".json"):
            path = os.path.join(config.NICHES_DIR, f)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            name = f[:-5]
            niches.append({
                "id":           name,
                "name":         data.get("name", name),
                "library_mode": data.get("library_mode", "standard"),
                "pipeline_type": data.get("pipeline_type", "standard"),
            })
    return niches


def _languages() -> list:
    settings = config.load_settings()
    profiles = settings.get("voice_profiles", {})
    return [
        {"code": code, "name": p["name"]}
        for code, p in profiles.items()
        if p.get("voice_id")
    ]


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           niches=_niches(),
                           languages=_languages())


@app.route("/settings")
def settings_page():
    return render_template("settings.html", settings=config.load_settings())


@app.route("/competitors")
def competitors_page():
    return render_template("competitors.html", niches=_niches())


@app.route("/library")
def library_page():
    niches = _niches()
    stats  = {}
    for n in niches:
        stats[n["id"]] = media_library.get_library_stats(n["id"])
    return render_template("library.html", niches=niches, stats=stats)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["POST"])
def save_settings():
    data = request.json or {}
    if not data:
        return jsonify({"error": "No data provided"}), 400
    settings = config.load_settings()
    # Deep merge: for dict-valued keys (like voice_profiles), merge instead of overwrite
    for key, value in data.items():
        if key not in config.DEFAULT_SETTINGS:
            continue
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    config.save_settings(settings)
    return jsonify({"ok": True})


@app.route("/api/prepare", methods=["POST"])
def api_prepare():
    global _job_active
    data  = request.json or {}
    try:
        niche = _safe_id(data.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    source_url = data.get("source_url") or None

    # Detect niche pipeline type
    niche_path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    pipeline_type = "standard"
    movie_names = []
    if os.path.exists(niche_path):
        with open(niche_path, encoding="utf-8") as f:
            niche_data = json.load(f)
        pipeline_type = niche_data.get("pipeline_type", "standard")
        movie_names = niche_data.get("movie_library", [])

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active, _job_last_msg, _job_started_at

        _job_started_at = time.time()
        _job_last_msg = "Starting..."

        def _emit(step, msg):
            global _job_last_msg
            _job_last_msg = f"[{step}] {msg}"
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)

        try:
            if pipeline_type == "movie" or movie_names:
                # Movie pipeline: just transcribe source URL
                if not source_url:
                    raise ValueError("Movie pipeline requires a source YouTube URL. Please paste a URL in Step 1.")
                result = movie_pipeline.prepare(source_url, emit=_emit)
                # Store movie name in state for later produce
                prepare_dir = result.get("prepare_dir", os.path.join(config.PROJECTS_DIR, f"_prepare_{result['prepare_id']}"))
                state_path = os.path.join(prepare_dir, "state.json")
                if os.path.exists(state_path):
                    with open(state_path, encoding="utf-8") as f:
                        state = json.load(f)
                    state["_niche_pipeline_type"] = "movie"
                    state["_movie_names"] = movie_names
                    state["_niche"] = niche
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                socketio.emit("prepare_done", {
                    **result,
                    "pipeline_type": "movie",
                    "movie_names": movie_names,
                    "niche": niche,
                })
            else:
                # Standard pipeline: find top video + transcribe
                result = pipeline.prepare(
                    niche,
                    source_url=source_url,
                    emit=_emit,
                )
                socketio.emit("prepare_done", {**result, "pipeline_type": "standard"})
        except Exception as e:
            import traceback
            print(f"[app] ERROR in prepare: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/produce", methods=["POST"])
def api_produce():
    global _job_active
    data        = request.json or {}
    prepare_id  = data.get("prepare_id", "").strip()
    youtube_urls = data.get("youtube_urls", [])
    languages   = data.get("languages", [])
    movie_name  = data.get("movie_name", "").strip()
    main_character = data.get("main_character", "").strip()
    test_mode   = bool(data.get("test_mode", False))

    if not prepare_id or not languages:
        return jsonify({"error": "prepare_id and languages required"}), 400
    if os.path.sep in prepare_id or ".." in prepare_id:
        return jsonify({"error": "Invalid prepare_id"}), 400

    # Detect pipeline type from prepare state
    prepare_dir = os.path.join(config.PROJECTS_DIR, f"_prepare_{prepare_id}")
    state_path = os.path.join(prepare_dir, "state.json")
    pipeline_type = "standard"
    movie_names = []
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        pipeline_type = state.get("_niche_pipeline_type", "standard")
        movie_names = state.get("_movie_names", [])

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active, _job_last_msg, _job_started_at

        _job_started_at = time.time()
        _job_last_msg = "Starting production..."

        def _emit(step, msg):
            global _job_last_msg
            _job_last_msg = f"[{step}] {msg}"
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)

        try:
            if pipeline_type == "movie" or movie_names:
                # Movie pipeline
                _movie = movie_name if movie_name else (movie_names[0] if movie_names else "")
                if not _movie:
                    _emit("error", "No movie selected for movie pipeline. Index a movie first.")
                    socketio.emit("error", {"message": "No movie selected for movie pipeline. Index a movie first."})
                    return
                for lang in languages:
                    try:
                        socketio.emit("progress", {"step": "produce", "message": f"Starting language: {lang}"})
                        eventlet.sleep(0)
                        result = movie_pipeline.produce(
                            prepare_id = prepare_id,
                            movie_name = _movie,
                            language   = lang,
                            emit=_emit,
                            test_mode=test_mode,
                            main_character=main_character,
                        )
                        socketio.emit("produce_done", result)
                    except Exception as e:
                        import traceback
                        print(f"[app] ERROR in movie produce [{lang}]: {e}\n{traceback.format_exc()}", flush=True)
                        socketio.emit("error", {"message": f"[{lang}] {e}"})
            else:
                # Standard pipeline
                for lang in languages:
                    try:
                        socketio.emit("progress", {"step": "produce", "message": f"Starting language: {lang}"})
                        eventlet.sleep(0)
                        result = pipeline.produce(
                            prepare_id   = prepare_id,
                            youtube_urls = youtube_urls,
                            language     = lang,
                            emit=_emit,
                        )
                        socketio.emit("produce_done", result)
                    except Exception as e:
                        import traceback
                        print(f"[app] ERROR in produce [{lang}]: {e}\n{traceback.format_exc()}", flush=True)
                        socketio.emit("error", {"message": f"[{lang}] {e}"})
            socketio.emit("all_done", {})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/generate", methods=["POST"])
def generate():
    return jsonify({"error": "Deprecated. Use POST /api/prepare then POST /api/produce."}), 410


@app.route("/api/library/add", methods=["POST"])
def library_add():
    global _job_active
    data      = request.json or {}
    video_url = data.get("url")
    try:
        niche = _safe_id(data.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not video_url:
        return jsonify({"error": "url required"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        try:
            clips = media_library.download_from_channel(video_url, niche)
            socketio.emit("library_progress", {"message": f"Downloaded {len(clips)} clips"})
            niche_path = os.path.join(config.NICHES_DIR, f"{niche}.json")
            with open(niche_path, encoding="utf-8") as f:
                niche_data = json.load(f)
            description = niche_data.get("description", niche)
            result = media_library.validate_library(niche, description)
            socketio.emit("library_done", result)
        except Exception as e:
            socketio.emit("library_progress", {"message": f"ERROR: {e}"})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/library/stats")
def library_stats():
    try:
        niche = _safe_id(request.args.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(media_library.get_library_stats(niche))


@app.route("/api/download/<project_id>")
def download_video(project_id):
    # Sanitize: strip any directory traversal attempts
    safe_id = os.path.basename(project_id)
    # Standard pipeline output
    output = os.path.join(config.PROJECTS_DIR, safe_id, "output.mp4")
    # Movie pipeline output (named after project_id)
    output2 = os.path.join(config.PROJECTS_DIR, safe_id, f"{safe_id}.mp4")
    # Verify the resolved path is still inside PROJECTS_DIR
    if os.path.exists(output):
        if not os.path.realpath(output).startswith(os.path.realpath(config.PROJECTS_DIR)):
            return jsonify({"error": "Invalid project id"}), 400
        return send_file(output, as_attachment=True, download_name=f"{safe_id}.mp4")
    if os.path.exists(output2):
        if not os.path.realpath(output2).startswith(os.path.realpath(config.PROJECTS_DIR)):
            return jsonify({"error": "Invalid project id"}), 400
        return send_file(output2, as_attachment=True, download_name=f"{safe_id}.mp4")
    return jsonify({"error": "Not found"}), 404


@app.route("/api/library/fetch_stocks", methods=["POST"])
def fetch_stocks():
    global _job_active
    data  = request.json or {}
    extra = data.get("extra_keywords", "")
    try:
        niche = _safe_id(data.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    niche_path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    with open(niche_path, encoding="utf-8") as f:
        niche_data = json.load(f)

    keywords = list(niche_data.get("stock_tags", []))
    if extra:
        keywords += [k.strip() for k in extra.split(",") if k.strip()]
    description = niche_data.get("description", niche)

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        try:
            result = clip_sourcer.fetch_and_validate(niche, keywords, description)
            socketio.emit("library_progress", {"message": f"Fetched {result.get('fetched',0)} raw clips"})
            socketio.emit("library_done", result)
        except Exception as e:
            socketio.emit("library_progress", {"message": f"ERROR: {e}"})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


# ── CLIP-classification library mode (e.g. russia_ukraine_war) ──────────────
@app.route("/api/library/classify", methods=["POST"])
def library_classify():
    global _job_active
    data = request.json or {}
    try:
        niche = _safe_id(data.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    niche_path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    if not os.path.exists(niche_path):
        return jsonify({"error": f"Niche not found: {niche}"}), 404
    with open(niche_path, encoding="utf-8") as f:
        niche_cfg = json.load(f)

    if niche_cfg.get("library_mode") != "clip_classification":
        return jsonify({"error": f"Niche '{niche}' is not in clip_classification mode"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def _emit(step, msg):
        socketio.emit("library_progress", {"message": msg})
        eventlet.sleep(0)

    def run():
        global _job_active
        try:
            from backend import clip_classifier
            result = clip_classifier.process_library(niche, niche_cfg, emit=_emit)
            socketio.emit("library_classify_done", result)
        except Exception as e:
            import traceback
            print(f"[app] ERROR in library_classify: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("library_progress", {"message": f"ERROR: {e}"})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/library/classify_stats")
def library_classify_stats():
    try:
        niche = _safe_id(request.args.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    niche_path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    if not os.path.exists(niche_path):
        return jsonify({"error": "Niche not found"}), 404
    with open(niche_path, encoding="utf-8") as f:
        niche_cfg = json.load(f)
    if niche_cfg.get("library_mode") != "clip_classification":
        return jsonify({"error": "Not a clip_classification niche"}), 400

    from backend import clip_classifier
    state = clip_classifier._load_state(niche)
    cats = list((niche_cfg.get("categories") or {}).keys()) + ["_unsorted"]
    counts = clip_classifier._count_per_category(niche, cats)
    return jsonify({
        "niche":               niche,
        "categories":          counts,
        "sources_processed":   len(state.get("processed_sources", [])),
        "categorized_clips":   state.get("categorized_clips", 0),
        "unsorted_clips":      state.get("unsorted_clips", 0),
        "sources_dir":         clip_classifier._sources_dir(niche),
    })


@app.route("/api/find_competitors", methods=["POST"])
def find_competitors():
    data             = request.json or {}
    url              = data.get("url", "").strip()
    min_score        = float(data.get("min_score", 0.90))
    min_subs         = int(data.get("min_subs", 8_000))
    max_subs         = int(data.get("max_subs", 200_000))
    min_videos_month = int(data.get("min_videos_month", 15))
    min_views_month  = int(data.get("min_views_month", 30_000))
    if not url:
        return jsonify({"error": "url required"}), 400

    def run():
        try:
            results = competitor_finder.find_competitors(
                seed_url=url,
                min_score=min_score,
                min_subs=min_subs,
                max_subs=max_subs,
                min_videos_month=min_videos_month,
                min_views_month=min_views_month,
                emit=lambda msg: socketio.emit("competitor_progress", {"message": msg}),
            )
            hidden = _load_hidden_competitors()
            results = [r for r in results if r["id"] not in hidden]
            socketio.emit("competitor_done", {"results": results})
        except Exception as e:
            socketio.emit("competitor_error", {"message": str(e)})

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/niche/add_channel", methods=["POST"])
def niche_add_channel():
    data = request.json or {}
    url  = data.get("url", "").strip()
    try:
        niche = _safe_id(data.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not url:
        return jsonify({"error": "url required"}), 400

    path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    if not os.path.exists(path):
        return jsonify({"error": "niche not found"}), 404

    with open(path, encoding="utf-8") as f:
        niche_data = json.load(f)

    channels = niche_data.get("channels", [])
    if url not in channels:
        channels.append(url)
        niche_data["channels"] = channels
        with open(path, "w", encoding="utf-8") as f:
            json.dump(niche_data, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True})


@app.route("/api/stocks/stats")
def stocks_stats():
    return jsonify(stocks_library.get_stats())


@app.route("/api/stocks/analyze", methods=["POST"])
def stocks_analyze():
    global _job_active
    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        try:
            result = stocks_library.scan_and_analyze(
                emit=lambda step, msg: socketio.emit("library_progress", {"message": msg})
            )
            socketio.emit("library_done", {
                "message": f"Analyzed {result['analyzed_new']} new clips ({result['already_done']} already done)"
            })
        except Exception as e:
            socketio.emit("library_progress", {"message": f"ERROR: {e}"})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/niche/info")
def niche_info():
    """Return full niche config including pipeline_type and movie_library."""
    try:
        niche = _safe_id(request.args.get("niche", ""), "niche")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    path = os.path.join(config.NICHES_DIR, f"{niche}.json")
    if not os.path.exists(path):
        return jsonify({"error": "niche not found"}), 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/niche", methods=["POST"])
def create_niche():
    data = request.json or {}
    raw  = data.get("id", "").strip().lower().replace(" ", "_")
    try:
        name = _safe_id(raw, "niche id")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    path = os.path.join(config.NICHES_DIR, f"{name}.json")

    # If file exists and only crop fields sent — update in place
    if os.path.exists(path) and "crop_top_pct" in data and "name" not in data:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        existing["crop_top_pct"] = data.get("crop_top_pct", 0)
        existing["crop_bottom_pct"] = data.get("crop_bottom_pct", 0)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True})

    niche = {
        "name": data.get("name", name),
        "description": data.get("description", ""),
        "pipeline_type": data.get("pipeline_type", "standard"),
        "montage_style": data.get("montage_style", "standard"),
        "channels": data.get("channels", []),
        "search_keywords": data.get("keywords", []),
        "stock_tags": data.get("stock_tags", []),
    }
    if data.get("movie_library"):
        niche["movie_library"] = data["movie_library"]
    if data.get("clip_score_threshold"):
        niche["clip_score_threshold"] = data["clip_score_threshold"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(niche, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True})


def _load_hidden_competitors() -> set:
    if not os.path.exists(config.HIDDEN_COMPETITORS_FILE):
        return set()
    try:
        with open(config.HIDDEN_COMPETITORS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_hidden_competitors(hidden: set):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.HIDDEN_COMPETITORS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(hidden), f)


@app.route("/api/competitors/hide", methods=["POST"])
def hide_competitor():
    channel_id = (request.json or {}).get("id", "").strip()
    if not channel_id:
        return jsonify({"error": "id required"}), 400
    hidden = _load_hidden_competitors()
    hidden.add(channel_id)
    _save_hidden_competitors(hidden)
    return jsonify({"ok": True})


# ── Movie Library routes ──────────────────────────────────────────────────────

@app.route("/movies")
def movies_page():
    movies = movie_library.list_movies()
    languages = _languages()
    return render_template("movies.html", movies=movies, languages=languages)


@app.route("/api/movies/list")
def api_movies_list():
    return jsonify(movie_library.list_movies())


@app.route("/api/movies/process", methods=["POST"])
def api_movies_process():
    global _job_active
    data       = request.json or {}
    movie_path = data.get("movie_path", "").strip()
    movie_name = data.get("movie_name", "").strip()

    if not movie_path or not movie_name:
        return jsonify({"error": "movie_path and movie_name required"}), 400
    if not os.path.exists(movie_path):
        return jsonify({"error": f"File not found: {movie_path}"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        def _emit(step, msg):
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)
        try:
            result = movie_library.process_movie(movie_path, movie_name, emit=_emit)
            socketio.emit("movie_indexed", {
                "movie_name": movie_name,
                "clip_count": len(result.get("clips", [])),
            })
        except Exception as e:
            import traceback
            print(f"[app] movie process error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/movies/process_folder", methods=["POST"])
def api_movies_process_folder():
    global _job_active
    data        = request.json or {}
    folder_path = data.get("folder_path", "").strip()
    movie_name  = data.get("movie_name", "").strip()

    if not folder_path or not movie_name:
        return jsonify({"error": "folder_path and movie_name required"}), 400
    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Folder not found: {folder_path}"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        def _emit(step, msg):
            socketio.emit("progress", {"step": step, "message": msg})
        try:
            result = movie_library.process_movie_folder(folder_path, movie_name, emit=_emit)
            socketio.emit("movie_indexed", {
                "movie_name": movie_name,
                "clip_count": len(result.get("clips", [])),
            })
        except Exception as e:
            import traceback
            print(f"[app] movie folder process error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/movies/prepare", methods=["POST"])
def api_movies_prepare():
    global _job_active
    data       = request.json or {}
    source_url = data.get("source_url", "").strip()
    if not source_url:
        return jsonify({"error": "source_url required"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        def _emit(step, msg):
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)
        try:
            result = movie_pipeline.prepare(source_url, emit=_emit)
            socketio.emit("movie_prepare_done", result)
        except Exception as e:
            import traceback
            print(f"[app] movie prepare error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/movies/produce", methods=["POST"])
def api_movies_produce():
    global _job_active
    data       = request.json or {}
    prepare_id = data.get("prepare_id", "").strip()
    movie_name = data.get("movie_name", "").strip()
    languages  = data.get("languages", [])

    if not prepare_id or not movie_name or not languages:
        return jsonify({"error": "prepare_id, movie_name and languages required"}), 400
    if os.path.sep in prepare_id or ".." in prepare_id:
        return jsonify({"error": "Invalid prepare_id"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        def _emit(step, msg):
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)
        try:
            for lang in languages:
                socketio.emit("progress", {"step": "produce", "message": f"Starting: {lang}"})
                eventlet.sleep(0)
                result = movie_pipeline.produce(prepare_id, movie_name, lang, emit=_emit)
                socketio.emit("produce_done", result)
            socketio.emit("all_done", {})
        except Exception as e:
            import traceback
            print(f"[app] movie produce error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/movies/download/<project_id>")
def download_movie_video(project_id):
    safe_id = os.path.basename(project_id)
    proj_dir = os.path.join(config.PROJECTS_DIR, safe_id)
    output = os.path.join(proj_dir, f"{safe_id}.mp4")
    if not os.path.realpath(output).startswith(os.path.realpath(config.PROJECTS_DIR)):
        return jsonify({"error": "Invalid project id"}), 400
    if not os.path.exists(output):
        return jsonify({"error": "Not found"}), 404
    return send_file(output, as_attachment=True, download_name=f"{safe_id}.mp4")


# ── Writer routes ─────────────────────────────────────────────────────────────

@app.route("/writer")
def writer_page():
    movies = movie_library.list_movies()
    languages = _languages()
    return render_template("writer.html", movies=movies, languages=languages)


@app.route("/api/writer/generate", methods=["POST"])
def api_writer_generate():
    global _job_active
    data        = request.json or {}
    topic       = data.get("topic", "").strip()
    language    = data.get("language", "en").strip()
    style_notes = data.get("style_notes", "").strip()
    feedback    = data.get("feedback", "").strip()

    if not topic:
        return jsonify({"error": "topic required"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        try:
            script = writer_backend.generate_script(
                topic=topic, language=language,
                style_notes=style_notes, feedback=feedback,
            )
            draft_id = writer_backend.save_draft(topic, language, script, style_notes)
            socketio.emit("writer_generated", {
                "draft_id":   draft_id,
                "script":     script,
                "script_len": len(script),
            })
        except Exception as e:
            import traceback
            print(f"[app] writer generate error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/writer/metadata", methods=["POST"])
def api_writer_metadata():
    data     = request.json or {}
    topic    = data.get("topic", "").strip()
    language = data.get("language", "en").strip()
    script   = data.get("script", "").strip()

    if not script:
        return jsonify({"error": "script required"}), 400

    try:
        meta = writer_backend.generate_metadata(topic, language, script)
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/writer/produce", methods=["POST"])
def api_writer_produce():
    global _job_active
    data       = request.json or {}
    script     = data.get("script", "").strip()
    title      = data.get("title", "").strip()
    movie_name = data.get("movie_name", "").strip()
    language   = data.get("language", "en").strip()
    draft_id   = data.get("draft_id")
    metadata   = data.get("metadata", {})

    if not script:
        return jsonify({"error": "script required"}), 400
    if not movie_name:
        return jsonify({"error": "movie_name required"}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running. Wait for it to finish."}), 429
        _job_active = True

    def run():
        global _job_active
        def _emit(step, msg):
            socketio.emit("progress", {"step": step, "message": msg})
            eventlet.sleep(0)
        try:
            result = movie_pipeline.produce_from_script(
                script=script,
                title=title or "writer_video",
                movie_name=movie_name,
                language=language,
                emit=_emit,
                metadata=metadata,
            )
            socketio.emit("writer_produce_done", result)
        except Exception as e:
            import traceback
            print(f"[app] writer produce error: {e}\n{traceback.format_exc()}", flush=True)
            socketio.emit("error", {"message": str(e)})
        finally:
            with _job_lock:
                _job_active = False

    socketio.start_background_task(run)
    return jsonify({"ok": True})


@app.route("/api/cleanup", methods=["POST"])
def cleanup_old_projects():
    """Delete project folders older than N days to free disk space."""
    data = request.json or {}
    days = max(1, int(data.get("days", 7)))
    cutoff = time.time() - days * 86400
    removed_names = []
    for folder in os.listdir(config.PROJECTS_DIR):
        path = os.path.join(config.PROJECTS_DIR, folder)
        if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed_names.append(folder)
    if removed_names:
        print(f"[cleanup] Removed {len(removed_names)} folder(s) older than {days}d: {', '.join(removed_names)}", flush=True)
    return jsonify({"ok": True, "removed_folders": len(removed_names), "removed_names": removed_names, "older_than_days": days})


@app.route("/api/status")
def api_status():
    return jsonify({
        "job_running": _job_active,
        "last_msg": _job_last_msg,
        "started_at": _job_started_at,
    })


@app.route("/api/job_reset", methods=["POST"])
def api_job_reset():
    global _job_active, _job_last_msg, _job_started_at
    with _job_lock:
        _job_active = False
        _job_last_msg = ""
        _job_started_at = 0.0
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not _AUTH_USER or not _AUTH_PASS:
        print("[app] WARNING: FAA_USER/FAA_PASS not set — site is open without password!", flush=True)
    os.makedirs(config.LIBRARY_DIR, exist_ok=True)
    os.makedirs(config.NICHES_DIR,  exist_ok=True)
    os.makedirs(config.PROJECTS_DIR, exist_ok=True)
    stocks_dir = config.get_stocks_dir()
    if os.path.exists(os.path.dirname(stocks_dir)) or os.path.exists(stocks_dir):
        for _cat in config.STOCK_CATEGORIES:
            os.makedirs(os.path.join(stocks_dir, _cat), exist_ok=True)
    else:
        print(f"[app] WARNING: stocks_dir not accessible: {stocks_dir}", flush=True)
        print("[app] Stock footage will be unavailable. Set stocks_dir in Settings.", flush=True)
    host = os.environ.get("FAA_HOST", "127.0.0.1")
    port = int(os.environ.get("FAA_PORT", "5050"))
    socketio.run(app, host=host, port=port, debug=False)

# end of app.py
