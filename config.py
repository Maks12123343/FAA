import json
import os
import platform
import shutil
import subprocess

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
LIBRARY_DIR  = os.path.join(DATA_DIR, "library")
NICHES_DIR   = os.path.join(DATA_DIR, "niches")
HIDDEN_COMPETITORS_FILE = os.path.join(DATA_DIR, "competitors_hidden.json")
PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "projects")

STOCK_CATEGORIES = [
    "construction",
    "ships_ports",
    "energy",
    "cities",
    "technology",
    "infrastructure",
    "military",
    "space",
    "nature",
    "general",
]

if platform.system() == "Windows":
    _ffmpeg_fixed = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
    _ffprobe_fixed = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe"
    FFMPEG  = _ffmpeg_fixed if os.path.exists(_ffmpeg_fixed) else (shutil.which("ffmpeg") or "ffmpeg")
    FFPROBE = _ffprobe_fixed if os.path.exists(_ffprobe_fixed) else (shutil.which("ffprobe") or "ffprobe")
    # Prefer an explicitly supplied local credential path, then the current
    # Windows user's gcloud ADC location. Do not hard-code the developer's
    # username: the repo is also run from a different Windows account.
    _cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    _cred_home = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Roaming", "gcloud", "application_default_credentials.json",
    )
    _cred_legacy = r"C:\Users\Ukraine\AppData\Roaming\gcloud\application_default_credentials.json"
    VERTEX_CREDENTIALS = next(
        (p for p in [_cred_env, _cred_home, _cred_legacy] if p and os.path.exists(p)),
        _cred_env or _cred_home,
    )
    _DEFAULT_STOCKS_DIR = r"G:\My Drive\FAA\stocks"
    _DEFAULT_MOVIES_DIR = r"G:\My Drive\FAA\movies"
else:
    FFMPEG  = shutil.which("ffmpeg")  or "ffmpeg"
    FFPROBE = shutil.which("ffprobe") or "ffprobe"
    # Vast.ai / RunPod: credentials in home dir or app dir
    _cred_app = "/opt/faa/.config/gcloud/application_default_credentials.json"
    _cred_home = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    _cred_workspace = "/workspace/FAA/.config/gcloud/application_default_credentials.json"
    VERTEX_CREDENTIALS = next(
        (p for p in [_cred_workspace, _cred_home, _cred_app] if os.path.exists(p)),
        _cred_home,
    )
    # Stocks/movies: local copy preferred (faster), rclone mount as fallback
    _local_stocks = os.path.join(os.path.dirname(__file__), "stocks")
    _local_movies = os.path.join(os.path.dirname(__file__), "movies")
    _mount_stocks = "/mnt/gdrive/stocks"
    _mount_movies = "/mnt/gdrive/movies"
    _DEFAULT_STOCKS_DIR = _local_stocks if os.path.isdir(_local_stocks) else _mount_stocks
    _DEFAULT_MOVIES_DIR = _local_movies if os.path.isdir(_local_movies) else _mount_movies

STOCKS_DIR = _DEFAULT_STOCKS_DIR

DEFAULT_SETTINGS = {
    # Paths
    "stocks_dir":  _DEFAULT_STOCKS_DIR,
    "movies_dir":  _DEFAULT_MOVIES_DIR,

    # Vertex AI
    "vertex_project_id": "",
    "vertex_location": "us-central1",
    "gemini_model": "gemini-2.5-flash",

    # Google Flow image bridge (local service; browser profile stays outside the repo)
    "gemini_image_enabled": False,
    "gemini_image_bridge_url": "http://127.0.0.1:4981",
    "gemini_image_api_key": "",
    "gemini_image_model": "flow-nano-pro",
    "gemini_image_timeout": 600,
    "gemini_image_double_preview": False,

    # Automatic download of ready projects (runs locally on the client machine)
    "auto_download_enabled": False,
    "auto_download_base_url": "http://127.0.0.1:5050",
    "auto_download_out_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FAA_downloads")),
    "auto_download_interval_minutes": 1,
    "auto_download_languages": "fi,hu,bg,da,de,ro,sv,cs,tr,pl,ja,it,sk,ko,es",
    "auto_download_watch_new_only": False,
    "auto_download_all_ready": True,
    "auto_download_retries": 5,
    "auto_download_timeout": 30,
    "auto_download_download_timeout": 7200,

    # Rewrite API (OpenAI-compatible chat completions)
    "rewrite_active_provider": "a6api",
    "rewrite_fallback_provider": "",
    "rewrite_providers": {
        "a6api": {
            "name": "A6API",
            "api_key": "",
            "model": "gpt-5.5",
            "api_url": "https://a6api.com/v1/chat/completions",
            "reasoning_effort": "high",
            "max_tokens": "12000",
        },
        "byesu": {
            "name": "Byesu",
            "api_key": "",
            "model": "gpt-5.5",
            "api_url": "https://byesu.com/v1/chat/completions",
            "reasoning_effort": "none",
            "max_tokens": "12000",
        },
        "custom": {
            "name": "Custom",
            "api_key": "",
            "model": "",
            "api_url": "",
            "reasoning_effort": "none",
            "max_tokens": "12000",
        },
    },
    "rewrite_api_key": "",
    "rewrite_model": "gpt-5.5",
    "rewrite_api_url": "https://a6api.com/v1/chat/completions",
    "rewrite_reasoning_effort": "high",
    "rewrite_max_tokens": "12000",
    "rewrite_script_enabled": True,
    "rewrite_thumbnail_enabled": True,
    "rewrite_metadata_enabled": True,
    "rewrite_chunks": 6,

    # TTS
    "tts_api_key": "",
    "tts_api_url": "https://qw1voicegencore.pro",
    "tts_voice_engine": "elevenLabsV3",
    "tts_settings_preset": "standard",
    "tts_chunk_size": 1200,
    "tts_delay_between_chunks": 0.5,
    "tts_thread_count": 10,

    # YouTube API keys (rotated automatically when quota exceeded)
    "youtube_api_key":   "",
    "youtube_api_key_2": "",
    "youtube_api_key_3": "",

    # Voice profiles: language code → voice settings
    "voice_profiles_version": 2,
    "voice_profiles": {
        "fi": {"name": "Finnish Voice", "voice_id": "ESapivUCtGNuYKDCwzcI", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "hu": {"name": "Hungarian Voice", "voice_id": "M336tBVZHWWiWb4R54ui", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "bg": {"name": "Bulgarian Voice", "voice_id": "iWNf11sz1GrUE4ppxTOL", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "da": {"name": "Danish Voice", "voice_id": "ygiXC2Oa1BiHksD3WkJZ", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "de": {"name": "German Voice", "voice_id": "jdKpAe6rxAe99tFGbsAc", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "ro": {"name": "Romanian Voice", "voice_id": "8nBBDfYxYXmDNaqTCxPH", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "sv": {"name": "Swedish Voice", "voice_id": "QTGiyJvep6bcx4WD1qAq", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "cs": {"name": "Czech Voice", "voice_id": "7FpO7yFcBAfqM6vZJCg7", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "tr": {"name": "Turkish Voice", "voice_id": "LCHGt3rsPMP50Vs28amI", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "pl": {"name": "Polish Voice", "voice_id": "1nUkvoDFCcCTjJk9U8mL", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "ja": {"name": "Japanese Voice", "voice_id": "H8ZPDxbrPcks5hEsi2fq", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "it": {"name": "Italian Voice", "voice_id": "fzDFBB4mgvMlL36gPXcz", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "sk": {"name": "Slovak Voice", "voice_id": "Zai7B4Aol2bJtneyq0L1", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "ko": {"name": "Korean Voice", "voice_id": "8lidWTlnwgjObqCImnE2", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "es": {"name": "Spanish Voice", "voice_id": "OjrdP8Z2fWjVyt0scrL7", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "uk": {"name": "Ukrainian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
    },

    # Unofficial Gemini cookie API (optional — doubles throughput alongside Vertex AI)
    # Fill psid / psidts from browser cookies at gemini.google.com
    "gemini_cookies": {
        "psid":   "",
        "psidts": "",
    },

    # Validation
    "clip_score_threshold": 0.85,
    "clip_frames_positions": [0.01, 0.10, 0.50, 0.90],

    # Montage
    "clip_min_duration": 2,
    "clip_max_duration": 5,
    "stock_max_duration": 6,
    "competitor_ratio": 0.60,
    "output_width": 1920,
    "output_height": 1080,
    "fps": 30,
}


def _coerce_settings(data: dict) -> dict:
    """Ensure numeric settings are correct types even if stored/sent as strings."""
    float_fields = {
        "competitor_ratio":    (0.0,  1.0),
        "clip_score_threshold":(0.0,  1.0),
        "clip_min_duration":   (0.1,  None),
        "clip_max_duration":   (0.1,  None),
        "stock_max_duration":  (0.1,  None),
    }
    int_fields = {
        "output_width":  1,
        "output_height": 1,
        "fps":           1,
        "rewrite_chunks": 1,
        "gemini_image_timeout": 30,
        "auto_download_interval_minutes": 1,
        "auto_download_retries": 1,
        "auto_download_timeout": 5,
        "auto_download_download_timeout": 60,
    }
    bool_fields = {
        "rewrite_script_enabled",
        "rewrite_thumbnail_enabled",
        "rewrite_metadata_enabled",
        "gemini_image_enabled",
        "gemini_image_double_preview",
        "auto_download_enabled",
        "auto_download_watch_new_only",
        "auto_download_all_ready",
    }
    for key, (lo, hi) in float_fields.items():
        if key in data:
            try:
                val = float(data[key])
                if lo is not None:
                    val = max(lo, val)
                if hi is not None:
                    val = min(hi, val)
                data[key] = val
            except (TypeError, ValueError):
                data.pop(key, None)  # Drop invalid — DEFAULT_SETTINGS fallback covers it
    for key, minimum in int_fields.items():
        if key in data:
            try:
                val = max(minimum, int(data[key]))
                if key == "rewrite_chunks":
                    val = min(10, val)
                if key == "gemini_image_timeout":
                    val = min(1800, val)
                data[key] = val
            except (TypeError, ValueError):
                data.pop(key, None)
    for key in bool_fields:
        if key in data:
            val = data[key]
            if isinstance(val, str):
                data[key] = val.strip().lower() in {"1", "true", "yes", "on", "checked"}
            else:
                data[key] = bool(val)
    # Guard: clip_min <= clip_max
    lo = data.get("clip_min_duration")
    hi = data.get("clip_max_duration")
    if lo is not None and hi is not None and lo > hi:
        data["clip_max_duration"] = lo
    return data


_settings_cache = {"data": None, "mtime": 0.0}

def load_settings() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        mtime = os.path.getmtime(SETTINGS_FILE)
    except OSError:
        mtime = 0.0
    if _settings_cache["data"] is not None and _settings_cache["mtime"] == mtime:
        return _settings_cache["data"].copy()
    with open(SETTINGS_FILE, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    known_data = {key: value for key, value in data.items() if key in DEFAULT_SETTINGS}
    merged = _coerce_settings({**DEFAULT_SETTINGS, **known_data})
    # One-time migration for the built-in VoiceGen language set. The settings
    # file is intentionally ignored by Git, so existing installations need a
    # local migration after pulling the updated defaults. Keep all unrelated
    # settings (including credentials) untouched.
    if data.get("voice_profiles_version") != DEFAULT_SETTINGS["voice_profiles_version"]:
        merged["voice_profiles"] = {
            code: dict(profile)
            for code, profile in DEFAULT_SETTINGS["voice_profiles"].items()
        }
        merged["voice_profiles_version"] = DEFAULT_SETTINGS["voice_profiles_version"]
        merged["auto_download_languages"] = DEFAULT_SETTINGS["auto_download_languages"]
        save_settings(merged)
    _settings_cache["data"] = merged
    _settings_cache["mtime"] = mtime
    return merged.copy()


def save_settings(settings: dict):
    """Atomic settings write — crash during write won't corrupt the file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = SETTINGS_FILE + ".tmp"
    clean = {key: settings.get(key, value) for key, value in DEFAULT_SETTINGS.items()}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, SETTINGS_FILE)
    _settings_cache["data"] = None
    _settings_cache["mtime"] = 0.0


def get_setting(key: str):
    return load_settings().get(key)


def get_stocks_dir() -> str:
    return load_settings().get("stocks_dir", STOCKS_DIR)


def get_movies_dir() -> str:
    return load_settings().get("movies_dir", _DEFAULT_MOVIES_DIR)


def _qsv_available() -> bool:
    if not hasattr(_qsv_available, "_cached"):
        try:
            subprocess = __import__("subprocess")
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "h264_qsv" not in r.stdout:
                _qsv_available._cached = False
            else:
                # FFmpeg can list h264_qsv even when the machine has no
                # usable Intel iGPU (for example an i5-9400F). Verify a real
                # encode so clip preparation falls back to libx264 instead of
                # silently failing every selected clip.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    test = subprocess.run(
                        [FFMPEG, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                         "-f", "lavfi",
                         "-i", "color=black:size=256x256:duration=0.1",
                         "-c:v", "h264_qsv", "-frames:v", "1", tmp_path],
                        capture_output=True, timeout=15,
                    )
                    _qsv_available._cached = test.returncode == 0
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        except Exception:
            _qsv_available._cached = False
        print(f"[config] h264_qsv available: {_qsv_available._cached}", flush=True)
    return _qsv_available._cached


def _nvenc_available() -> bool:
    if not hasattr(_nvenc_available, "_cached"):
        try:
            import tempfile
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "h264_nvenc" not in r.stdout:
                _nvenc_available._cached = False
            else:
                # Do a real test encode — nvenc may be compiled in but CUDA absent
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    r2 = subprocess.run(
                        [FFMPEG, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                         "-f", "lavfi",
                         "-i", "color=black:size=256x256:duration=0.1",
                         "-c:v", "h264_nvenc", "-frames:v", "1", tmp_path],
                        capture_output=True, timeout=15,
                    )
                    _nvenc_available._cached = r2.returncode == 0
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        except Exception:
            _nvenc_available._cached = False
        print(f"[config] h264_nvenc available: {_nvenc_available._cached}", flush=True)
    return _nvenc_available._cached


def get_video_encoder_args(preset: str = "ultrafast", crf: int = None) -> list:
    """Return ffmpeg video encoder args optimized for current platform.
    Priority in auto mode:
      Windows  → h264_qsv, then h264_nvenc
      Linux    → h264_nvenc
      Fallback → libx264 (CPU)
    crf: quality for libx264/nvenc (18=high, 23=default, 28=lower).
    """
    preference = os.environ.get("FAA_VIDEO_ENCODER", "auto").strip().lower()
    if preference in {"cpu", "libx264", "x264"}:
        return _libx264_args(preset, crf)

    if preference in {"auto", "qsv"} and platform.system() == "Windows" and _qsv_available():
        qsv_preset_map = {
            "ultrafast": "veryfast",
            "superfast": "veryfast",
            "veryfast":  "veryfast",
            "faster":    "faster",
            "fast":      "fast",
            "medium":    "medium",
            "slow":      "slow",
        }
        args = ["-c:v", "h264_qsv", "-preset", qsv_preset_map.get(preset, "veryfast")]
        if crf is not None:
            args += ["-global_quality", str(crf)]
        return args

    if preference in {"auto", "nvenc"} and _nvenc_available():
        nvenc_preset_map = {
            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast":  "p3",
            "faster":    "p4",
            "fast":      "p4",
            "medium":    "p5",
            "slow":      "p6",
        }
        args = ["-c:v", "h264_nvenc", "-preset", nvenc_preset_map.get(preset, "p4")]
        if crf is not None:
            args += ["-cq", str(crf)]
        return args

    if preference in {"qsv", "nvenc"}:
        print(f"[config] requested encoder '{preference}' unavailable; falling back to libx264", flush=True)
    return _libx264_args(preset, crf)


def _libx264_args(preset: str, crf: int | None) -> list:
    args = ["-c:v", "libx264", "-preset", preset]
    if crf is not None:
        args += ["-crf", str(crf)]
    return args


def get_video_encoder_name() -> str:
    """Return and log the actual encoder selected for this machine."""
    args = get_video_encoder_args("fast")
    try:
        name = args[args.index("-c:v") + 1]
    except (ValueError, IndexError):
        name = "unknown"
    if not hasattr(get_video_encoder_name, "_logged"):
        print(f"[config] selected video encoder: {name} ({' '.join(args)})", flush=True)
        get_video_encoder_name._logged = True
    return name
