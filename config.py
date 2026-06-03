import json
import os
import platform
import shutil

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
    FFMPEG  = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
    FFPROBE = r"C:\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe"
    VERTEX_CREDENTIALS = r"C:\Users\Ukraine\AppData\Roaming\gcloud\application_default_credentials.json"
    _DEFAULT_STOCKS_DIR = r"G:\My Drive\FAA\stocks"
    _DEFAULT_MOVIES_DIR = r"G:\My Drive\FAA\movies"
else:
    FFMPEG  = shutil.which("ffmpeg")  or "ffmpeg"
    FFPROBE = shutil.which("ffprobe") or "ffprobe"
    # faa system user has no home dir, credentials stored under app dir
    _cred_app = "/opt/faa/.config/gcloud/application_default_credentials.json"
    _cred_home = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    VERTEX_CREDENTIALS = _cred_app if os.path.exists(_cred_app) else _cred_home
    # On Linux stocks are served from Google Drive mounted via rclone at /mnt/gdrive
    _DEFAULT_STOCKS_DIR = "/mnt/gdrive/FAA/stocks"
    _DEFAULT_MOVIES_DIR = "/mnt/gdrive/FAA/movies"

STOCKS_DIR = _DEFAULT_STOCKS_DIR

DEFAULT_SETTINGS = {
    # Paths
    "stocks_dir":  _DEFAULT_STOCKS_DIR,
    "movies_dir":  _DEFAULT_MOVIES_DIR,

    # Vertex AI
    "vertex_project_id": "",
    "vertex_location": "us-central1",
    "gemini_model": "gemini-2.5-flash",

    # Claude API
    "claude_api_key": "",
    "claude_model": "claude-sonnet-4-6",

    # TTS
    "tts_api_key": "",
    "tts_api_url": "https://voiceapi.csv666.ru",

    # YouTube API keys (rotated automatically when quota exceeded)
    "youtube_api_key":   "",
    "youtube_api_key_2": "",
    "youtube_api_key_3": "",

    # Voice profiles: language code → voice settings
    "voice_profiles": {
        "en": {"name": "English Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "pl": {"name": "Polish Voice",  "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "de": {"name": "German Voice",  "voice_id": "", "stability": 0.80, "similarity_boost": 0.75, "speed": 1.0},
        "fr": {"name": "French Voice",  "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "es": {"name": "Spanish Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "it": {"name": "Italian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "pt": {"name": "Portuguese Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "uk": {"name": "Ukrainian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "ru": {"name": "Russian Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
        "tr": {"name": "Turkish Voice", "voice_id": "", "stability": 0.85, "similarity_boost": 0.75, "speed": 1.0},
    },

    # Pioneer.ai API (OpenAI-compatible — used for script writing & rewriting)
    # Add up to N keys for parallel validation (1 key per thread)
    "pioneer_api_keys": [],
    "pioneer_model": "ca143171-0ff9-4ca9-86e6-f5731d36bdea",
    "pioneer_api_url": "https://api.pioneer.ai/v1/chat/completions",

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
                data[key] = max(minimum, int(data[key]))
            except (TypeError, ValueError):
                data.pop(key, None)
    # Guard: clip_min <= clip_max
    lo = data.get("clip_min_duration")
    hi = data.get("clip_max_duration")
    if lo is not None and hi is not None and lo > hi:
        data["clip_max_duration"] = lo
    return data


def load_settings() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    with open(SETTINGS_FILE, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    merged = {**DEFAULT_SETTINGS, **data}
    return _coerce_settings(merged)


def save_settings(settings: dict):
    """Atomic settings write — crash during write won't corrupt the file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = SETTINGS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, SETTINGS_FILE)


def get_setting(key: str):
    return load_settings().get(key)


def get_stocks_dir() -> str:
    return load_settings().get("stocks_dir", STOCKS_DIR)


def get_movies_dir() -> str:
    return load_settings().get("movies_dir", _DEFAULT_MOVIES_DIR)


def _qsv_available() -> bool:
    if not hasattr(_qsv_available, "_cached"):
        try:
            r = __import__("subprocess").run(
                [FFMPEG, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            _qsv_available._cached = "h264_qsv" in r.stdout
        except Exception:
            _qsv_available._cached = False
        print(f"[config] h264_qsv available: {_qsv_available._cached}", flush=True)
    return _qsv_available._cached


def _nvenc_available() -> bool:
    if not hasattr(_nvenc_available, "_cached"):
        try:
            r = __import__("subprocess").run(
                [FFMPEG, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            _nvenc_available._cached = "h264_nvenc" in r.stdout
        except Exception:
            _nvenc_available._cached = False
        print(f"[config] h264_nvenc available: {_nvenc_available._cached}", flush=True)
    return _nvenc_available._cached


def get_video_encoder_args(preset: str = "ultrafast", crf: int = None) -> list:
    """Return ffmpeg video encoder args optimized for current platform.
    Priority:
      Windows  → h264_qsv  (Intel Quick Sync, iGPU)
      Linux    → h264_nvenc (NVIDIA GPU, e.g. Paperspace/Vast.ai)
      Fallback → libx264   (CPU)
    crf: quality for libx264/nvenc (18=high, 23=default, 28=lower).
    """
    if platform.system() == "Windows" and _qsv_available():
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

    if platform.system() != "Windows" and _nvenc_available():
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

    args = ["-c:v", "libx264", "-preset", preset]
    if crf is not None:
        args += ["-crf", str(crf)]
    return args
