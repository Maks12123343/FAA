import os
import platform
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FFMPEG = config.FFMPEG

if platform.system() == "Windows":
    FONT_PATH = r"C:\Windows\Fonts\arialbd.ttf"
else:
    _candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    FONT_PATH = next((f for f in _candidates if os.path.exists(f)), _candidates[-1])


def _esc_textfile(text: str) -> str:
    """Escape text for FFmpeg drawtext textfile content."""
    return (
        text.replace("\\", "\\\\")
            .replace("%", "%%")
            .replace("\n", " ")
            .replace("\r", "")
    )


def _build_drawtext(overlay: dict, text_dir: str, idx: int) -> str:
    text = _esc_textfile(overlay["text"])
    start = overlay.get("start", 0.0)
    duration = overlay.get("duration", 3.0)
    position = overlay.get("position", "bottom-right")
    size = overlay.get("size", 36)
    color = overlay.get("color", "white")
    bg_color = overlay.get("bg_color", "black@0.45")

    fade_in = 0.5
    fade_out = 0.5
    slide_px = 20

    end = start + duration

    if position == "bottom-right":
        base_x, base_y = "w-tw-40", "h-th-60"
    elif position == "bottom-left":
        base_x, base_y = "40", "h-th-60"
    elif position == "top-right":
        base_x, base_y = "w-tw-40", "60"
    elif position == "center":
        base_x, base_y = "(w-tw)/2", "(h-th)/2"
    else:
        base_x, base_y = "w-tw-40", "h-th-60"

    txt_file = os.path.join(text_dir, f"overlay_{idx}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text)

    slide_expr = f"min(1,(t-{start:.2f})/{fade_in:.2f})"
    y_expr = f"{base_y}+{slide_px}-{slide_px}*{slide_expr}"

    alpha_expr = (
        f"if(lt(t-{start:.2f},{fade_in:.2f}),"
        f"(t-{start:.2f})/{fade_in:.2f},"
        f"if(gt(t,{end:.2f}-{fade_out:.2f}),"
        f"({end:.2f}-t)/{fade_out:.2f},"
        f"1))"
    )

    enable = f"between(t,{start:.2f},{end:.2f})"

    font_esc = FONT_PATH.replace("\\", "/")
    txt_esc = txt_file.replace("\\", "/")

    return (
        f"drawtext=fontfile='{font_esc}'"
        f":textfile='{txt_esc}'"
        f":fontsize={size}"
        f":fontcolor={color}"
        f":alpha='{alpha_expr}'"
        f":box=1:boxcolor={bg_color}:boxborderw=8"
        f":x='{base_x}':y='{y_expr}'"
        f":enable='{enable}'"
    )


def apply_text_overlays(input_path: str, overlays: list, output_path: str):
    """
    overlays: list of dicts with keys:
      text, start (sec), duration (sec), position, size, color, bg_color
    """
    if not overlays:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    import tempfile
    import shutil as _shutil

    text_dir = tempfile.mkdtemp(prefix="faa_txt_")
    try:
        filters = [_build_drawtext(o, text_dir, i) for i, o in enumerate(overlays)]
        vf = ",".join(filters)

        subprocess.run(
            [FFMPEG, "-y", "-i", input_path,
             "-vf", vf,
             *config.get_video_encoder_args("fast"), "-pix_fmt", "yuv420p",
             "-c:a", "copy",
             "-movflags", "+faststart",
             output_path],
            check=True, timeout=3600,
        )
    finally:
        _shutil.rmtree(text_dir, ignore_errors=True)


def generate_stat_overlays(script: str, audio_duration: float) -> list:
    """
    Generate text overlay events from script:
    - Numbers/percentages → corner stat overlay
    - Key phrases → center overlay every ~3-4 minutes
    """
    import re
    overlays = []

    # Find digit numbers and word-form numbers (Claude writes numbers as words per prompt)
    digit_stats = re.findall(
        r'\b\d[\d,.]*(?:\s*(?:%|percent|billion|trillion|million|thousand))?\b', script
    )
    word_stats = re.findall(
        r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|'
        r'[\w]+-[\w]+)(?:\s+(?:billion|trillion|million|thousand|hundred|percent))?\b',
        script, re.IGNORECASE,
    )
    numbers = digit_stats + [w.strip() for w in word_stats if 2 < len(w.strip()) < 40]
    unique_stats = list(dict.fromkeys(numbers))[:15]

    interval = audio_duration / max(len(unique_stats), 1)
    for i, stat in enumerate(unique_stats):
        t = i * interval + random.uniform(2, 5)
        if t >= audio_duration - 5:
            break
        overlays.append({
            "text": stat,
            "start": round(t, 1),
            "duration": random.uniform(2.5, 4.0),
            "position": "bottom-right",
            "size": 38,
            "color": "white",
            "bg_color": "black@0.5",
        })

    # Key phrase overlays every ~3.5 minutes
    phrase_interval = 210  # seconds
    phrases = re.findall(r'[A-Z][^.!?]{20,60}[.!?]', script)
    t = phrase_interval
    for phrase in phrases[:5]:
        if t >= audio_duration - 10:
            break
        overlays.append({
            "text": phrase[:50],
            "start": round(t, 1),
            "duration": 3.5,
            "position": "center",
            "size": 44,
            "color": "white",
            "bg_color": "black@0.55",
        })
        t += phrase_interval + random.uniform(-20, 20)

    overlays.sort(key=lambda x: x["start"])
    return overlays
