"""Final MP4 polish: short fade-in and metadata cleanup."""

import os
import subprocess
import time

import config


def finalize_mp4(path: str, fade_in: float = 0.8, emit=None) -> bool:
    """Rewrite final MP4 with clean metadata and a short visual fade-in.

    Returns True when the file was replaced. On failure the original file is
    kept and the caller can continue; this step should not kill production.
    """
    if not path or not os.path.exists(path):
        return False

    root, ext = os.path.splitext(path)
    tmp = f"{root}.finalizing{ext or '.mp4'}"
    if os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except Exception:
            pass

    vf = f"fade=t=in:st=0:d={float(fade_in):.2f}" if fade_in and fade_in > 0 else "null"
    cmd = [
        config.FFMPEG,
        "-y",
        "-i",
        path,
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        vf,
        *config.get_video_encoder_args("fast", crf=23),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-metadata",
        "title=",
        "-metadata",
        "comment=",
        "-metadata",
        "description=",
        "-metadata:s:v:0",
        "handler_name=VideoHandler",
        "-metadata:s:a:0",
        "handler_name=SoundHandler",
        "-movflags",
        "+faststart",
        tmp,
    ]

    if emit:
        emit("finalize", "Cleaning MP4 metadata and adding fade-in...")
    started = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 100_000:
            err = r.stderr.decode(errors="replace")[-800:] if r else "unknown ffmpeg error"
            print(f"[video_finalize] Final polish failed: {err}", flush=True)
            if emit:
                emit("finalize", "Final metadata/fade step failed; keeping original video.")
            return False
        os.replace(tmp, path)
        elapsed = time.time() - started
        print(f"[video_finalize] Final polish done in {elapsed:.1f}s: {path}", flush=True)
        if emit:
            emit("finalize", f"Final metadata/fade step done ({elapsed:.1f}s).")
        return True
    except Exception as e:
        print(f"[video_finalize] Final polish failed: {e}", flush=True)
        if emit:
            emit("finalize", "Final metadata/fade step failed; keeping original video.")
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
