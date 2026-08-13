"""Print the actual local Whisper and FFmpeg acceleration available to FAA."""

from __future__ import annotations

import os
import platform
import shutil

import config


def main() -> int:
    print(f"OS: {platform.platform()}")
    print(f"CPU threads: {os.cpu_count() or 1}")
    print(f"FFmpeg: {config.FFMPEG}")
    print(
        "FFmpeg exists: "
        f"{os.path.isfile(config.FFMPEG) if os.path.isabs(config.FFMPEG) else shutil.which(config.FFMPEG) is not None}"
    )
    print(f"Video encoder: {config.get_video_encoder_name()}")
    print(f"Encoder args: {' '.join(config.get_video_encoder_args('fast'))}")

    try:
        import torch

        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    except Exception as exc:
        print(f"Torch check failed: {type(exc).__name__}: {exc}")

    print(f"Whisper requested device: {os.environ.get('FAA_WHISPER_DEVICE', 'auto')}")
    print(f"Whisper model: {os.environ.get('FAA_WHISPER_MODEL', 'large-v3')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
