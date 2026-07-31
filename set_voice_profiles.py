"""Install the required VoiceGen language profiles into data/settings.json."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import config


VOICE_PROFILES = {
    "fi": ("Finnish Voice", "ESapivUCtGNuYKDCwzcI"),
    "hu": ("Hungarian Voice", "M336tBVZHWWiWb4R54ui"),
    "bg": ("Bulgarian Voice", "iWNf11sz1GrUE4ppxTOL"),
    "da": ("Danish Voice", "ygiXC2Oa1BiHksD3WkJZ"),
    "hr": ("Croatian Voice", "DAGnQ7r9sMtV0Q44g1Mi"),
    "ro": ("Romanian Voice", "8nBBDfYxYXmDNaqTCxPH"),
    "sv": ("Swedish Voice", "QTGiyJvep6bcx4WD1qAq"),
    "cs": ("Czech Voice", "7FpO7yFcBAfqM6vZJCg7"),
    "tr": ("Turkish Voice", "LCHGt3rsPMP50Vs28amI"),
    "pl": ("Polish Voice", "1nUkvoDFCcCTjJk9U8mL"),
}

DEFAULT_PROFILE_SETTINGS = {
    "stability": 0.85,
    "similarity_boost": 0.75,
    "speed": 1.0,
}


def main() -> None:
    settings_path = Path(config.SETTINGS_FILE)
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = settings_path.with_name(f"settings.backup.{stamp}.json")
        shutil.copy2(settings_path, backup_path)
        print(f"backup: {backup_path}")

    settings = config.load_settings()
    profiles = settings.setdefault("voice_profiles", {})

    for code, (name, voice_id) in VOICE_PROFILES.items():
        current = profiles.get(code, {})
        if not isinstance(current, dict):
            current = {}
        profile = {**DEFAULT_PROFILE_SETTINGS, **current}
        profile["name"] = name
        profile["voice_id"] = voice_id
        profiles[code] = profile

    # Old settings used "sw" for Swedish. Keep only the correct ISO code.
    old_sw = profiles.get("sw")
    if isinstance(old_sw, dict) and "swedish" in (old_sw.get("name", "").lower()):
        profiles.pop("sw", None)

    config.save_settings(settings)

    saved = config.load_settings().get("voice_profiles", {})
    for code in sorted(VOICE_PROFILES):
        profile = saved.get(code, {})
        print(f"{code}: {profile.get('name', '')} {profile.get('voice_id', '')}")

    print("done")


if __name__ == "__main__":
    main()
