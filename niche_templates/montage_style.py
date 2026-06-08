"""
Montage style configuration for this niche.

Standard montage: simple cuts, text overlays, stocks mixed with clips
Movie montage: Ken Burns, speed ramping, flash cuts, cinematic effects
"""

# For STANDARD niches:
MONTAGE_CONFIG = {
    "style": "standard",  # or "cinematic"
    "clip_min_duration": 2,
    "clip_max_duration": 5,
    "competitor_ratio": 0.60,  # 60% competitor clips, 40% stocks
    "text_overlay_style": "numbers_and_phrases",  # simple numeric overlays
    "transitions": ["fade", "dissolve"],  # simple transitions
}

# For CINEMATIC / MOVIE niches:
CINEMATIC_CONFIG = {
    "style": "cinematic",
    "clip_min_duration": 1.5,
    "clip_max_duration": 3.5,
    "effects": {
        "ken_burns": True,
        "speed_ramping": True,
        "flash_cuts": True,
        "vignette": True,
    },
    "transitions": [
        "fade", "dissolve", "fadeblack", "hblur",
        "wipeleft", "wiperight", "slideleft", "slideright",
    ],
    "text_overlay_style": "cinematic_text_screens",  # big centered text
}
