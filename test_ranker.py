"""
Quick standalone test of the ranker: takes 5 fake candidates and one segment,
calls the real rank_clips_by_text() like the pipeline does, prints scores
and the exact API exception if anything fails. No TTS, no video, no credits burned.

Usage:  python test_ranker.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)

from backend.movie_library import rank_clips_by_text  # noqa: E402

# Five fake candidates — 1 is the perfect match (Tigress training), the rest are filler
candidates = [
    {
        "id": "test_001",
        "file": "/tmp/test_001.mp4",
        "characters": ["Tigress"],
        "scene_type": "training",
        "emotion": "determination",
        "themes": ["trauma", "identity"],
        "tags": ["dojo", "punching", "intense"],
        "description": "Tigress trains hard in the dojo, punching a wooden post, sweat on her face.",
    },
    {
        "id": "test_002",
        "file": "/tmp/test_002.mp4",
        "characters": ["Po"],
        "scene_type": "comedy",
        "emotion": "joy",
        "themes": ["acceptance"],
        "tags": ["panda", "dumplings", "happy"],
        "description": "Po eats dumplings happily in his father's noodle shop.",
    },
    {
        "id": "test_003",
        "file": "/tmp/test_003.mp4",
        "characters": ["Tai Lung"],
        "scene_type": "fight",
        "emotion": "anger",
        "themes": ["rejection", "anger"],
        "tags": ["villain", "fight"],
        "description": "Tai Lung breaks out of prison in a rage.",
    },
    {
        "id": "test_004",
        "file": "/tmp/test_004.mp4",
        "characters": [],
        "scene_type": "credits",
        "emotion": "neutral",
        "themes": [],
        "tags": ["text", "logo"],
        "description": "End credits scroll over a black background.",
    },
    {
        "id": "test_005",
        "file": "/tmp/test_005.mp4",
        "characters": ["Tigress", "Shifu"],
        "scene_type": "emotional_dialogue",
        "emotion": "vulnerability",
        "themes": ["trauma", "vulnerability"],
        "tags": ["confession", "softlight"],
        "description": "Tigress and Shifu have a quiet conversation about her past, both looking distressed.",
    },
]

# A real-looking narration segment in Polish (matches what was failing in production)
segment_text = "Tygrysica trenowała całe życie, żeby zostać Smoczym Wojownikiem."
prev_text = "Jako dziecko Tygrysica była odrzucona przez wszystkich w sierocińcu."
next_text = "Ale to nigdy nie wystarczyło, żeby zasłużyć na akceptację Shifu."

score_rules = {
    "character_mentioned_bonus": 0.40,
    "main_character_bonus": 0.20,
    "wrong_character_penalty": -0.20,
    "no_relevant_character_penalty": -0.15,
    "character_bonus_cap": 0.50,
    "scene_penalties": {"credits": -1.0, "title_card": -1.0},
    "theme_bonus_per_match": 0.05,
    "theme_bonus_cap": 0.20,
    "psychology_themes": ["trauma", "identity", "isolation", "fear", "rejection", "vulnerability"],
}

print("=" * 70)
print("Calling rank_clips_by_text with 5 candidates")
print(f"Segment (PL): {segment_text}")
print(f"Main character: Tigress")
print("=" * 70)

ranked = rank_clips_by_text(
    candidates=candidates,
    segment_text=segment_text,
    prev_text=prev_text,
    next_text=next_text,
    main_characters=["Tigress"],
    score_rules=score_rules,
)

print()
print("Result (sorted, best first):")
for clip, score, breakdown in ranked:
    chars = ", ".join(clip.get("characters") or []) or "—"
    print(f"  score={score:+.3f} id={clip['id']} scene={clip['scene_type']} chars=[{chars}]")
    if breakdown:
        print(f"      breakdown: {breakdown}")

print()
print("Expected: test_001 or test_005 should win (Tigress + relevant scene).")
print("If all scores are 0.5 — the API call failed. Check console for [ranker] lines above.")
