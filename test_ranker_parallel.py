"""
Stress test: call rank_clips_by_text() 8 times in parallel via eventlet
(same as production pipeline does). If single-threaded works but parallel
breaks — that's where the production bug is.

Usage: python test_ranker_parallel.py
"""
import eventlet
eventlet.monkey_patch()  # match what app.py does at startup

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)

from backend.movie_library import rank_clips_by_text  # noqa: E402

candidates = [
    {"id": f"c{i}", "file": f"/tmp/c{i}.mp4",
     "characters": ["Tigress"] if i in (0, 4) else ["Po"],
     "scene_type": "training" if i == 0 else "comedy",
     "emotion": "determination",
     "themes": ["trauma"],
     "tags": ["dojo"],
     "description": f"Test clip {i} — Tigress in training" if i in (0, 4) else f"Test clip {i} — Po eats"}
    for i in range(5)
]

score_rules = {
    "character_mentioned_bonus": 0.40,
    "main_character_bonus": 0.20,
    "wrong_character_penalty": -0.20,
    "no_relevant_character_penalty": -0.15,
    "character_bonus_cap": 0.50,
    "scene_penalties": {"credits": -1.0, "title_card": -1.0},
    "theme_bonus_per_match": 0.05,
    "theme_bonus_cap": 0.20,
    "psychology_themes": ["trauma", "fear"],
}

segments = [
    "Tygrysica trenowała w dojo.",
    "Tygrysica była zmęczona.",
    "Tygrysica patrzyła w lustro.",
    "Tygrysica czuła się odrzucona.",
    "Tygrysica spotkała Shifu.",
    "Tygrysica skoczyła w powietrze.",
    "Tygrysica uderzała w worek.",
    "Tygrysica medytowała sama.",
]

def _one_call(idx: int):
    t0 = time.time()
    try:
        ranked = rank_clips_by_text(
            candidates=candidates,
            segment_text=segments[idx],
            prev_text="",
            next_text="",
            main_characters=["Tigress"],
            score_rules=score_rules,
        )
        elapsed = time.time() - t0
        top = ranked[0] if ranked else None
        if top:
            print(f"[seg#{idx}] OK in {elapsed:.1f}s: top={top[0]['id']} score={top[1]:.2f}", flush=True)
        else:
            print(f"[seg#{idx}] FAIL in {elapsed:.1f}s: empty ranked", flush=True)
    except Exception as e:
        print(f"[seg#{idx}] EXCEPTION in {time.time()-t0:.1f}s: {type(e).__name__}: {e}", flush=True)


print("=" * 60)
print("Sequential test (1 call at a time):")
print("=" * 60)
seq_start = time.time()
for i in range(len(segments)):
    _one_call(i)
print(f"Sequential total: {time.time()-seq_start:.1f}s")

print()
print("=" * 60)
print("Parallel test (4 workers via eventlet.GreenPool):")
print("=" * 60)
par_start = time.time()
pool = eventlet.GreenPool(size=4)
for i in range(len(segments)):
    pool.spawn(_one_call, i)
pool.waitall()
print(f"Parallel total: {time.time()-par_start:.1f}s")
