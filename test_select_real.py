"""
Reproduces exactly what _select_clips_for_segments does in production:
loads real clip data from Kung Fu Panda index.json, runs Vertex semantic
search, then calls rank_clips_by_text in parallel for 10 sample segments.

Catches and prints any exception with full traceback — so we see exactly
why production gets score=0.00 for everything.

Usage:  python test_select_real.py
"""
import eventlet
eventlet.monkey_patch()

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)

from backend.movie_library import (  # noqa: E402
    rank_clips_by_text, search_clips, get_movie_clips,
)


MOVIE = "Kung Fu Panda"
MAIN_CHARACTERS = ["Tigress"]

# 10 short fake Polish segments — enough to test parallel ranking
SEGMENTS = [
    "Tygrysica trenowała całe życie, żeby zostać Smoczym Wojownikiem.",
    "Jako dziecko była odrzucona przez wszystkich w sierocińcu.",
    "Shifu nigdy nie powiedział jej, że jest dumny.",
    "Pewnego dnia przyszedł Oogway i wybrał Po.",
    "Tygrysica była wstrząśnięta tym wyborem.",
    "Po początku się z niej śmiała, ale potem zrozumiała.",
    "W bitwie z Tai Lungiem stanęła ramię w ramię z innymi.",
    "Tai Lung pokonał całą piątkę bez problemu.",
    "Tygrysica leżała ranna na śniegu, patrząc w niebo.",
    "Ale ona wstała znowu, bo zawsze tak robiła.",
]

# Niche score rules (same as psychology_movies.json)
NICHE_PATH = os.path.join(HERE, "data", "niches", "psychology_movies.json")
with open(NICHE_PATH, encoding="utf-8") as f:
    niche_cfg = json.load(f)
SCORE_RULES = niche_cfg.get("score_rules", {})

print(f"Loading clips from '{MOVIE}'...")
all_clips = get_movie_clips(MOVIE)
print(f"Got {len(all_clips)} clips")
all_known_chars = set()
for c in all_clips:
    for ch in c.get("characters", []) or []:
        if ch:
            all_known_chars.add(ch)
print(f"Known characters: {len(all_known_chars)} unique")

# Sequential ranking
print()
print("=" * 60)
print(f"Ranking {len(SEGMENTS)} segments sequentially")
print("=" * 60)

def _do_one(idx: int):
    seg_text = SEGMENTS[idx]
    prev_text = SEGMENTS[idx - 1] if idx > 0 else ""
    next_text = SEGMENTS[idx + 1] if idx + 1 < len(SEGMENTS) else ""

    t0 = time.time()
    try:
        cands = search_clips(
            seg_text, movie_name=MOVIE,
            used_ids=set(), top_n=5, gemini_validate=False,
        )
        if not cands:
            print(f"[seg#{idx}] NO CANDIDATES from search_clips")
            return

        ranked = rank_clips_by_text(
            candidates=cands,
            segment_text=seg_text,
            prev_text=prev_text,
            next_text=next_text,
            main_characters=MAIN_CHARACTERS,
            score_rules=SCORE_RULES,
            all_known_chars=all_known_chars,
        )
        elapsed = time.time() - t0
        if not ranked:
            print(f"[seg#{idx}] EMPTY ranked in {elapsed:.1f}s")
            return
        top_clip, top_score, _ = ranked[0]
        chars = ", ".join(top_clip.get("characters") or []) or "—"
        all_scores = [r[1] for r in ranked]
        all_zero = all(abs(s) <= 0.0001 for s in all_scores)
        marker = " ⚠ ALL ZERO" if all_zero else ""
        print(f"[seg#{idx}] {elapsed:.1f}s top_score={top_score:+.2f} top_chars=[{chars}]{marker}")
    except Exception as e:
        print(f"[seg#{idx}] EXCEPTION in {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()


t0 = time.time()
for i in range(len(SEGMENTS)):
    _do_one(i)
print(f"Sequential: {time.time()-t0:.1f}s")

# Parallel ranking
print()
print("=" * 60)
print(f"Ranking {len(SEGMENTS)} segments in parallel (4 workers)")
print("=" * 60)
t0 = time.time()
pool = eventlet.GreenPool(size=4)
for i in range(len(SEGMENTS)):
    pool.spawn(_do_one, i)
pool.waitall()
print(f"Parallel: {time.time()-t0:.1f}s")
