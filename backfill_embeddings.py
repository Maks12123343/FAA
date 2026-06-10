"""
Backfill semantic embeddings into already-indexed movies.

Why
---
Movies indexed before semantic search existed have clips with descriptions/tags
but no "embedding" vector. This script reads each movie's index.json, computes a
vector for every clip from its EXISTING text fields (no video is opened, no Gemini
frame analysis is run — cheap and fast), and writes the vectors back.

Run once per machine after deploying the semantic-search change:

  python3 backfill_embeddings.py                 # all indexed movies
  python3 backfill_embeddings.py "Kung Fu Panda"  # one movie by name
  python3 backfill_embeddings.py --force          # recompute even if present

Safe to re-run: clips that already have an embedding are skipped unless --force.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config
from backend import embeddings as emb


def _index_path(movie_name: str) -> str:
    return os.path.join(config.get_movies_dir(), movie_name, "index.json")


def _list_indexed_movies() -> list:
    movies_dir = config.get_movies_dir()
    if not os.path.exists(movies_dir):
        return []
    out = []
    for name in os.listdir(movies_dir):
        if os.path.exists(_index_path(name)):
            out.append(name)
    return out


def backfill_movie(movie_name: str, force: bool = False) -> tuple:
    """Returns (computed, total) clip counts for the movie."""
    idx_path = _index_path(movie_name)
    if not os.path.exists(idx_path):
        print(f"  [skip] No index.json for '{movie_name}'")
        return (0, 0)

    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)

    clips = index.get("clips", [])
    if not clips:
        print(f"  [skip] '{movie_name}' has no clips")
        return (0, 0)

    need = clips if force else [c for c in clips if not c.get("embedding")]
    if not need:
        print(f"  [ok] '{movie_name}': all {len(clips)} clips already have vectors")
        return (0, len(clips))

    print(f"  '{movie_name}': computing {len(need)}/{len(clips)} vectors...")
    texts = [emb.clip_embed_text(c) for c in need]

    def _emit(step, msg):
        print(f"    {msg}")

    vectors = emb.embed_texts(texts, emit=_emit)
    if not vectors:
        print(f"  [FAIL] '{movie_name}': embeddings backend unavailable — nothing written")
        return (0, len(clips))

    computed = 0
    for c, v in zip(need, vectors):
        if v:
            c["embedding"] = v
            computed += 1

    # Atomic write so an interrupted run can't corrupt the index.
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, idx_path)

    print(f"  [done] '{movie_name}': wrote {computed} vectors → {idx_path}")
    return (computed, len(clips))


def main():
    args = [a for a in sys.argv[1:]]
    force = "--force" in args
    names = [a for a in args if not a.startswith("--")]

    if not names:
        names = _list_indexed_movies()
        if not names:
            print("No indexed movies found in", config.get_movies_dir())
            return
        print(f"Found {len(names)} indexed movie(s): {', '.join(names)}")

    total_computed = 0
    total_clips = 0
    for name in names:
        c, t = backfill_movie(name, force=force)
        total_computed += c
        total_clips += t

    print(f"\nDone. Computed {total_computed} new vectors across {total_clips} clips.")
    if total_computed:
        print("Semantic clip matching is now active for these movies.")


if __name__ == "__main__":
    main()
