"""
Clear cached clips.json for a project so next run re-validates all clips.
Also clears validation cache files (.val_txt_*.json) for the movie.

Usage:
  python3 clear_project_cache.py                  # clears ALL project clips.json
  python3 clear_project_cache.py <prepare_id>     # clears specific project
  python3 clear_project_cache.py --movie <name>   # clears movie validation cache
  python3 clear_project_cache.py --all            # clears everything
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config


def clear_project_clips(prepare_id=None):
    """Delete clips.json from project dirs so clips get re-selected."""
    count = 0
    for d in os.listdir(config.PROJECTS_DIR):
        dpath = os.path.join(config.PROJECTS_DIR, d)
        if not os.path.isdir(dpath):
            continue
        if prepare_id and prepare_id not in d:
            continue
        clips_json = os.path.join(dpath, "clips.json")
        if os.path.exists(clips_json):
            os.unlink(clips_json)
            print(f"  Deleted: {clips_json}")
            count += 1
    return count


def clear_movie_validation_cache(movie_name=None):
    """Delete .val_txt_*.json files next to movie clips."""
    movies_dir = config.get_movies_dir()
    count = 0
    if movie_name:
        search_dirs = [os.path.join(movies_dir, movie_name, "clips")]
    else:
        search_dirs = []
        for m in os.listdir(movies_dir):
            clips_dir = os.path.join(movies_dir, m, "clips")
            if os.path.isdir(clips_dir):
                search_dirs.append(clips_dir)

    for clips_dir in search_dirs:
        for f in os.listdir(clips_dir):
            if ".val_txt_" in f or ".val_" in f:
                fp = os.path.join(clips_dir, f)
                os.unlink(fp)
                count += 1
    return count


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or "--all" in args:
        print("Clearing ALL project clips.json...")
        n = clear_project_clips()
        print(f"  {n} clips.json deleted\n")

        print("Clearing ALL movie validation cache...")
        n = clear_movie_validation_cache()
        print(f"  {n} cache files deleted\n")

    elif "--movie" in args:
        idx = args.index("--movie")
        name = args[idx + 1] if idx + 1 < len(args) else None
        print(f"Clearing validation cache for movie: {name or 'ALL'}...")
        n = clear_movie_validation_cache(name)
        print(f"  {n} cache files deleted")

    else:
        prepare_id = args[0]
        print(f"Clearing clips.json for projects matching: {prepare_id}")
        n = clear_project_clips(prepare_id)
        print(f"  {n} clips.json deleted")

    print("Done!")
