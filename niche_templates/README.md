# Niche Template

## How to add a new niche (e.g., "mashana")

1. **Copy this folder** to `data/niches/mashana/` (or any name)

2. **Edit `config.json`**:
   - Change `name` to your niche display name
   - Set `pipeline_type`: `"standard"` (for YouTube clips + stocks) or `"movie"` (for film clips)
   - For `movie` type, add `"movie_library": ["Movie Name"]` pointing to indexed movies
   - Fill `channels` with competitor YouTube channels
   - Fill `search_keywords` for auto-discovery
   - Fill `stock_tags` for stock footage search

3. **Customize `generate_prompt.py`** (optional):
   - Edit the prompts to match your niche's writing style
   - This controls how scripts are rewritten

4. **Customize `montage_style.py`** (optional):
   - Adjust montage settings for your niche's visual style

5. **Save config** to `data/niches/mashana.json` (not in a subfolder — directly in `data/niches/`)

6. **Restart the server** — the new niche will appear in the Generate dropdown automatically

## Example: "mashana" (Ukrainian village memes)

```json
{
  "name": "Mashana",
  "description": "Ukrainian village life, farming memes, rural culture",
  "pipeline_type": "standard",
  "montage_style": "standard",
  "title_keywords": ["mashana", "village", "ukrainian", "farming", "rural"],
  "channels": [
    "https://www.youtube.com/@UkrainianVillageChannel"
  ],
  "search_keywords": [
    "ukrainian village life",
    "farming ukraine",
    "rural culture"
  ],
  "stock_tags": [
    "ukrainian village", "farm", "tractor", "chickens",
    "countryside", "fields", "traditional house"
  ],
  "clip_score_threshold": 0.85
}
```

## Example: "psychology_movies" (film analysis)

```json
{
  "name": "Psychology Movies",
  "description": "Psychology of movies, characters, dark themes, film analysis",
  "pipeline_type": "movie",
  "montage_style": "cinematic",
  "title_keywords": ["psychology", "movie", "film", "character"],
  "channels": [
    "https://www.youtube.com/@NerdSync"
  ],
  "search_keywords": [
    "psychology of movies"
  ],
  "stock_tags": [
    "cinema", "movie theater", "dark psychology"
  ],
  "clip_score_threshold": 0.80,
  "movie_library": ["Kung Fu Panda"]
}
```
