# FAA — Fully Automated AI Video Generator

---

## TASK FOR CODE ASSISTANT

You are given the full codebase of the FAA system described in this document. Your task is to perform a **deep, exhaustive analysis** across two dimensions:

### 1. Technical Errors & Bugs
Find all actual or potential technical problems in the code. For each issue provide:
- Exact file name and line number
- What the bug is
- Under what conditions it triggers
- What the consequence is (crash / wrong output / data loss / silent failure)
- How to fix it

Look specifically for:
- **Race conditions** — multiple languages produce() run in one job; shared files (candidates.json, global_used_clips.json, slot_counter.json) are read/written without locks
- **Exception swallowing** — bare `except Exception: pass` that silently hides failures
- **FFmpeg subprocess errors not checked** — many subprocess.run() calls without checking returncode
- **Infinite loops or missing exit conditions** — e.g. gap-filling while loop, TTS polling loop
- **Wrong data types** — e.g. settings loaded as string instead of float/int
- **Cache invalidation bugs** — stale .analysis.json or candidates.json used after clip pool changes
- **Path traversal / security** — download endpoint, project_id sanitization
- **Memory leaks** — large clip pools kept in memory, frame bytes not released
- **Encoding issues** — file paths with non-ASCII characters on Linux
- **yt-dlp failure handling** — what happens if a URL is invalid or geo-blocked
- **Gemini API failures** — what if Vertex AI credentials expire mid-job
- **TTS timeout** — what if TTS API never returns "ending" status
- **Empty clip pool** — what happens if all 5 YouTube URLs fail to download
- **Whisper segment edge cases** — segments with start > end, overlapping segments, zero-duration segments
- **FFmpeg xfade filter** — known issues with certain codecs/frame rates in transition filter
- **Stock validation threshold** — what if ALL stock candidates score below 0.85 for every segment (video gets 0% stocks)
- **global_used_clips.json corruption** — what if file is partially written and next language reads it

### 2. Conceptual & Architectural Weaknesses
Evaluate the design decisions. For each weakness provide:
- What the problem is
- Why it is a problem (what goes wrong in practice)
- How severe it is (critical / significant / minor)
- A concrete suggestion to improve it

Look specifically for:
- **Script quality**: The rewriter uses the ORIGINAL VIDEO's transcript (competitor content) as the base. If the source video is low quality, badly structured, or off-topic, the rewritten script inherits these flaws. Is there a better approach?
- **Tag-based matching quality**: Competitor clips are now matched purely by keyword overlap between Gemini-assigned tags and script section text. How reliable is this? What types of mismatches will commonly occur?
- **Stock validation bottleneck**: ~60 Gemini API calls per video just for stock validation (0.85 threshold). If Pexels results are poor quality for a niche, most will fail and fall back to competitors anyway — defeating the purpose.
- **5-video uniqueness math**: With 5 videos of ~150 clips each = 750 clip slots. 60% competitor = 450 competitor slots. Pool from 5 videos at ~150 clips each = ~750 clips. With max 2 uses per clip, pool can fill ~1500 slots — this seems fine, but what if the niche has fewer good clips after reject filtering?
- **Sequential language production**: Languages are produced one by one. A 5-language batch takes 5× the single-language time. Could be parallelized safely since each language writes to its own project folder.
- **No retry on failed video**: If produce() throws an exception for language 3, languages 4 and 5 still run. The user has no way to re-run just language 3 without restarting the whole batch.
- **Claude script expansion**: If Claude's initial script is under 20,000 characters, _expand_script() is called to add content. This content is generated WITHOUT access to the original competitor video — Claude just continues writing about the topic. Quality and accuracy risk.
- **Whisper on TTS audio**: The system uses Whisper to segment the TTS-generated voiceover (not the original video). This means clip cuts happen at speech pauses in the AI voiceover. Is this the best approach, or would sentence-based splitting be more accurate?
- **Clip pool exhaustion**: After producing several batches for the same niche, global_used_clips.json grows. But it is scoped to a single prepare session (not across sessions). So the same clips can appear in different prepare sessions. Is this intentional?
- **No quality validation on final video**: The system checks duration and file size but does NOT check if the video actually looks good — e.g. if 80% of clips were rejected and replaced with fallback pool clips that may be visually irrelevant.
- **Pixabay music**: Background music is fetched from Pixabay but there is no validation that the downloaded audio is the right duration or format. What happens if the music file is corrupt?
- **Error reporting to user**: Errors during produce() are emitted as SocketIO "error" events. But if the browser tab was closed or reconnected, the user may never see the error and just sees a stalled UI.
- **Settings stored as plain JSON**: API keys (Claude, YouTube, TTS, Pexels) are stored in data/settings.json as plain text. If the server is compromised, all keys are exposed. No encryption, no environment variable isolation beyond the .env file.
- **Single niche limitation**: Currently only one niche (china_economy.json) exists. How well does the system generalize to other niches with different visual vocabularies?

### Output Format

Structure your response as:

```
## CRITICAL BUGS (will cause crashes or data loss)
[list]

## SIGNIFICANT BUGS (wrong behavior, silent failures)
[list]

## MINOR BUGS (edge cases, rarely triggered)
[list]

## CRITICAL ARCHITECTURAL WEAKNESSES
[list]

## SIGNIFICANT WEAKNESSES
[list]

## MINOR / SUGGESTIONS
[list]
```

For every item include: file, line (if applicable), problem description, impact, fix/recommendation.

Be specific and reference actual code. Do not give generic advice. If you say "add error handling", specify exactly what error, where, and what the handler should do.

---

## What This System Does

FAA is a production-grade automated pipeline that creates YouTube faceless documentary-style videos. Given a YouTube niche (e.g. "china economy"), the system:
1. Finds the top-performing competitor video on that topic
2. Rewrites its transcript into a full voiceover script using Claude AI
3. Generates professional voiceover audio via ElevenLabs-compatible TTS
4. Assembles a finished 1080p video by combining footage clips with the voiceover
5. Produces the same video in up to 5 languages simultaneously

The entire process runs on a remote Linux VPS (DigitalOcean, Ubuntu, 4 vCPU / 8 GB RAM) and is controlled through a password-protected web interface.

---

## Infrastructure

```
User Browser
     │
     ▼
Nginx (port 80/443)  ←── reverse proxy, handles SSL
     │
     ▼
Flask + SocketIO (port 5050, internal only)
     │
     ├── /opt/faa/                   ← app root on server
     │   ├── data/                   ← settings, niches, library
     │   ├── projects/               ← generated video projects
     │   └── stocks/                 ← local stock footage
     │
     └── /mnt/gdrive/FAA/stocks/     ← Google Drive mounted via rclone
```

- **App runs as**: systemd service `faa.service`, user `faa`
- **Google Drive mount**: `faa-gdrive.service` (rclone FUSE mount)
- **Auto-cleanup**: `faa-cleanup.timer` removes project folders older than 7 days
- **Real-time UI updates**: Flask-SocketIO with eventlet (WebSocket events stream progress to browser)
- **Concurrency**: Only ONE video generation job runs at a time (`_job_lock` + `_job_active` flag)

---

## Configuration

All runtime settings live in `data/settings.json` and are managed through the `/settings` UI page. Key settings:

| Setting | Purpose |
|---|---|
| `claude_api_key` | Anthropic API key for script rewriting |
| `claude_model` | Currently `claude-sonnet-4-6` |
| `tts_api_key` | ElevenLabs-compatible TTS API key |
| `tts_api_url` | TTS endpoint (custom proxy at voiceapi.csv666.ru) |
| `vertex_project_id` | Google Cloud project for Gemini API |
| `gemini_model` | Currently `gemini-2.5-flash` |
| `youtube_api_key` / `_2` / `_3` | YouTube Data API keys (auto-rotated on quota exceeded) |
| `pexels_api_key` | Pexels stock footage API |
| `competitor_ratio` | 0.60 = 60% competitor clips, 40% stock footage |
| `voice_profiles` | Per-language ElevenLabs voice IDs and settings |
| `stocks_dir` | Path to stock footage folder (local or rclone mount) |

---

## Niches

A "niche" is a topic configuration stored as `data/niches/{name}.json`:

```json
{
  "name": "China Economy",
  "description": "Videos about China's economic growth and infrastructure",
  "channels": ["https://youtube.com/@SomeChannel"],
  "search_keywords": ["china economy", "china gdp"],
  "stock_tags": ["china", "construction", "economy"]
}
```

The niche defines which YouTube channels to scan for source videos and what keywords to use for stock footage searches.

---

## Full Production Pipeline

### Phase 1: PREPARE (triggered by "Analyze" button)

**Goal**: Find best source video, get its transcript. No video production yet.

```
channel_scanner.find_top_video(niche)
    └── Calls YouTube Data API
    └── Scans all channels in niche, fetches video stats
    └── Returns the video with highest views in last 30 days

channel_scanner.get_video_metadata(url)
    └── Gets title, description, tags from YouTube API

transcriber.get_transcript(url)
    └── Tries YouTube auto-subtitles first (fast, free)
    └── Falls back to yt-dlp subtitle extraction
    └── Returns full text transcript
```

**Result saved to**: `projects/_prepare_{id}/state.json`

**Returned to UI**: source video URL, title, views, transcript preview (2000 chars)

---

### Phase 2: PRODUCE (triggered after user pastes competitor video URLs)

**Input**: prepare_id + list of 5 competitor YouTube URLs + list of languages (e.g. ["en", "pl", "de", "fr", "es"])

**Runs once per language. Each language is processed sequentially.**

#### Step 1: Download & Cut Competitor Clips
```
clip_downloader.build_pool(youtube_urls, pool_dir)
    └── Downloads 2 videos in parallel (ThreadPoolExecutor, max_workers=2)
    └── For each video:
        └── yt-dlp downloads at max 720p, no audio
        └── FFmpeg scene detection (threshold 0.35)
        └── Cuts at scene boundaries into 2–5 second clips
        └── If no scene changes: fixed-duration cuts
        └── Saves clips as {video_id}_{index:04d}.mp4
        └── Saves {video_id}_index.json with metadata per clip
    └── Saves combined clips_index.json
```

**Pool is shared across all languages** — downloaded only once per prepare session.

#### Step 2: Rewrite Script (Claude)
```
rewriter.rewrite_all(transcript, language, source_title, description, tags)
    └── Call 1: _rewrite_script()
        └── System prompt from data/rewrite_prompt.txt
        └── Rewrites transcript as voiceover script in target language
        └── Auto-continues if Claude hits max_tokens (up to 5 parts)
        └── If result < 20,000 chars: _expand_script() adds more content
    └── Call 2: _rewrite_metadata()
        └── System prompt from data/metadata_prompt.txt
        └── Generates 5 title options, description, tags
        └── Based on SOURCE video's metadata (not rewritten script)
```

**Output**: `script.txt`, `metadata.json`, `source.txt` in project folder

#### Step 3: Generate Voiceover (TTS)
```
tts.generate(script, language, audio_path)
    └── Looks up voice_id from voice_profiles[language] in settings
    └── POST /tasks to TTS API with ElevenLabs multilingual_v2 model
    └── Polls GET /tasks/{id}/status every 5 seconds (max 600s)
    └── Downloads MP3 when status == "ending"
    └── Saves as voiceover.mp3
```

#### Step 4: Segment Voiceover (Whisper)
```
transcriber.transcribe_segments(audio_path)
    └── Runs Whisper (via Google Vertex AI or local)
    └── Returns segments: [{text, start, end}, ...]
    └── These timestamps define exactly when each clip cuts
    └── Cached as whisper_segments.json
```

#### Step 5: Analyze Clips with Gemini (one-time per clip, cached)
```
clip_matcher.analyze_all_clips(clips_index)
    └── 5 parallel workers (ThreadPoolExecutor)
    └── For each new clip (skips if .analysis.json exists):
        └── Extracts 3 frames (start/middle/end) via FFmpeg
        └── Sends to Gemini with prompt asking for:
            - description (1-2 sentences)
            - 10-15 tags
            - category (city/factory/nature/technology/etc.)
            - quality checks (is_blurry, is_static)
            - overlay checks → action: use / crop_bottom / crop_corner / reject
        └── Saves result as {clip}.analysis.json (permanent cache)
```

**Gemini API**: Google Vertex AI, uses application_default_credentials.json

#### Step 6: Match Clips to Script Sections (tag-based, free)
```
clip_matcher.match_clips_multi(section_texts, clips_index, top_n=10)
    └── Splits transcript into ~35-word sections
    └── For each section: scores every clip by tag/description overlap
    └── Returns top 10 clip candidates per section
    └── Cached as candidates.json (shared across languages)
```

**No AI calls here** — pure string matching.

#### Step 7: Assemble Clip List
```
pipeline._assemble_clips_from_candidates(candidates, whisper_segments, pool, audio_dur, slot, global_used)
```

This is the core assembly logic. For each Whisper segment:

**Decision: stock (40%) or competitor clip (60%)?**

```
IF stock (random.random() >= 0.60) AND seg_text not empty:
    pick_stock_clips(seg_text, n=5)
        └── Score local stocks by tag overlap
        └── Fallback to Pexels API if not enough local matches
        └── Returns up to 5 candidates
    
    For each candidate:
        └── Skip if used in THIS video already
        └── Skip if global_used count >= 1 (stocks must be 100% unique across all 5 videos)
        └── validate_stock_for_section(clip, seg_text)
            └── Extracts 3 frames → Gemini rates visual match 0.0–1.0
            └── Result cached as {clip}.stockval_{hash}.json
            └── Retry 3x on rate limit (5/10/15s wait)
        └── If score >= 0.85: USE THIS STOCK
    
    If no stock passes: fall back to competitor clip

ELSE competitor clip:
    From section's top-10 tag-matched candidates:
        └── Skip if used in THIS video
        └── Skip if global_used[clip] >= 2 (max 2 uses across 5 videos)
        └── Skip if action == reject/crop_bottom/crop_corner
        └── Skip if same source video used 3+ times in a row
        └── Skip if same category used 2+ times in a row
        └── First clip that passes all checks: USE IT
    
    If no candidate: next_competitor() from shuffled fallback pool
```

**global_used dict** tracks clip usage across all language productions for this prepare session. Saved to `global_used_clips.json` after each language. Ensures:
- Stocks: max 1 use (fully unique per video)
- Competitor clips: max 2 uses out of 5 videos (~30% overlap)

**Slot system**: each language gets a different starting offset into the candidates list, ensuring language 1 and language 2 prefer different clips even from the same section.

#### Step 8: Generate Text Overlays
```
text_renderer.generate_stat_overlays(script, audio_dur)
    └── Extracts numbers/percentages from script
    └── Schedules them as animated lower-third overlays
    └── Also adds key sentence overlays every ~3.5 minutes
```

#### Step 9: Assemble Video (FFmpeg)
```
montage.assemble(clips, audio_path, output_path, text_overlays)

Step A: _build_concat() — parallel clip processing
    └── 2 parallel workers
    └── For each clip:
        └── _prepare_clip(): scale to 1920x1080, set fps=30, trim, -pix_fmt yuv420p
            └── crop_bottom action: zoom 1.18x to cut subtitle strip
            └── crop_corner action: zoom 1.10x to cut watermark corners
        └── _uniqualize_clip(): consistent zoom+color grade + per-clip grain noise
            └── Same zoom/brightness/contrast for all clips in video (no flicker)
            └── Only grain varies per clip (±4-10)
    └── Groups clips into 3–6 clip segments
    └── Within each segment: hard cuts (concat demuxer)
    └── Between segments: xfade transitions (fade/dissolve/blur/etc, 0.35s)
    └── Outputs _raw_video.mp4

Step B: _add_audio()
    └── If Pixabay API key set: downloads background music track
        └── Music mixed at 2% volume under voiceover
    └── Mux voiceover MP3 + raw video
    └── -c:v copy (no re-encode), -c:a aac 192k
    └── -movflags +faststart (moov atom at start for instant playback)
    └── Outputs _with_audio.mp4

Step C: apply_text_overlays() (if overlays exist)
    └── FFmpeg drawtext filter with slide-up animation + fade in/out
    └── Outputs final output.mp4
```

All intermediate files (`_raw_video.mp4`, `_with_audio.mp4`) are cached — if a step crashes and restarts, it continues from where it left off.

#### Step 10: Validation
```
Checks output.mp4:
    └── File exists
    └── Size > 100 KB
    └── Duration > 10 seconds
    └── Duration >= 90% of voiceover length (catches incomplete renders)
```

---

## Stock Footage System

Two sources, used together:

**Local stocks** (`stocks_dir` folder, organized by category):
- Categories: construction, ships_ports, energy, cities, technology, infrastructure, military, space, nature, general
- Each clip analyzed once by Gemini → `.analysis.json` with tags/description
- `stocks_library.scan_and_analyze()` processes new clips in bulk

**Pexels API** (fallback when local stocks are insufficient):
- Searches by first 5 words of segment text
- Downloads best 16:9 HD video
- Analyzes with Gemini, saves to `stocks/general/` folder
- Rate-limited, uses `pexels_api_key` from settings

---

## Competitor Discovery

`competitor_finder.find_competitors(seed_url)`:
- Takes a seed channel URL
- Finds similar channels via YouTube recommendations + search
- Filters by: subscriber count (8K–200K), upload frequency (15+/month), views (30K+/month)
- Scores similarity using Gemini
- Results stored in `data/niches/{niche}.json` channels list
- Hidden competitors tracked in `data/competitors_hidden.json`

---

## Media Library

`media_library.py` manages per-niche downloaded clip pools:
- `download_from_channel(url, niche)`: downloads full channel, cuts into clips
- `validate_library(niche, description)`: runs Gemini analysis on all clips
- Stats available per niche (total clips, analyzed, categories)

---

## Key File Locations (on server)

```
/opt/faa/
├── app.py                          # Flask app, all API routes
├── config.py                       # Paths, defaults, settings I/O
├── backend/
│   ├── pipeline.py                 # Main orchestration (prepare + produce)
│   ├── channel_scanner.py          # YouTube channel scanning
│   ├── transcriber.py              # Transcript extraction + Whisper segments
│   ├── rewriter.py                 # Claude script + metadata rewriting
│   ├── tts.py                      # ElevenLabs TTS API
│   ├── clip_downloader.py          # yt-dlp + FFmpeg scene cutting
│   ├── clip_matcher.py             # Gemini clip analysis + tag matching
│   ├── clip_sourcer.py             # YouTube search-based sourcing (legacy)
│   ├── stocks_library.py           # Local + Pexels stock management
│   ├── media_library.py            # Per-niche clip library
│   ├── competitor_finder.py        # YouTube competitor discovery
│   ├── montage.py                  # FFmpeg video assembly
│   ├── text_renderer.py            # Animated text overlays
│   └── aligner.py                  # Whisper segment utilities
├── data/
│   ├── settings.json               # All runtime config (edited via UI)
│   ├── rewrite_prompt.txt          # Claude system prompt for script rewriting
│   ├── metadata_prompt.txt         # Claude system prompt for metadata
│   └── niches/
│       └── china_economy.json      # Niche config
├── projects/
│   ├── _prepare_{id}/              # Prepare session data
│   │   ├── state.json              # Source video info + transcript
│   │   ├── clip_pool/              # Downloaded competitor clips (.mp4 + .analysis.json)
│   │   ├── candidates.json         # Tag-matched clip candidates (cached)
│   │   ├── global_used_clips.json  # Cross-video uniqueness tracker
│   │   └── slot_counter.json       # Slot index for language diversity
│   └── {niche}_{lang}_{timestamp}/ # Finished video project
│       ├── output.mp4              # Final video
│       ├── script.txt              # Rewritten voiceover script
│       ├── metadata.json           # Title options, description, tags
│       ├── voiceover.mp3           # Generated TTS audio
│       └── whisper_segments.json   # Audio timestamps (cached)
└── templates/                      # Jinja2 HTML templates
    ├── index.html                  # Main generate page
    ├── settings.html               # Settings editor
    ├── library.html                # Media library manager
    └── competitors.html            # Competitor discovery
```

---

## Performance Characteristics

| Step | Time | Notes |
|---|---|---|
| Prepare (scan + transcribe) | 1–3 min | Fast |
| Download 5 videos + cut clips | 10–20 min | Parallel (2 at a time) |
| Rewrite script (Claude) | 3–8 min | Multi-part for long scripts |
| TTS generation | 5–15 min | Depends on script length |
| Whisper segmentation | 2–5 min | Cached after first language |
| Gemini clip analysis | 10–20 min | 5 parallel workers, cached permanently |
| Tag matching | < 1 min | No AI, pure string ops |
| Stock validation (Gemini) | 5–15 min | ~60 segments × ~1 call each |
| FFmpeg assembly | 5–15 min | 2 parallel workers |
| **Total per language** | **~40–60 min** | First language; subsequent languages ~20–30 min (most steps cached) |

---

## Important Design Decisions

1. **No per-segment Gemini validation** — Competitor clips are matched by tag overlap only. This was the original bottleneck (1500+ API calls). Removed in favor of trusting tag-based matching when clips come from videos on the same topic.

2. **Two-phase pipeline** (prepare → produce) — Allows the user to review the source video and manually select competitor URLs before production begins. Also allows producing multiple languages from one prepare session without re-downloading or re-transcribing.

3. **Aggressive caching** — Every expensive operation caches to disk: clip analysis (`.analysis.json`), stock validation (`.stockval_{hash}.json`), Whisper segments, tag candidates. This means interrupted jobs resume fast, and subsequent languages are much cheaper.

4. **Global used clips tracker** — `global_used_clips.json` ensures that across 5 language productions: stock clips appear in max 1 video, competitor clips appear in max 2 videos (~30% overlap).

5. **Uniqualization** — Each video gets consistent video-level color grading (same zoom/brightness/saturation) with per-clip grain variation. Applied at assembly time, not pre-processed, so only used clips are processed.

6. **FFmpeg codec compatibility** — All encoding uses `-pix_fmt yuv420p` (H.264 baseline compatible with all players) and `-movflags +faststart` (moov atom at file start for instant playback).
