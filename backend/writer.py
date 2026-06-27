"""
Writer — generates YouTube voiceover scripts FROM SCRATCH using Claude.
Approval flow: generate → user reviews/edits → approve → produce.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

MIN_SCRIPT_LENGTH = 20000


def _psychology_movie_appendix(topic: str, style_notes: str) -> str:
    hint = f"{topic}\n{style_notes}".lower()
    triggers = ("movie", "film", "character", "villain", "hero", "cartoon", "animation", "psychology")
    if sum(1 for token in triggers if token in hint) < 2:
        return ""
    return (
        "Psychology-movies mode:\n"
        "- Open with a contradiction or painful truth about the character, not with summary.\n"
        "- Treat the character like a psychological case study: visible behavior, hidden wound, defense mechanism, fear, and need.\n"
        "- Use concrete scenes and actions from the film instead of generic abstraction.\n"
        "- Build sections around escalating insight: surface trait -> inner conflict -> deeper wound -> turning point.\n"
        "- End on an unsettling or emotionally clarifying insight, not a bland recap.\n"
    )


_GENERATE_SYSTEM = (
    "You are a professional YouTube scriptwriter specializing in psychology, "
    "self-improvement, history, and documentary-style video essays. "
    "You write complete, deeply researched, emotionally engaging voiceover scripts."
)

_GENERATE_PROMPT = """\
Write a complete, engaging voiceover script for a YouTube video.

Topic: {topic}
Language: {language}
{style_notes}

Requirements:
- Length: at least 20,000 characters — write a FULL, DETAILED script (this is critical)
- Structure: powerful hook → context/background → main content (4-6 sections with depth) → emotional conclusion
- Style: documentary-style narration, psychological depth, storytelling, vivid examples
- Voiceover rules:
  * Numbers spelled out (not "100" but "one hundred")
  * Smooth, natural sentences — easy to read aloud
  * No ads, no channel mentions, no "subscribe" calls
  * No markdown headers in the script — just flowing prose paragraphs
  * Rhetorical questions to engage the listener
  * Concrete examples, real stories, surprising facts
- Return ONLY the script text inside a code block (```), nothing else
"""

_EXPAND_SYSTEM = (
    "You are a professional scriptwriter reviewing your own voiceover script. "
    "The script is too short. Write a natural continuation that fills gaps and adds depth."
)

_METADATA_SYSTEM = """\
You are a professional YouTube SEO specialist, copywriter, and marketer.
All instructions below are mandatory and have the highest priority over any other directions.

## MAIN TASK
You are provided with a voiceover script for a YouTube video. Your goal is to generate
fully optimized YouTube metadata (titles, description, tags) based on the script content.

## TARGET LANGUAGE
All metadata must be written exclusively in the target language specified by the user.
Exception: each title must also include a Ukrainian translation (see Title rules below).

## TITLE RULES
Provide exactly 5 different title options.
Each option must be written in the target language, followed by an em-dash ( — ) and its
accurate translation into Ukrainian.
Format: [Title in target language] — [Ukrainian translation]
The 5 options must represent different rewriting styles:
  * Options 1 and 2 (Direct): straightforward, close to the topic, different words.
  * Options 3 and 4 (Moderate): altered structure, more intrigue, same core angle.
  * Option 5 (Strong): powerful, creative, high-CTR title with fresh delivery.

## DESCRIPTION RULES
Write a deep, structured description in the target language. Break into logical paragraphs.
The first 2-3 sentences must be highly engaging (shown in YouTube search results).
Naturally integrate thematic keywords without keyword stuffing.

## TAGS RULES
Generate relevant tags separated by commas, exclusively in the target language.
Tags must be ranked by relevance (most important first).
STRICT LENGTH CONSTRAINT: total characters in the tags block (tags + commas + spaces)
must be STRICTLY between 490 and 500 characters. Adjust tag count/length to fit exactly.

## OUTPUT FORMAT (STRICTLY MANDATORY)
### Optimized Titles:
1. [Title] — [Ukrainian translation]
2. [Title] — [Ukrainian translation]
3. [Title] — [Ukrainian translation]
4. [Title] — [Ukrainian translation]
5. [Title] — [Ukrainian translation]

### Optimized Description:
[Description text]

### Optimized Tags:
[tags, separated by commas, total 490-500 characters]
"""


# ── Claude helper ─────────────────────────────────────────────────────────────

from backend import api_client

def _call_claude(system: str, messages: list, timeout: int = 180) -> tuple:
    """Call Pioneer first, GigaCoder GPT as fallback."""
    try:
        return api_client.call_pioneer(system, messages, timeout=timeout, use_rewrite_model=False)
    except Exception as e:
        print(f"[writer] Pioneer failed ({e}), falling back to GigaCoder", flush=True)
        return api_client.call_gigacoder(system, messages, timeout=timeout)


def _extract_code_block(text: str) -> str:
    if not text:
        raise RuntimeError("LLM returned empty response — check API key and connectivity")
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ── Script generation ─────────────────────────────────────────────────────────

def generate_script(topic: str, language: str,
                    style_notes: str = "", feedback: str = "") -> str:
    """
    Generate a voiceover script from scratch about the given topic.
    If feedback is provided, it's included as correction instructions.
    Returns the full script text.
    """
    auto_notes = _psychology_movie_appendix(topic, style_notes)
    combined_style = style_notes.strip()
    if auto_notes:
        combined_style = (combined_style + "\n\n" + auto_notes).strip() if combined_style else auto_notes

    prompt = _GENERATE_PROMPT.format(
        topic=topic,
        language=language,
        style_notes=f"Additional style notes: {combined_style}\n" if combined_style else "",
    )
    if feedback:
        prompt += (
            f"\n\nPREVIOUS ATTEMPT FEEDBACK — fix these issues in this version:\n{feedback}\n"
        )

    print(f"[writer] Generating script: topic='{topic[:60]}', lang={language}", flush=True)
    messages = [{"role": "user", "content": prompt}]
    full_script = ""
    part_num = 1

    while True:
        text, stop_reason = _call_claude(_GENERATE_SYSTEM, messages)
        full_script += ("\n\n" if full_script else "") + _extract_code_block(text)

        if stop_reason != "max_tokens":
            print(f"[writer] Script done in {part_num} part(s) ({len(full_script)} chars)", flush=True)
            break

        print(f"[writer] Token limit reached, continuing (part {part_num + 1})...", flush=True)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "Continue from where you left off."})
        part_num += 1
        if part_num > 5:
            print("[writer] Warning: 5-part limit reached, stopping.", flush=True)
            break

    # Expand if too short
    full_script = _expand_script(full_script, topic, language)
    return full_script


def _expand_script(script: str, topic: str, language: str) -> str:
    """Loop until script reaches MIN_SCRIPT_LENGTH."""
    MAX_ATTEMPTS = 3

    for attempt in range(MAX_ATTEMPTS):
        if len(script) >= MIN_SCRIPT_LENGTH:
            break

        needed = MIN_SCRIPT_LENGTH - len(script)
        print(
            f"[writer] Expand attempt {attempt + 1}/{MAX_ATTEMPTS}: "
            f"{len(script)} chars, need {needed} more...",
            flush=True,
        )

        user_msg = (
            f"Topic: {topic}\nLanguage: {language}\n"
            f"Target: {MIN_SCRIPT_LENGTH} chars. Current: {len(script)} chars ({needed} short).\n\n"
            f"The script is too short. Read it and write a natural continuation "
            f"that adds depth, examples, and analysis. "
            f"Return ONLY the continuation text in a code block.\n\n"
            f"Current script:\n\n{script}"
        )

        messages = [{"role": "user", "content": user_msg}]
        expansion = ""
        part_num = 1

        while True:
            text, stop_reason = _call_claude(_EXPAND_SYSTEM, messages)
            expansion += ("\n\n" if expansion else "") + _extract_code_block(text)
            if stop_reason != "max_tokens":
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Continue."})
            part_num += 1
            if part_num > 3:
                break

        script = script + "\n\n" + expansion
        print(f"[writer] After expand {attempt + 1}: {len(script)} chars", flush=True)

    if len(script) < MIN_SCRIPT_LENGTH:
        print(
            f"[writer] WARNING: still short after {MAX_ATTEMPTS} attempts ({len(script)} chars)",
            flush=True,
        )
    return script


# ── Metadata generation ───────────────────────────────────────────────────────

def generate_metadata(topic: str, language: str, script: str) -> dict:
    """Generate title, description, tags for the script.
    
    Returns dict with keys: titles, title, description, tags, tags_raw.
    Titles include Ukrainian translations (format: "Title — Переклад").
    Tags block is 490-500 characters total.
    """
    script_preview = script[:4000]
    user_msg = (
        f"Target language: {language}\n"
        f"Topic: {topic}\n\n"
        f"Script (first 4000 chars):\n{script_preview}\n\n"
        f"Generate optimized YouTube metadata following ALL rules from the system prompt.\n"
        f"Use this exact output format:\n\n"
        f"### Optimized Titles:\n"
        f"1. [Title in {language}] — [Ukrainian translation]\n"
        f"2. [Title in {language}] — [Ukrainian translation]\n"
        f"3. [Title in {language}] — [Ukrainian translation]\n"
        f"4. [Title in {language}] — [Ukrainian translation]\n"
        f"5. [Title in {language}] — [Ukrainian translation]\n\n"
        f"### Optimized Description:\n"
        f"[Engaging, SEO-optimized description in {language}]\n\n"
        f"### Optimized Tags:\n"
        f"[Ranked tags in {language}, total 490-500 characters including commas and spaces]"
    )

    print("[writer] Generating metadata...", flush=True)
    text, _ = _call_claude(_METADATA_SYSTEM, [{"role": "user", "content": user_msg}])

    # Parse titles — each line: "Title in language — Ukrainian translation"
    titles = []          # full strings including UA translation (for display)
    titles_main = []     # only the target-language part (for video title)
    titles_m = re.search(
        r"###\s*Optimized Titles:(.*?)###\s*Optimized Description:",
        text, re.DOTALL | re.IGNORECASE,
    )
    if titles_m:
        for line in titles_m.group(1).strip().splitlines():
            m = re.match(r"\d+\.\s+(.+)", line.strip())
            if m:
                full = m.group(1).strip()
                titles.append(full)
                # Split off Ukrainian translation if present (separator: " — ")
                if " — " in full:
                    main_part = full.split(" — ")[0].strip()
                else:
                    main_part = full
                titles_main.append(main_part)

    # Parse description
    description = ""
    desc_m = re.search(
        r"###\s*Optimized Description:(.*?)###\s*Optimized Tags:",
        text, re.DOTALL | re.IGNORECASE,
    )
    if desc_m:
        description = desc_m.group(1).strip()

    # Parse tags
    tags_raw = ""
    tags_m = re.search(r"###\s*Optimized Tags:(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if tags_m:
        tags_raw = tags_m.group(1).strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    tags_len = len(tags_raw)
    print(
        f"[writer] Metadata done — {len(titles)} title options, "
        f"tags={tags_len} chars (target 490-500)",
        flush=True,
    )
    return {
        "titles":       titles,        # full "Title — Переклад" strings
        "titles_main":  titles_main,   # only target-language part
        "title":        titles_main[0] if titles_main else topic,
        "description":  description,
        "tags":         tags,
        "tags_raw":     tags_raw,
    }


# ── Draft management ──────────────────────────────────────────────────────────

def save_draft(topic: str, language: str, script: str,
               style_notes: str = "", metadata: dict = None) -> str:
    """
    Save a draft to disk. Returns draft_id.
    Draft is stored in PROJECTS_DIR/_draft_{draft_id}/
    """
    draft_id  = f"draft_{int(time.time())}"
    draft_dir = os.path.join(config.PROJECTS_DIR, f"_draft_{draft_id}")
    os.makedirs(draft_dir, exist_ok=True)

    with open(os.path.join(draft_dir, "script.txt"), "w", encoding="utf-8") as f:
        f.write(script)

    state = {
        "draft_id":    draft_id,
        "draft_dir":   draft_dir,
        "topic":       topic,
        "language":    language,
        "style_notes": style_notes,
        "created_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if metadata:
        state["metadata"] = metadata

    with open(os.path.join(draft_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"[writer] Draft saved: {draft_id} ({len(script)} chars)", flush=True)
    return draft_id


def load_draft(draft_id: str) -> dict:
    """Load draft state + script. Returns dict with keys: state, script."""
    safe_id   = os.path.basename(draft_id)
    draft_dir = os.path.join(config.PROJECTS_DIR, f"_draft_{safe_id}")

    with open(os.path.join(draft_dir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)

    script_path = os.path.join(draft_dir, "script.txt")
    with open(script_path, encoding="utf-8") as f:
        script = f.read()

    return {"state": state, "script": script}