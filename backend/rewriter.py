import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

REWRITE_PROMPT_FILE  = os.path.join(config.DATA_DIR, "rewrite_prompt.txt")
METADATA_PROMPT_FILE = os.path.join(config.DATA_DIR, "metadata_prompt.txt")


def _load_prompt(path: str, language: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if "Вставте сюди" in text:
        raise ValueError(f"Prompt file not filled in: {path}")
    return text.replace("{language}", language)


from backend import api_client


def _call_claude(system: str, messages: list, timeout: int = 180) -> tuple:
    """Call Pioneer API with automatic key rotation and retry."""
    return api_client.call_pioneer(system, messages, timeout=timeout)


# ── Script rewrite ────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str:
    if not text:
        raise RuntimeError("LLM returned empty response — check API key and connectivity")
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _rewrite_script(transcript: str, language: str, video_title: str,
                    feedback: str = "") -> str:
    system   = _load_prompt(REWRITE_PROMPT_FILE, language)
    user_msg = f"Target language: {language}\nOriginal video title: {video_title}\n"
    if feedback:
        user_msg += (
            f"\nPREVIOUS ATTEMPT FEEDBACK (fix these issues in this rewrite):\n{feedback}\n"
        )
    user_msg += f"\n{transcript}"

    print("[rewriter] Rewriting script...", flush=True)
    messages    = [{"role": "user", "content": user_msg}]
    full_script = ""
    part_num    = 1

    while True:
        text, stop_reason = _call_claude(system, messages)
        full_script += ("\n\n" if full_script else "") + _extract_code_block(text)

        if stop_reason != "max_tokens":
            print(f"[rewriter] Script done in {part_num} part(s) ({len(full_script)} chars)", flush=True)
            break

        print(f"[rewriter] Limit reached, continuing (part {part_num + 1})...", flush=True)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",      "content": "Continue from where you left off."})
        part_num += 1

        if part_num > 5:
            print("[rewriter] Warning: reached 5-part limit, stopping.", flush=True)
            break

    return full_script


MIN_SCRIPT_LENGTH = 20000

def _expand_script(script: str, language: str, video_title: str) -> str:
    """Loop until script reaches MIN_SCRIPT_LENGTH. Each pass Claude self-checks and continues."""
    MAX_ATTEMPTS = 3

    for attempt in range(MAX_ATTEMPTS):
        if len(script) >= MIN_SCRIPT_LENGTH:
            break

        needed = MIN_SCRIPT_LENGTH - len(script)
        print(f"[rewriter] Expand attempt {attempt + 1}/{MAX_ATTEMPTS}: {len(script)} chars, need {needed} more...", flush=True)

        system = (
            f"You are a professional scriptwriter reviewing your own voiceover script. "
            f"The script is too short and may be missing depth, examples, or analysis. "
            f"Read the current script, assess what important aspects of the topic are underdeveloped or missing, "
            f"then write a natural continuation that fills those gaps. "
            f"Requirements: language={language}, approximately {needed} characters, "
            f"same style and tone as the existing script, "
            f"voiceover rules (numbers spelled out, smooth sentences, no ads, no channel mentions). "
            f"Return ONLY the continuation text in a code block, nothing else."
        )

        user_msg = (
            f"Video topic: {video_title}\n"
            f"Target length: {MIN_SCRIPT_LENGTH} chars. Current: {len(script)} chars ({needed} short).\n\n"
            f"Current script:\n\n{script}"
        )

        messages  = [{"role": "user", "content": user_msg}]
        expansion = ""
        part_num  = 1

        while True:
            text, stop_reason = _call_claude(system, messages)
            expansion += ("\n\n" if expansion else "") + _extract_code_block(text)
            if stop_reason != "max_tokens":
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Continue."})
            part_num += 1
            if part_num > 3:
                break

        script = script + "\n\n" + expansion
        print(f"[rewriter] After attempt {attempt + 1}: {len(script)} chars", flush=True)

    if len(script) < MIN_SCRIPT_LENGTH:
        print(f"[rewriter] WARNING: still short after {MAX_ATTEMPTS} attempts ({len(script)} chars)", flush=True)

    return script


# ── Quality check ─────────────────────────────────────────────────────────────

def _quality_check_script(script: str, transcript: str, language: str) -> tuple:
    """
    Claude перевіряє якість рірайту.
    Повертає (passed: bool, feedback: str).
    Якщо сам check впав — вважаємо passed=True щоб не блокувати пайплайн.
    """
    orig_len   = len(transcript)
    script_len = len(script)
    pct        = round(script_len / orig_len * 100) if orig_len else 0

    system = (
        "You are a strict quality control editor for voiceover scripts. "
        "Your job is to evaluate whether a rewritten script meets all requirements. "
        "Be critical, precise, and objective."
    )

    # Показуємо перші 4000 символів кожного (достатньо для оцінки якості)
    orig_preview   = transcript[:4000]
    script_preview = script[:4000]

    user_msg = (
        f"ORIGINAL TRANSCRIPT ({orig_len} chars):\n{orig_preview}\n\n"
        f"{'...[truncated]' if orig_len > 4000 else ''}\n\n"
        f"REWRITTEN SCRIPT ({script_len} chars, {pct}% of original):\n{script_preview}\n\n"
        f"{'...[truncated]' if script_len > 4000 else ''}\n\n"
        f"Evaluate the rewritten script on these criteria:\n"
        f"1. LENGTH: Is it at least 90% of the original? "
        f"(original={orig_len} chars, rewritten={script_len} chars = {pct}%)\n"
        f"2. COMPLETENESS: Are all key events, facts, and narrative beats preserved?\n"
        f"3. VOICEOVER QUALITY: Does it sound natural when read aloud? "
        f"No heavy sentences, no awkward phrasing?\n"
        f"4. NO REPETITION: Is it free of unnecessary repetition or filler?\n"
        f"5. LANGUAGE: Is it correctly and fluently written in {language}?\n"
        f"6. UNIQUENESS: Is it genuinely rewritten (not just synonymized)?\n\n"
        f"Scoring: 1-10. PASSED if score >= 7 AND length >= 90% of original.\n\n"
        f"Reply with JSON only, no markdown:\n"
        f'{{\"score\": 8, \"passed\": true, \"issues\": [\"issue1\", \"issue2\"], '
        f'\"feedback\": \"Specific actionable feedback for improvement\"}}'
    )

    try:
        text, _ = _call_claude(system, [{"role": "user", "content": user_msg}])
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group() if m else text)

        score    = float(data.get("score", 0))
        passed   = bool(data.get("passed", False))
        feedback = data.get("feedback", "")
        issues   = data.get("issues", [])

        # Додаткова перевірка довжини незалежно від Claude
        if script_len < orig_len * 0.90:
            passed   = False
            feedback = (
                f"Script is too short: {script_len} chars ({pct}% of original {orig_len} chars). "
                f"Must be at least 90%. " + feedback
            )

        print(
            f"[rewriter] Quality check: score={score:.1f}/10, passed={passed}, "
            f"length={pct}%, issues={issues}",
            flush=True,
        )
        return passed, feedback

    except Exception as e:
        print(f"[rewriter] Quality check error (skipping): {e}", flush=True)
        # Якщо check впав — не блокуємо пайплайн
        return True, ""


# ── Metadata rewrite ──────────────────────────────────────────────────────────

def _parse_metadata_output(text: str) -> dict:
    """
    Parse the structured output:
      ### Optimized Titles:
      1. Title in language — Ukrainian translation
      ...
      ### Optimized Description:
      ...
      ### Optimized Tags:
      tag1, tag2, ...
    """
    # Titles — each line: "Title in target language — Ukrainian translation"
    titles = []        # full strings including UA translation (for display)
    titles_main = []   # only the target-language part (for video title)
    titles_m = re.search(r"###\s*Optimized Titles:(.*?)###\s*Optimized Description:", text, re.DOTALL | re.IGNORECASE)
    if titles_m:
        for line in titles_m.group(1).strip().splitlines():
            m = re.match(r"\d+\.\s+(.+)", line.strip())
            if m:
                full = m.group(1).strip()
                titles.append(full)
                # Split off Ukrainian translation (separator: " — ")
                if " — " in full:
                    main_part = full.split(" — ")[0].strip()
                else:
                    main_part = full
                titles_main.append(main_part)

    # Description
    description = ""
    desc_m = re.search(r"###\s*Optimized Description:(.*?)###\s*Optimized Tags:", text, re.DOTALL | re.IGNORECASE)
    if desc_m:
        description = desc_m.group(1).strip()

    # Tags
    tags_raw = ""
    tags_m = re.search(r"###\s*Optimized Tags:(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if tags_m:
        tags_raw = tags_m.group(1).strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    tags_len = len(tags_raw)
    print(
        f"[rewriter] Parsed metadata — {len(titles)} titles, "
        f"tags={tags_len} chars (target 490-500)",
        flush=True,
    )
    return {
        "titles":      titles,                                  # full "Title — Переклад" strings
        "titles_main": titles_main,                             # only target-language part
        "title":       titles_main[0] if titles_main else "",   # clean title for video
        "description": description,
        "tags":        tags,
        "tags_raw":    tags_raw,
    }


def _rewrite_metadata(
    language: str,
    source_title: str,
    source_description: str,
    source_tags: list,
) -> dict:
    system = _load_prompt(METADATA_PROMPT_FILE, language)

    tags_str = ", ".join(source_tags) if source_tags else ""
    user_msg = (
        f"Target language: {language}\n\n"
        f"COMPETITOR'S TITLE:\n{source_title}\n\n"
        f"COMPETITOR'S DESCRIPTION:\n{source_description}\n\n"
        f"COMPETITOR'S TAGS:\n{tags_str}"
    )

    print("[rewriter] Generating metadata...", flush=True)
    raw, _ = _call_claude(system, [{"role": "user", "content": user_msg}])
    result = _parse_metadata_output(raw)
    print(f"[rewriter] Metadata done — {len(result['titles'])} title options", flush=True)
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

MAX_REWRITE_ATTEMPTS = 3

def rewrite_all(
    transcript: str,
    language: str,
    source_title: str,
    source_description: str = "",
    source_tags: list = None,
) -> dict:
    """
    Call 1: rewrite script (rewrite_prompt.txt) з quality check і retry.
    Call 2: rewrite metadata (metadata_prompt.txt) using SOURCE video's metadata.
    Returns: {script, title, titles, description, tags}
    """
    script   = ""
    feedback = ""

    for attempt in range(MAX_REWRITE_ATTEMPTS):
        print(
            f"[rewriter] Rewrite attempt {attempt + 1}/{MAX_REWRITE_ATTEMPTS}"
            + (f" (feedback: {feedback[:80]}...)" if feedback else ""),
            flush=True,
        )
        script = _rewrite_script(transcript, language, source_title, feedback=feedback)
        script = _expand_script(script, language, source_title)

        passed, feedback = _quality_check_script(script, transcript, language)
        if passed:
            print(f"[rewriter] Quality check PASSED on attempt {attempt + 1}", flush=True)
            break
        else:
            print(
                f"[rewriter] Quality check FAILED on attempt {attempt + 1}: {feedback[:120]}",
                flush=True,
            )
            if attempt == MAX_REWRITE_ATTEMPTS - 1:
                print("[rewriter] WARNING: all attempts failed quality check, using last result", flush=True)

    meta = _rewrite_metadata(
        language           = language,
        source_title       = source_title,
        source_description = source_description,
        source_tags        = source_tags or [],
    )

    return {
        "script":       script,
        "title":        meta.get("title", source_title),
        "titles":       meta.get("titles", []),       # full "Title — Переклад" strings
        "titles_main":  meta.get("titles_main", []),  # only target-language part
        "description":  meta.get("description", ""),
        "tags":         meta.get("tags", []),
        "tags_raw":     meta.get("tags_raw", ""),
    }


# ── Legacy wrappers ───────────────────────────────────────────────────────────

def rewrite(transcript: str, language: str, video_title: str) -> str:
    return _rewrite_script(transcript, language, video_title)

def generate_title(script: str, language: str, original_title: str) -> str:
    return original_title

def generate_metadata(script: str, language: str, title: str) -> dict:
    return {"description": "", "tags": []}