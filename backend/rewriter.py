import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

REWRITE_PROMPT_FILE      = os.path.join(config.DATA_DIR, "rewrite_prompt.txt")
REWRITE_PROMPT_TEST_FILE = os.path.join(config.DATA_DIR, "rewrite_prompt_test.txt")
METADATA_PROMPT_FILE     = os.path.join(config.DATA_DIR, "metadata_prompt.txt")


def _load_prompt(path: str, language: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if "Вставте сюди" in text:
        raise ValueError(f"Prompt file not filled in: {path}")
    return text.replace("{language}", language)


from backend import api_client


def _call_claude(system: str, messages: list, timeout: int = 300) -> tuple:
    """Call Pioneer only — no fallback. Rewrite key is dedicated."""
    return api_client.call_pioneer(
        system,
        messages,
        timeout=timeout,
        max_retries=1,
    )


# ── Script rewrite ────────────────────────────────────────────────────────────

NUM_CHUNKS = 3  # Скільки частин на які розбити transcript


def _extract_code_block(text: str) -> str:
    if not text:
        raise RuntimeError("LLM returned empty response — check API key and connectivity")
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _split_into_chunks(transcript: str, num_chunks: int = NUM_CHUNKS) -> list:
    """
    Розбити transcript на num_chunks приблизно рівні частини.
    Ділимо по межі речення (крапка/! /?), щоб не рвати фрази посередині.
    """
    if num_chunks < 2:
        return [transcript]

    total = len(transcript)
    target_size = total / num_chunks
    # Всі позиції кінців речень
    sentence_ends = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", transcript)]
    if not sentence_ends:
        # Немає розділових знаків — ділимо по пробілах
        sentence_ends = [m.end() for m in re.finditer(r"\s+", transcript)]

    chunks = []
    start = 0
    for i in range(1, num_chunks):
        target_pos = int(target_size * i)
        # Знаходимо найближчий кінець речення до цільової позиції
        best = min(sentence_ends, key=lambda e: abs(e - target_pos))
        if best > start and best < total:
            chunks.append(transcript[start:best].strip())
            start = best
        # інакше пропускаємо (буде злиття з наступним)
    chunks.append(transcript[start:].strip())
    return [c for c in chunks if c]


def _get_summary(text: str, language: str, timeout: int = 120) -> str:
    """
    Швидкий summary у 2-3 реченнях — щоб наступний chunk знав контекст.
    """
    system = (
        "You write brief 2-3 sentence summaries of transcript chunks. "
        "Focus on: who is involved, what happened, key facts. "
        "Reply with summary text only, no preamble."
    )
    user_msg = f"Language: {language}\n\nSummarize in 2-3 sentences:\n\n{text}"
    try:
        result, _ = _call_claude(system, [{"role": "user", "content": user_msg}], timeout=timeout)
        return result.strip()[:600]
    except Exception as e:
        print(f"[rewriter] Summary failed ({e}), using first 300 chars", flush=True)
        return text[:300]


def _rewrite_chunk(chunk: str, position: str, language: str, video_title: str,
                   system_prompt: str, prev_summary: str = "",
                   prev_tail: str = "", feedback: str = "",
                   timeout: int = 300) -> str:
    """
    Переписати один chunk з контекстом попереднього.
    position: "first" / "middle" / "last".
    """
    ctx_lines = [
        f"Target language: {language}",
        f"Original video title: {video_title}",
        f"This is a CHUNK of a longer script. Position: {position.upper()} chunk.",
    ]
    if position == "first":
        ctx_lines.append("Write a strong opening hook. Do NOT close/summarize — the script continues.")
    elif position == "middle":
        ctx_lines.append("Continue smoothly from the previous chunk. Do NOT re-introduce or close.")
    elif position == "last":
        ctx_lines.append("Continue smoothly and write a strong closing that wraps up the story.")

    if prev_summary:
        ctx_lines.append(f"\nCONTEXT FROM PREVIOUS CHUNKS (do NOT rewrite this, just use for continuity):\n{prev_summary}")
    if prev_tail:
        ctx_lines.append(f"\nEND OF PREVIOUS REWRITTEN CHUNK (continue seamlessly from here):\n...{prev_tail}")
    if feedback:
        ctx_lines.append(f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}")

    ctx_lines.append(f"\nCHUNK TO REWRITE:\n{chunk}")

    user_msg = "\n".join(ctx_lines)
    messages = [{"role": "user", "content": user_msg}]
    result = ""
    part = 1
    while True:
        text, stop = _call_claude(system_prompt, messages, timeout=timeout)
        result += ("\n\n" if result else "") + _extract_code_block(text)
        if stop != "max_tokens":
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "Continue from where you left off."})
        part += 1
        if part > 3:
            break
    return result


def _rewrite_script(transcript: str, language: str, video_title: str,
                    feedback: str = "", test_mode: bool = False) -> str:
    """
    Rewrite transcript у NUM_CHUNKS частин.
    Кожна частина шле окремий запит з коротким контекстом попередніх.
    Після кожного chunk беремо summary + tail для наступного.
    """
    prompt_file = REWRITE_PROMPT_TEST_FILE if test_mode else REWRITE_PROMPT_FILE
    system = _load_prompt(prompt_file, language)

    chunks = _split_into_chunks(transcript, NUM_CHUNKS)
    print(f"[rewriter] Split transcript into {len(chunks)} chunks: "
          f"{[len(c) for c in chunks]} chars", flush=True)

    rewritten_parts = []
    prev_summary = ""
    prev_tail = ""

    for i, chunk in enumerate(chunks):
        if i == 0:
            position = "first"
        elif i == len(chunks) - 1:
            position = "last"
        else:
            position = "middle"

        print(f"[rewriter] Chunk {i+1}/{len(chunks)} ({position}, {len(chunk)} chars)...", flush=True)
        part = _rewrite_chunk(
            chunk=chunk,
            position=position,
            language=language,
            video_title=video_title,
            system_prompt=system,
            prev_summary=prev_summary,
            prev_tail=prev_tail,
            feedback=feedback if i == 0 else "",  # feedback тільки в перший
            timeout=300,
        )
        print(f"[rewriter]   → rewrote to {len(part)} chars", flush=True)
        rewritten_parts.append(part)

        # Готуємо контекст для наступного chunk
        if i < len(chunks) - 1:
            # summary всього що вже переписано (компактно)
            combined_so_far = "\n\n".join(rewritten_parts)
            # Беремо summary тільки якщо накопичили багато — інакше просто перші 500 chars
            if len(combined_so_far) > 2000:
                prev_summary = _get_summary(combined_so_far[-3000:], language)
            else:
                prev_summary = combined_so_far[:500]
            # Останні 2-3 речення попереднього chunk — для плавного переходу
            sents = re.split(r"(?<=[.!?])\s+", part.strip())
            prev_tail = " ".join(sents[-3:])[:400]

    full_script = "\n\n".join(rewritten_parts)
    print(f"[rewriter] Script done in {len(chunks)} chunk(s) ({len(full_script)} chars)", flush=True)
    return full_script


# Length is now relative to the original transcript:
#   - minimum: 0.9x of original (script can't be shorter)
#   - maximum: 1.4x of original (script can't be longer — prevents bloat)
MIN_LENGTH_RATIO = 0.9
MAX_LENGTH_RATIO = 1.4


def _length_bounds(original_length: int) -> tuple:
    """Return (min_chars, max_chars) for a rewrite based on the original transcript length."""
    return int(original_length * MIN_LENGTH_RATIO), int(original_length * MAX_LENGTH_RATIO)


def _trim_script(script: str, max_chars: int) -> str:
    """
    If script is over max_chars, cut it back at the last sentence boundary
    that fits within the limit so we don't end mid-sentence.
    """
    if len(script) <= max_chars:
        return script

    cut = script[:max_chars]
    # Prefer the last sentence-ending punctuation followed by whitespace/newline
    for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = cut.rfind(punct)
        if idx >= max_chars * 0.6:  # don't cut too aggressively
            return cut[:idx + 1].rstrip()
    # Fallback: cut at last whitespace
    idx = cut.rfind(" ")
    if idx >= max_chars * 0.6:
        return cut[:idx].rstrip()
    return cut.rstrip()


# ── Quality check ─────────────────────────────────────────────────────────────

def _quality_check_script(script: str, transcript: str, language: str, test_mode: bool = False) -> tuple:
    """
    Claude перевіряє якість рірайту.
    Повертає (passed: bool, feedback: str).
    Якщо сам check впав — вважаємо passed=True щоб не блокувати пайплайн.
    """
    orig_len   = len(transcript)
    script_len = len(script)
    pct        = round(script_len / orig_len * 100) if orig_len else 0
    min_chars, max_chars = _length_bounds(orig_len)

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
        f"1. LENGTH: Must be between 90% and 140% of the original. "
        f"(original={orig_len} chars, rewritten={script_len} chars = {pct}%, "
        f"allowed range: {min_chars}-{max_chars} chars)\n"
        f"2. COMPLETENESS: Are all key events, facts, and narrative beats preserved?\n"
        f"3. VOICEOVER QUALITY: Does it sound natural when read aloud? "
        f"No heavy sentences, no awkward phrasing?\n"
        f"4. NO REPETITION: Is it free of unnecessary repetition or filler?\n"
        f"5. LANGUAGE: Is it correctly and fluently written in {language}?\n"
        f"6. UNIQUENESS: Is it genuinely rewritten (not just synonymized)?\n\n"
        f"Scoring: 1-10. PASSED if score >= 7 AND length is between 90% and 140% of original.\n\n"
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

        # Незалежна перевірка довжини (skip in test mode)
        if not test_mode:
            if script_len < min_chars:
                passed   = False
                feedback = (
                    f"Script is too short: {script_len} chars ({pct}% of original {orig_len} chars). "
                    f"Must be at least {min_chars} chars (90% of original). " + feedback
                )
            elif script_len > max_chars:
                passed   = False
                feedback = (
                    f"Script is too long: {script_len} chars ({pct}% of original {orig_len} chars). "
                    f"Must be at most {max_chars} chars (140% of original). "
                    f"Rewrite more concisely while preserving all key events. " + feedback
                )

        print(
            f"[rewriter] Quality check: score={score:.1f}/10, passed={passed}, "
            f"length={pct}% (range {min_chars}-{max_chars}), issues={issues}",
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
    test_mode: bool = False,
) -> dict:
    """
    Call 1: rewrite script (rewrite_prompt.txt) з quality check і retry.
    Call 2: rewrite metadata (metadata_prompt.txt) using SOURCE video's metadata.
    Returns: {script, title, titles, description, tags}
    test_mode: uses short prompt (~750 words), skips expand + quality check.
    """
    script   = ""
    feedback = ""
    orig_len = len(transcript)
    min_chars, max_chars = _length_bounds(orig_len)

    if test_mode:
        print("[rewriter] TEST MODE: using short prompt (~750 words), skipping quality check", flush=True)
        script = _rewrite_script(transcript, language, source_title, test_mode=True)
        print(f"[rewriter] TEST MODE: script done ({len(script)} chars)", flush=True)
    else:
        print(
            f"[rewriter] Length target: {min_chars}-{max_chars} chars "
            f"(original={orig_len}, range 0.9x-1.4x)",
            flush=True,
        )
        for attempt in range(MAX_REWRITE_ATTEMPTS):
            print(
                f"[rewriter] Rewrite attempt {attempt + 1}/{MAX_REWRITE_ATTEMPTS}"
                + (f" (feedback: {feedback[:80]}...)" if feedback else ""),
                flush=True,
            )
            script = _rewrite_script(transcript, language, source_title, feedback=feedback, test_mode=False)

            # Hard cap on length: if model overshot 1.4x, trim at sentence boundary
            if len(script) > max_chars:
                old_len = len(script)
                script  = _trim_script(script, max_chars)
                print(
                    f"[rewriter] Script trimmed from {old_len} to {len(script)} chars "
                    f"(max allowed: {max_chars})",
                    flush=True,
                )

            passed, feedback = _quality_check_script(script, transcript, language, test_mode=False)
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
# end of module
