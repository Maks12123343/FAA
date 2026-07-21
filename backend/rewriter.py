import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend import languages as lang_utils

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


def _call_claude(system: str, messages: list, timeout: int = 300, max_retries: int = 1) -> tuple:
    """Call Pioneer only — no fallback. Rewrite key is dedicated."""
    return api_client.call_pioneer(
        system,
        messages,
        timeout=timeout,
        max_retries=max_retries,
    )


# ── Script rewrite ────────────────────────────────────────────────────────────

NUM_CHUNKS = 3  # Скільки частин на які розбити transcript
_LAST_REWRITTEN_PARTS = []


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
    global _LAST_REWRITTEN_PARTS
    _LAST_REWRITTEN_PARTS = []

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
    _LAST_REWRITTEN_PARTS = list(rewritten_parts)
    print(f"[rewriter] Script done in {len(chunks)} chunk(s) ({len(full_script)} chars)", flush=True)
    return full_script


# Length is relative to the original transcript:
#   - minimum: 0.50x of original
#   - target:  0.55x of original
#   - maximum: 0.60x of original
MIN_LENGTH_RATIO = 0.50
TARGET_LENGTH_RATIO = 0.55
MAX_LENGTH_RATIO = 0.60


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
        f"1. LENGTH: Must be between 50% and 60% of the original, with an ideal target near 55%. "
        f"(original={orig_len} chars, rewritten={script_len} chars = {pct}%, "
        f"allowed range: {min_chars}-{max_chars} chars)\n"
        f"2. COMPLETENESS: Are all key events, facts, and narrative beats preserved?\n"
        f"3. VOICEOVER QUALITY: Does it sound natural when read aloud? "
        f"No heavy sentences, no awkward phrasing?\n"
        f"4. NO REPETITION: Is it free of unnecessary repetition or filler?\n"
        f"5. LANGUAGE: Is it correctly and fluently written in {language}?\n"
        f"6. UNIQUENESS: Is it genuinely rewritten (not just synonymized)?\n\n"
        f"Scoring: 1-10. PASSED if score >= 7 AND length is between 50% and 60% of original.\n\n"
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
                    f"Must be at least {min_chars} chars (50% of original). " + feedback
                )
            elif script_len > max_chars:
                passed   = False
                feedback = (
                    f"Script is too long: {script_len} chars ({pct}% of original {orig_len} chars). "
                    f"Must be at most {max_chars} chars (60% of original). "
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


def _continuity_windows(parts: list, window_chars: int = 900) -> str:
    """Return compact boundary excerpts so the model can inspect chunk joins."""
    windows = []
    for i in range(len(parts) - 1):
        left = parts[i].strip()[-window_chars:]
        right = parts[i + 1].strip()[:window_chars]
        windows.append(
            f"BOUNDARY {i + 1}->{i + 2}\n"
            f"END OF PART {i + 1}:\n{left}\n\n"
            f"START OF PART {i + 2}:\n{right}"
        )
    return "\n\n---\n\n".join(windows)


def _continuity_check_script(script: str, parts: list, language: str) -> tuple:
    """
    Check whether rewritten chunks join naturally. Returns (passed, feedback).
    If the check itself fails, do not block production.
    """
    if len(parts) < 2:
        return True, ""

    system = (
        "You are a strict continuity editor for long YouTube voiceover scripts. "
        "Your job is to inspect joins between rewritten chunks and identify only real problems."
    )
    user_msg = (
        f"Target language: {language}\n\n"
        f"Inspect these chunk boundaries from one already rewritten script.\n"
        f"Check for repeated openings, repeated summaries, abrupt transitions, duplicated facts, "
        f"contradictions between parts, or a middle part that starts like a new video.\n"
        f"Do NOT complain about normal topic continuation.\n\n"
        f"{_continuity_windows(parts)}\n\n"
        f"Reply with JSON only, no markdown:\n"
        f'{{"passed": true, "issues": ["issue1"], "feedback": "short actionable edit instruction"}}'
    )

    try:
        text, _ = _call_claude(system, [{"role": "user", "content": user_msg}], timeout=180)
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group() if m else text)
        passed = bool(data.get("passed", False))
        issues = data.get("issues", [])
        feedback = data.get("feedback", "")
        print(f"[rewriter] Continuity check: passed={passed}, issues={issues}", flush=True)
        return passed, feedback
    except Exception as e:
        print(f"[rewriter] Continuity check error (skipping): {e}", flush=True)
        return True, ""


def _polish_script_continuity(script: str, language: str, min_chars: int, max_chars: int,
                              feedback: str) -> str:
    """
    One full-script polish pass for chunk joins only. Keeps facts and length budget.
    """
    system = (
        "You are a professional continuity editor for YouTube voiceover scripts. "
        "You do not add facts. You only smooth transitions, remove duplicated openings, "
        "remove repeated summaries, and make the script read as one continuous narration."
    )
    user_msg = (
        f"Target language: {language}\n"
        f"Required length: {min_chars}-{max_chars} characters. Ideal target is about "
        f"{int((min_chars + max_chars) / 2)} characters.\n"
        f"Continuity feedback to fix:\n{feedback}\n\n"
        f"Polish the full script below so the chunk joins feel seamless.\n"
        f"Rules:\n"
        f"- Keep the same language.\n"
        f"- Do not add new facts, claims, scenes, numbers, names, or events.\n"
        f"- Do not remove key events.\n"
        f"- Do not make it longer than {max_chars} characters or shorter than {min_chars} characters.\n"
        f"- Preserve a strong opening and a strong final closing.\n"
        f"- Output only the polished script in one code block.\n\n"
        f"SCRIPT:\n{script}"
    )
    text, _ = _call_claude(system, [{"role": "user", "content": user_msg}], timeout=360)
    return _extract_code_block(text)


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


MAX_METADATA_ATTEMPTS = 3


def _call_metadata_part(system: str, label: str, user_msg: str, required_marker: str) -> str:
    last_err = None
    for attempt in range(1, MAX_METADATA_ATTEMPTS + 1):
        try:
            print(f"[rewriter]   -> {label} attempt {attempt}/{MAX_METADATA_ATTEMPTS}...", flush=True)
            raw, _ = _call_claude(
                system,
                [{"role": "user", "content": user_msg}],
                timeout=180,
                max_retries=2,
            )
            if raw and required_marker.lower() in raw.lower():
                return raw
            last_err = f"missing marker {required_marker}"
            print(f"[rewriter]   {label} invalid metadata response ({last_err})", flush=True)
        except Exception as e:
            last_err = str(e)
            print(
                f"[rewriter]   {label} failed attempt {attempt}/{MAX_METADATA_ATTEMPTS}: {e}",
                flush=True,
            )
        if attempt < MAX_METADATA_ATTEMPTS:
            time.sleep(5 * attempt)
    raise RuntimeError(f"Metadata {label} failed after {MAX_METADATA_ATTEMPTS} attempts: {last_err}")


def _rewrite_metadata(
    language: str,
    source_title: str,
    source_description: str,
    source_tags: list,
) -> dict:
    """
    Generate metadata in 3 SEPARATE Opus calls (title / description / tags).
    Splitting keeps each prompt small enough that Opus responds well under
    CloudFront's ~60s upstream timeout — a single combined call was hitting HTTP 504.

    Each call uses the SAME rewrite tone: minimal changes to the competitor's
    text, translated precisely into the video's language, kept SEO-relevant.
    """
    system_full = _load_prompt(METADATA_PROMPT_FILE, language)
    tags_str = ", ".join(source_tags) if source_tags else ""

    # Shared style guidance used in every mini-call.
    style = (
        f"You are rewriting a YouTube video's metadata for a {language} audience.\n"
        f"Rewrite MINIMALLY — preserve the competitor's structure, hooks, and SEO angles.\n"
        f"Translate the text into {language} (natural, native-sounding).\n"
        f"Do NOT invent content. Only rephrase.\n"
    )

    print("[rewriter] Generating metadata (3 separate Opus calls)...", flush=True)

    # ── Call 1: titles (5 options) ────────────────────────────────────────────
    title_user = (
        f"{style}\n"
        f"COMPETITOR'S TITLE:\n{source_title}\n\n"
        f"Produce 5 alternative titles for the same video, rewritten into {language}.\n"
        f"Keep each title very close to the competitor's title: same structure, same order of ideas, same SEO entities, and same curiosity hook.\n"
        f"Do not aggressively shorten. Do not remove endings like 'Then THIS Happened', 'And THIS Happened', or similar hooks; translate them naturally into {language}.\n"
        f"Aim for the target-language title to be under 100 characters when possible. If it is too long, compress only minor filler words, not the main entities or final hook.\n"
        f"Keep capitalized emphasis where it makes sense for YouTube style, and avoid clickbait exaggeration beyond the source.\n\n"
        f"Reply STRICTLY in this format (no other text):\n"
        f"### Optimized Titles:\n"
        f"1. Title in {language} — Ukrainian translation\n"
        f"2. Title in {language} — Ukrainian translation\n"
        f"3. Title in {language} — Ukrainian translation\n"
        f"4. Title in {language} — Ukrainian translation\n"
        f"5. Title in {language} — Ukrainian translation\n"
    )
    print("[rewriter]   → titles...", flush=True)
    titles_raw = _call_metadata_part(system_full, "titles", title_user, "### Optimized Titles:")

    # ── Call 2: description ───────────────────────────────────────────────────
    desc_user = (
        f"{style}\n"
        f"COMPETITOR'S DESCRIPTION:\n{source_description}\n\n"
        f"Rewrite this description into {language}. Keep the same length and structure.\n"
        f"Preserve any hashtags / CTA lines but translate them.\n\n"
        f"Reply STRICTLY in this format (no other text):\n"
        f"### Optimized Description:\n"
        f"<the rewritten description here>\n"
    )
    print("[rewriter]   → description...", flush=True)
    desc_raw = _call_metadata_part(system_full, "description", desc_user, "### Optimized Description:")

    # ── Call 3: tags ──────────────────────────────────────────────────────────
    tags_user = (
        f"{style}\n"
        f"COMPETITOR'S TAGS:\n{tags_str}\n\n"
        f"Rewrite these tags for a {language}-speaking audience. Target ~490-500 chars total.\n"
        f"Mix {language} and universal English tags (names, weapon models). Comma-separated.\n\n"
        f"Reply STRICTLY in this format (no other text):\n"
        f"### Optimized Tags:\n"
        f"tag1, tag2, tag3, ...\n"
    )
    print("[rewriter]   → tags...", flush=True)
    tags_raw_resp = _call_metadata_part(system_full, "tags", tags_user, "### Optimized Tags:")

    # Combine the three fragments into the format _parse_metadata_output expects.
    combined = (
        f"{titles_raw.strip()}\n\n"
        f"{desc_raw.strip()}\n\n"
        f"{tags_raw_resp.strip()}\n"
    )
    result = _parse_metadata_output(combined)
    missing = []
    if not result.get("titles"):
        missing.append("titles")
    if not result.get("description"):
        missing.append("description")
    if not result.get("tags"):
        missing.append("tags")
    if missing:
        raise RuntimeError("Metadata parse missing: " + ", ".join(missing))
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
    language_name = lang_utils.configured_language_name(language)
    script   = ""
    feedback = ""
    orig_len = len(transcript)
    min_chars, max_chars = _length_bounds(orig_len)

    if test_mode:
        print("[rewriter] TEST MODE: using short prompt (~750 words), skipping quality check", flush=True)
        script = _rewrite_script(transcript, language_name, source_title, test_mode=True)
        print(f"[rewriter] TEST MODE: script done ({len(script)} chars)", flush=True)
    else:
        print(
            f"[rewriter] Length target: {min_chars}-{max_chars} chars "
            f"(original={orig_len}, target {int(TARGET_LENGTH_RATIO * 100)}%, "
            f"range {int(MIN_LENGTH_RATIO * 100)}%-{int(MAX_LENGTH_RATIO * 100)}%)",
            flush=True,
        )
        for attempt in range(MAX_REWRITE_ATTEMPTS):
            print(
                f"[rewriter] Rewrite attempt {attempt + 1}/{MAX_REWRITE_ATTEMPTS}"
                + (f" (feedback: {feedback[:80]}...)" if feedback else ""),
                flush=True,
            )
            script = _rewrite_script(transcript, language_name, source_title, feedback=feedback, test_mode=False)

            # Hard cap on length: if model overshot 60%, trim at sentence boundary
            if len(script) > max_chars:
                old_len = len(script)
                script  = _trim_script(script, max_chars)
                print(
                    f"[rewriter] Script trimmed from {old_len} to {len(script)} chars "
                    f"(max allowed: {max_chars})",
                    flush=True,
                )

            parts = list(_LAST_REWRITTEN_PARTS) or [p for p in script.split("\n\n") if p.strip()]
            continuity_ok, continuity_feedback = _continuity_check_script(script, parts, language_name)
            if not continuity_ok:
                print("[rewriter] Continuity polish pass...", flush=True)
                old_len = len(script)
                script = _polish_script_continuity(
                    script=script,
                    language=language_name,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    feedback=continuity_feedback,
                )
                if len(script) > max_chars:
                    script = _trim_script(script, max_chars)
                print(
                    f"[rewriter] Continuity polish done: {old_len} -> {len(script)} chars",
                    flush=True,
                )

            passed, feedback = _quality_check_script(script, transcript, language_name, test_mode=False)
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
        language           = language_name,
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
    return _rewrite_script(transcript, lang_utils.configured_language_name(language), video_title)

def generate_title(script: str, language: str, original_title: str) -> str:
    return original_title

def generate_metadata(script: str, language: str, title: str) -> dict:
    return {"description": "", "tags": []}
# end of module
