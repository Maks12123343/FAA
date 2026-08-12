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


def _call_claude(system: str, messages: list, timeout: int = 300, max_retries: int = 3) -> tuple:
    """Call the rewrite model."""
    return api_client.call_rewrite_api(
        system,
        messages,
        timeout=timeout,
        max_retries=max(3, max_retries),
    )


# ── Script rewrite ────────────────────────────────────────────────────────────

def _rewrite_chunk_count() -> int:
    try:
        settings = config.load_settings()
        raw = settings.get("rewrite_chunks") or os.environ.get("FAA_REWRITE_CHUNKS") or 6
        return max(1, min(10, int(raw)))
    except (TypeError, ValueError):
        return 6


NUM_CHUNKS = 6
_LAST_REWRITTEN_PARTS = []


def _extract_code_block(text: str) -> str:
    if not text:
        raise RuntimeError("LLM returned empty response — check API key and connectivity")
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _word_bounds(source_words: int) -> tuple:
    return int(source_words * MIN_LENGTH_RATIO), int(source_words * MAX_LENGTH_RATIO)


def _split_into_chunks(transcript: str, num_chunks: int = NUM_CHUNKS) -> list:
    """
    Розбити transcript на num_chunks приблизно рівні частини.
    Ділимо по межі речення (крапка/! /?), щоб не рвати фрази посередині.
    """
    num_chunks = max(1, int(num_chunks or 1))
    if num_chunks == 1:
        return [transcript.strip()]

    total = len(transcript)
    if total == 0:
        return [""]
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
        remaining_chunks = num_chunks - i
        min_pos = start + 1
        max_pos = total - remaining_chunks
        candidates = [e for e in sentence_ends if min_pos <= e <= max_pos]
        if candidates:
            best = min(candidates, key=lambda e: abs(e - target_pos))
        else:
            best = max(min_pos, min(max_pos, target_pos))
        chunks.append(transcript[start:best].strip())
        start = best
    chunks.append(transcript[start:].strip())
    return chunks


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
                   timeout: int = 300, total_len: int = 0,
                   total_min: int = 0, total_max: int = 0,
                   total_target: int = 0) -> str:
    """
    Переписати один chunk з контекстом попереднього.
    position: "first" / "middle" / "last".
    """
    chunk_len = len(chunk)
    chunk_words = _word_count(chunk)
    chunk_min = max(400, int(chunk_len * MIN_LENGTH_RATIO))
    chunk_max = max(chunk_min + 200, int(chunk_len * MAX_LENGTH_RATIO))
    chunk_target = max(chunk_min, int(chunk_len * TARGET_LENGTH_RATIO))
    chunk_min_words, chunk_max_words = _word_bounds(chunk_words)
    chunk_target_words = max(chunk_min_words, int(chunk_words * TARGET_LENGTH_RATIO))
    strict_system = (
        system_prompt
        + "\n\nCRITICAL NUMERIC LENGTH CONTRACT FOR THIS REQUEST:\n"
        + (
            f"- Full source transcript length: {total_len} characters.\n"
            f"- Required full script length: {total_min}-{total_max} characters.\n"
            f"- Ideal full script target: {total_target} characters.\n"
            if total_len else ""
        )
        + f"- Source chunk length: {chunk_len} characters.\n"
        + f"- Source chunk word count: {chunk_words} words.\n"
        + f"- Required rewritten chunk length: {chunk_min}-{chunk_max} characters.\n"
        + f"- Ideal target: {chunk_target} characters.\n"
        + f"- Required rewritten chunk word count: {chunk_min_words}-{chunk_max_words} words.\n"
        + f"- Ideal word target: {chunk_target_words} words.\n"
        + "- Draft by word count first; use the character maximum as the hard safety limit.\n"
        + f"- The rewritten chunk MUST be shorter than the source chunk and close to the target.\n"
        + f"- Output above {chunk_max} characters is invalid. Do not rely on a later editor to fix length.\n"
        + "- Output only one code block containing the rewritten script. No text before or after the code block.\n"
        + "- If any generic percentage rule conflicts with these exact numbers, follow these exact numbers.\n"
    )
    ctx_lines = [
        "CRITICAL NUMERIC LENGTH CONTRACT",
        f"Full source transcript length: {total_len} characters." if total_len else "",
        f"Required full script length: {total_min}-{total_max} characters." if total_len else "",
        f"Ideal full script target: {total_target} characters." if total_len else "",
        f"Source chunk length: {chunk_len} characters.",
        f"Source chunk word count: {chunk_words} words.",
        f"Required rewritten chunk length: {chunk_min}-{chunk_max} characters.",
        f"Ideal target: {chunk_target} characters.",
        f"Required rewritten chunk word count: {chunk_min_words}-{chunk_max_words} words.",
        f"Ideal word target: {chunk_target_words} words.",
        "Draft by word count first, then tighten enough to stay below the character maximum.",
        f"Hard maximum: {chunk_max} characters. The answer fails if it is longer.",
        "Before finalizing, estimate the character count and tighten the text until it fits this range.",
        "Do not preserve every sentence. Preserve the story logic and key events, but remove secondary description, repeated setup, and slow explanations.",
        "Return only the rewritten script in one code block. Do not write status notes after the code block.",
        "",
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
    elif position == "single":
        ctx_lines.append("This is the ONLY chunk. Rewrite the complete script with a strong opening and a natural final closing.")

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
        text, stop = _call_claude(strict_system, messages, timeout=timeout)
        result += ("\n\n" if result else "") + _extract_code_block(text)
        if stop != "max_tokens":
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                "Continue from where you left off, but keep the TOTAL rewritten chunk "
                f"within {chunk_min}-{chunk_max} characters."
            ),
        })
        part += 1
        if part > 3:
            break
    if len(result) > chunk_max:
        chunk_soft_max = _soft_max_from_hard(chunk_max)
        if len(result) <= chunk_soft_max:
            print(
                f"[rewriter] Chunk slightly over hard max "
                f"({len(result)}>{chunk_max}, soft max {chunk_soft_max}); keeping coherent text",
                flush=True,
            )
            return result
        print(
            f"[rewriter] Chunk over length ({len(result)}>{chunk_max}); correcting before next chunk",
            flush=True,
        )
        result = _compress_script_to_bounds(
            script=result,
            transcript=chunk,
            language=language,
            min_chars=chunk_min,
            max_chars=chunk_max,
            feedback=feedback,
        )
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
    orig_len = len(transcript)
    min_chars, max_chars = _length_bounds(orig_len)

    prompt_file = REWRITE_PROMPT_TEST_FILE if test_mode else REWRITE_PROMPT_FILE
    system = _load_prompt(prompt_file, language)

    chunk_count = _rewrite_chunk_count()
    chunks = _split_into_chunks(transcript, chunk_count)
    print(
        f"[rewriter] Split transcript into {len(chunks)} chunks "
        f"(requested {chunk_count}): {[len(c) for c in chunks]} chars",
        flush=True,
    )

    rewritten_parts = []
    prev_summary = ""
    prev_tail = ""

    for i, chunk in enumerate(chunks):
        if len(chunks) == 1:
            position = "single"
        elif i == 0:
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
            total_len=orig_len,
            total_min=min_chars,
            total_max=max_chars,
            total_target=int(orig_len * TARGET_LENGTH_RATIO),
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
#   - minimum: 0.70x of original
#   - target:  0.75x of original
#   - maximum: 0.80x of original
MIN_LENGTH_RATIO = 0.70
TARGET_LENGTH_RATIO = 0.75
MAX_LENGTH_RATIO = 0.80
SOFT_MAX_LENGTH_RATIO = 0.85


def _soft_max_chars(original_length: int) -> int:
    """Quality-first ceiling: accept slightly longer scripts instead of cutting logic."""
    return int(original_length * SOFT_MAX_LENGTH_RATIO)


def _soft_max_from_hard(max_chars: int) -> int:
    return int(max_chars * SOFT_MAX_LENGTH_RATIO / MAX_LENGTH_RATIO)


def _length_bounds(original_length: int) -> tuple:
    """Return (min_chars, max_chars) for a rewrite based on the original transcript length."""
    return int(original_length * MIN_LENGTH_RATIO), int(original_length * MAX_LENGTH_RATIO)


def _compress_script_to_bounds(
    script: str,
    transcript: str,
    language: str,
    min_chars: int,
    max_chars: int,
    feedback: str = "",
) -> str:
    """
    Rewrite an overlong script down to the required length window.

    This is intentionally not a mechanical trim: the model must compress the
    whole narration, preserve the ending, and keep the source facts intact.
    """
    orig_len = len(transcript)
    target_chars = int(orig_len * TARGET_LENGTH_RATIO)
    soft_max = _soft_max_chars(orig_len)
    orig_words = _word_count(transcript)
    script_words = _word_count(script)
    min_words, max_words = _word_bounds(orig_words)
    target_words = max(min_words, int(orig_words * TARGET_LENGTH_RATIO))
    reduction_needed = max(0, len(script) - target_chars)
    reduction_pct = round(reduction_needed / max(len(script), 1) * 100)
    system = (
        "You are a senior voiceover editor. You compress completed scripts "
        "without cutting off the ending, without adding facts, and without "
        "changing the meaning. You obey numeric character limits exactly."
    )
    user_msg = (
        f"Target language: {language}\n"
        f"Original transcript length: {orig_len} characters.\n"
        f"Original transcript word count: {orig_words} words.\n"
        f"Required final script length: {min_chars}-{max_chars} characters.\n"
        f"Ideal target: {target_chars} characters.\n"
        f"Required final word count: {min_words}-{max_words} words.\n"
        f"Ideal word target: {target_words} words.\n"
        f"The current rewritten script is too long: {len(script)} characters, {script_words} words.\n"
        f"You must remove about {reduction_needed} characters ({reduction_pct}% of the current script).\n\n"
        f"Compress the WHOLE script to about {target_words} words / {target_chars} characters, "
        f"with hard range {min_chars}-{max_chars} characters.\n"
        f"Use the word target while editing; use the character maximum as the hard safety limit.\n"
        f"If your answer is longer than {max_chars} characters, it is invalid.\n\n"
        f"Rules:\n"
        f"- Do NOT simply delete the ending.\n"
        f"- Preserve the opening hook and the final closing thought.\n"
        f"- Preserve all key events, names, numbers, cause-and-effect, and narrative beats.\n"
        f"- Remove filler, repeated explanations, slow setup, and non-essential wording.\n"
        f"- Combine adjacent sentences and remove secondary descriptions before removing key events.\n"
        f"- Do not add new facts, new claims, or new scenes.\n"
        f"- Keep the result natural for voiceover.\n"
        f"- Output only the compressed script in one code block. No text before or after it.\n"
    )
    if feedback:
        user_msg += f"\nPrevious quality feedback to respect:\n{feedback}\n"
    user_msg += f"\nSCRIPT TO COMPRESS:\n{script}"

    try:
        text, _ = _call_claude(system, [{"role": "user", "content": user_msg}], timeout=360)
        compressed = _extract_code_block(text)
        if min_chars <= len(compressed) <= max_chars:
            return compressed
        if min_chars <= len(compressed) <= soft_max:
            print(
                f"[rewriter] Compression slightly over hard range "
                f"({len(compressed)} not in {min_chars}-{max_chars}, soft max {soft_max}); "
                "keeping coherent LLM-compressed text",
                flush=True,
            )
            return compressed
        if len(compressed) < min_chars:
            print(
                f"[rewriter] Compression too short ({len(compressed)}<{min_chars}); "
                "keeping previous coherent text",
                flush=True,
            )
            return script
        if len(compressed) < len(script):
            print(
                f"[rewriter] Compression still over soft max ({len(compressed)}>{soft_max}); "
                "keeping reduced LLM text without local sentence cuts",
                flush=True,
            )
            return compressed
        print(
            f"[rewriter] Compression did not improve length ({len(compressed)} chars); "
            "keeping previous coherent text",
            flush=True,
        )
        return script
    except Exception as e:
        print(f"[rewriter] Compression API failed ({e}); keeping coherent overlength text", flush=True)
        return script


def _compress_until_in_bounds(
    script: str,
    transcript: str,
    language: str,
    min_chars: int,
    max_chars: int,
    feedback: str = "",
    max_attempts: int = 1,
) -> str:
    """Try LLM compression; never remove sentences locally just to hit a number."""
    for attempt in range(1, max_attempts + 1):
        if len(script) <= max_chars:
            break
        old_len = len(script)
        print(
            f"[rewriter] Script over length ({old_len}>{max_chars}); "
            f"compression pass {attempt}/{max_attempts}",
            flush=True,
        )
        script = _compress_script_to_bounds(
            script=script,
            transcript=transcript,
            language=language,
            min_chars=min_chars,
            max_chars=max_chars,
            feedback=feedback,
        )
        print(f"[rewriter] Compression pass {attempt}: {old_len} -> {len(script)} chars", flush=True)
        if len(script) > max_chars:
            print(
                f"[rewriter] Compression still over hard max ({len(script)}>{max_chars}); "
                "keeping coherent text and relying on full-script soft limit",
                flush=True,
            )
            break
    return script


def _compress_parts_to_bounds(
    parts: list,
    transcript: str,
    language: str,
    min_chars: int,
    max_chars: int,
    feedback: str = "",
) -> tuple:
    """Compress rewritten chunks separately so the API does not receive one huge prompt."""
    clean_parts = [p.strip() for p in parts if p and p.strip()]
    if not clean_parts:
        return "", []

    current_total = sum(len(p) for p in clean_parts)
    target_total = int(len(transcript) * TARGET_LENGTH_RATIO)
    compressed = []

    system = (
        "You are a senior voiceover editor. You compress one part of a longer "
        "script while preserving facts, logic, and natural narration."
    )

    for idx, part in enumerate(clean_parts, start=1):
        share = len(part) / max(current_total, 1)
        part_target = max(600, int(target_total * share))
        part_min = max(400, int(min_chars * share))
        part_max = max(part_min + 200, int(max_chars * share))
        part_soft_max = _soft_max_from_hard(part_max)
        position = "first" if idx == 1 else "last" if idx == len(clean_parts) else "middle"
        user_msg = (
            f"Target language: {language}\n"
            f"This is part {idx}/{len(clean_parts)} of one longer voiceover script. "
            f"Position: {position.upper()}.\n"
            f"Compress this part to about {part_target} characters. "
            f"Hard range for this part: {part_min}-{part_max} characters.\n\n"
            f"Rules:\n"
            f"- Do not simply delete the ending of this part.\n"
            f"- Preserve all key events, names, numbers, cause-and-effect, and narrative beats.\n"
            f"- Remove filler, repeated explanations, slow setup, and non-essential wording.\n"
            f"- Do not add new facts, new claims, or new scenes.\n"
            f"- Keep the part natural for voiceover.\n"
            f"- If this is the first part, preserve the opening hook.\n"
            f"- If this is the last part, preserve the final closing thought.\n"
            f"- Output only the compressed part in one code block.\n"
        )
        if feedback:
            user_msg += f"\nPrevious quality feedback to respect:\n{feedback}\n"
        user_msg += f"\nPART TO COMPRESS:\n{part}"

        print(
            f"[rewriter] Compressing part {idx}/{len(clean_parts)} "
            f"({len(part)} chars -> target {part_target})",
            flush=True,
        )
        try:
            text, _ = _call_claude(system, [{"role": "user", "content": user_msg}], timeout=300)
            compressed_part = _extract_code_block(text)
            if part_min <= len(compressed_part) <= part_max:
                pass
            elif part_min <= len(compressed_part) <= part_soft_max:
                print(
                    f"[rewriter] Part {idx}/{len(clean_parts)} slightly over hard range "
                    f"({len(compressed_part)} not in {part_min}-{part_max}, soft max {part_soft_max}); "
                    "keeping coherent LLM-compressed part",
                    flush=True,
                )
            elif len(compressed_part) < part_min:
                print(
                    f"[rewriter] Part {idx}/{len(clean_parts)} compression too short "
                    f"({len(compressed_part)}<{part_min}); keeping original part",
                    flush=True,
                )
                compressed_part = part
            elif len(compressed_part) < len(part):
                print(
                    f"[rewriter] Part {idx}/{len(clean_parts)} still over soft max "
                    f"({len(compressed_part)}>{part_soft_max}); keeping reduced LLM part",
                    flush=True,
                )
            else:
                print(
                    f"[rewriter] Part {idx}/{len(clean_parts)} compression did not improve length; "
                    "keeping original part",
                    flush=True,
                )
                compressed_part = part
        except Exception as e:
            print(
                f"[rewriter] Compressing part {idx}/{len(clean_parts)} failed ({e}); "
                "keeping original coherent part",
                flush=True,
            )
            compressed_part = part
        compressed.append(compressed_part)

    script = "\n\n".join(compressed).strip()
    print(f"[rewriter] Chunk compression total: {current_total} -> {len(script)} chars", flush=True)
    return script, compressed


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
        f"1. LENGTH: Must be between 70% and 80% of the original, with an ideal target near 75%. "
        f"(original={orig_len} chars, rewritten={script_len} chars = {pct}%, "
        f"allowed range: {min_chars}-{max_chars} chars)\n"
        f"2. COMPLETENESS: Are all key events, facts, and narrative beats preserved?\n"
        f"3. VOICEOVER QUALITY: Does it sound natural when read aloud? "
        f"No heavy sentences, no awkward phrasing?\n"
        f"4. NO REPETITION: Is it free of unnecessary repetition or filler?\n"
        f"5. LANGUAGE: Is it correctly and fluently written in {language}?\n"
        f"6. UNIQUENESS: Is it genuinely rewritten (not just synonymized)?\n\n"
        f"Scoring: 1-10. PASSED if score >= 7 AND length is between 70% and 80% of original.\n\n"
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
                    f"Must be at least {min_chars} chars (70% of original). " + feedback
                )
            elif script_len > max_chars:
                passed   = False
                feedback = (
                    f"Script is too long: {script_len} chars ({pct}% of original {orig_len} chars). "
                    f"Must be at most {max_chars} chars (80% of original). "
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
                # Split off Ukrainian translation for clean YouTube title.
                main_part = _split_title_translation(full)
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
        # Accept comma-separated output as requested, but also tolerate a
        # model returning one tag per line or Markdown bullets.
        tags_raw = re.sub(r"```(?:\w+)?", "", tags_raw).strip()
        if "," not in tags_raw:
            lines = []
            for line in tags_raw.splitlines():
                line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
                if line:
                    lines.append(line)
            if len(lines) > 1:
                tags_raw = ", ".join(lines)
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


def _split_title_translation(full: str) -> str:
    if " || " in full:
        return full.split(" || ", 1)[0].strip()
    if " — " in full:
        return full.rsplit(" — ", 1)[0].strip()
    return full.strip()


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
        f"1. Title in {language} || Ukrainian translation\n"
        f"2. Title in {language} || Ukrainian translation\n"
        f"3. Title in {language} || Ukrainian translation\n"
        f"4. Title in {language} || Ukrainian translation\n"
        f"5. Title in {language} || Ukrainian translation\n"
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
        f"Write ALL tags in {language}. Localize names, acronyms, weapons, places, and search entities "
        f"into the form normally used by {language} speakers. Do not add generic English tags to "
        f"non-English videos. Comma-separated.\n\n"
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
        if source_title:
            result["titles"] = [source_title]
            result["titles_main"] = [source_title]
            result["title"] = source_title
    if not result.get("description"):
        missing.append("description")
        result["description"] = source_description or ""
    if not result.get("tags"):
        missing.append("tags")
        fallback_tags = [str(tag).strip() for tag in (source_tags or []) if str(tag).strip()]
        result["tags"] = fallback_tags
        result["tags_raw"] = ", ".join(fallback_tags)
    if missing:
        print(
            "[rewriter] Metadata fields missing after API response: "
            + ", ".join(missing)
            + "; using source metadata fallback",
            flush=True,
        )
    print(f"[rewriter] Metadata done — {len(result['titles'])} title options", flush=True)
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

MAX_REWRITE_ATTEMPTS = 1

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
    soft_max_chars = _soft_max_chars(orig_len)
    settings = config.load_settings()
    skip_rewrite = not bool(settings.get("rewrite_script_enabled", True))

    if skip_rewrite:
        print("[rewriter] script rewrite disabled: using original transcript as script", flush=True)
        script = transcript.strip()
    elif test_mode:
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
        orig_words = _word_count(transcript)
        min_words, max_words = _word_bounds(orig_words)
        print(
            f"[rewriter] Word target: {min_words}-{max_words} words "
            f"(original={orig_words}, ideal={int(orig_words * TARGET_LENGTH_RATIO)})",
            flush=True,
        )
        quality_passed = False
        for attempt in range(MAX_REWRITE_ATTEMPTS):
            print(
                f"[rewriter] Rewrite attempt {attempt + 1}/{MAX_REWRITE_ATTEMPTS}"
                + (f" (feedback: {feedback[:80]}...)" if feedback else ""),
                flush=True,
            )
            script = _rewrite_script(transcript, language_name, source_title, feedback=feedback, test_mode=False)

            # If the model overshot the max length, compress the whole script with an LLM
            # pass instead of cutting off sentences from the end.
            was_compressed = False
            parts = list(_LAST_REWRITTEN_PARTS) or [p for p in script.split("\n\n") if p.strip()]
            if len(script) > max_chars:
                if len(parts) > 1:
                    script, parts = _compress_parts_to_bounds(
                        parts=parts,
                        transcript=transcript,
                        language=language_name,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        feedback=feedback,
                    )
                    if len(script) > max_chars:
                        script, parts = _compress_parts_to_bounds(
                            parts=parts,
                            transcript=transcript,
                            language=language_name,
                            min_chars=min_chars,
                            max_chars=max_chars,
                            feedback=(
                                "The previous compression is still too long. "
                                "Compress more aggressively but keep the ending and key facts."
                            ),
                        )
                else:
                    script = _compress_until_in_bounds(
                        script=script,
                        transcript=transcript,
                        language=language_name,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        feedback=feedback,
                    )
                    parts = [script]
                was_compressed = True

            continuity_ok, continuity_feedback = _continuity_check_script(script, parts, language_name)
            if not continuity_ok:
                print("[rewriter] Continuity polish pass...", flush=True)
                old_len = len(script)
                try:
                    script = _polish_script_continuity(
                        script=script,
                        language=language_name,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        feedback=continuity_feedback,
                    )
                    if len(script) > max_chars:
                        script = _compress_until_in_bounds(
                            script=script,
                            transcript=transcript,
                            language=language_name,
                            min_chars=min_chars,
                            max_chars=max_chars,
                            feedback=continuity_feedback,
                        )
                    print(
                        f"[rewriter] Continuity polish done: {old_len} -> {len(script)} chars",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[rewriter] Continuity polish failed ({e}); keeping current script",
                        flush=True,
                    )

            if len(script) > max_chars:
                if len(script) <= soft_max_chars:
                    print(
                        f"[rewriter] Script over hard max but within soft max "
                        f"({len(script)}>{max_chars}, soft max {soft_max_chars}); "
                        "keeping coherent text",
                        flush=True,
                    )
                else:
                    old_len = len(script)
                    script = _compress_until_in_bounds(
                        script=script,
                        transcript=transcript,
                        language=language_name,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        feedback=feedback,
                    )
                    parts = [script]
                    print(
                        f"[rewriter] Final LLM length pass: {old_len} -> {len(script)} chars",
                        flush=True,
                    )

            passed, feedback = _quality_check_script(script, transcript, language_name, test_mode=False)
            if passed:
                quality_passed = True
                print(f"[rewriter] Quality check PASSED on attempt {attempt + 1}", flush=True)
                break
            else:
                print(
                    f"[rewriter] Quality check FAILED on attempt {attempt + 1}: {feedback[:120]}",
                    flush=True,
                )
                if min_chars <= len(script) <= max_chars:
                    quality_passed = True
                    print(
                        "[rewriter] Accepting script despite QC style feedback because length is valid; "
                        "avoiding full rewrite loop.",
                        flush=True,
                    )
                    break
                if max_chars < len(script) <= soft_max_chars:
                    quality_passed = True
                    print(
                        "[rewriter] Accepting script despite QC feedback because length is only "
                        "slightly over hard max; avoiding destructive local cuts.",
                        flush=True,
                    )
                    break
        if not quality_passed:
            if min_chars <= len(script) <= max_chars:
                print(
                    "[rewriter] Accepting script after QC fallback; "
                    f"last QC feedback: {feedback[:300]}",
                    flush=True,
                )
            elif max_chars < len(script) <= soft_max_chars:
                print(
                    "[rewriter] Accepting coherent script above hard max but within soft max; "
                    f"last QC feedback: {feedback[:300]}",
                    flush=True,
                )
            elif len(script) > soft_max_chars:
                print(
                    "[rewriter] Accepting coherent script despite overlength because local sentence "
                    f"cuts are disabled; length={len(script)} soft_max={soft_max_chars}. "
                    f"Last QC feedback: {feedback[:300]}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    "Rewrite quality check failed after "
                    f"{MAX_REWRITE_ATTEMPTS} attempts. Last feedback: {feedback[:500]}"
                )

        if len(script) < min_chars:
            raise RuntimeError(
                f"Rewrite length too short after correction: {len(script)} chars "
                f"(minimum {min_chars})."
            )
        if len(script) > max_chars:
            print(
                f"[rewriter] Rewrite length outside hard range but accepted: {len(script)} chars "
                f"(hard {min_chars}-{max_chars}, soft max {soft_max_chars}).",
                flush=True,
            )

    skip_metadata = not bool(settings.get("rewrite_metadata_enabled", True))
    if skip_metadata:
        print("[rewriter] metadata rewrite disabled: using source metadata without rewrite", flush=True)
        meta = {
            "title": source_title,
            "titles": [source_title] if source_title else [],
            "titles_main": [source_title] if source_title else [],
            "description": source_description or "",
            "tags": source_tags or [],
            "tags_raw": ", ".join(source_tags or []),
        }
    else:
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
