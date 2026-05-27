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


def _call_claude(system: str, messages: list) -> tuple:
    import anthropic
    settings = config.load_settings()
    client = anthropic.Anthropic(api_key=settings.get("claude_api_key", ""))
    model = settings.get("claude_model", "claude-sonnet-4-6")
    r = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=messages,
    )
    return r.content[0].text.strip(), r.stop_reason


# ── Script rewrite ────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _rewrite_script(transcript: str, language: str, video_title: str) -> str:
    system   = _load_prompt(REWRITE_PROMPT_FILE, language)
    user_msg = f"Target language: {language}\nOriginal video title: {video_title}\n\n{transcript}"

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
    """Expand script to MIN_SCRIPT_LENGTH by adding relevant content on the same topic."""
    if len(script) >= MIN_SCRIPT_LENGTH:
        return script

    needed = MIN_SCRIPT_LENGTH - len(script)
    print(f"[rewriter] Script too short ({len(script)} chars), expanding by ~{needed} chars...", flush=True)

    system = (
        f"You are a professional scriptwriter. The user will give you an existing script about a topic. "
        f"Your task: write a CONTINUATION that adds new interesting facts, analysis, and context about the same topic. "
        f"The continuation must be in {language}, match the style and tone of the existing script, "
        f"and be approximately {needed} characters long. "
        f"Write naturally as if it's part of the same script — no introductions like 'additionally' or 'furthermore' at the very start. "
        f"Follow the same voiceover rules: numbers spelled out, smooth sentences, no ads or channel mentions. "
        f"Return ONLY the new text in a code block, nothing else."
    )

    user_msg = (
        f"Topic: {video_title}\n"
        f"Language: {language}\n"
        f"Existing script ({len(script)} chars):\n\n{script}"
    )

    messages = [{"role": "user", "content": user_msg}]
    expansion = ""
    part_num = 1

    while True:
        text, stop_reason = _call_claude(system, messages)
        expansion += ("\n\n" if expansion else "") + _extract_code_block(text)

        if stop_reason != "max_tokens":
            break

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "Continue from where you left off."})
        part_num += 1
        if part_num > 3:
            break

    combined = script + "\n\n" + expansion
    print(f"[rewriter] Expanded: {len(script)} → {len(combined)} chars", flush=True)
    return combined


# ── Metadata rewrite ──────────────────────────────────────────────────────────

def _parse_metadata_output(text: str) -> dict:
    """
    Parse the structured output:
      ### Optimized Titles:
      1. Title — Ukrainian
      ...
      ### Optimized Description:
      ...
      ### Optimized Tags:
      tag1, tag2, ...
    """
    # Titles
    titles = []
    titles_m = re.search(r"###\s*Optimized Titles:(.*?)###\s*Optimized Description:", text, re.DOTALL | re.IGNORECASE)
    if titles_m:
        for line in titles_m.group(1).strip().splitlines():
            m = re.match(r"\d+\.\s+(.+)", line.strip())
            if m:
                titles.append(m.group(1).strip())

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

    return {
        "titles":      titles,                          # all 5 options
        "title":       titles[0] if titles else "",     # first option as default
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

def rewrite_all(
    transcript: str,
    language: str,
    source_title: str,
    source_description: str = "",
    source_tags: list = None,
) -> dict:
    """
    Call 1: rewrite script (rewrite_prompt.txt)
    Call 2: rewrite metadata (metadata_prompt.txt) using SOURCE video's metadata
    Returns: {script, title, titles, description, tags}
    """
    script = _rewrite_script(transcript, language, source_title)
    script = _expand_script(script, language, source_title)
    meta   = _rewrite_metadata(
        language      = language,
        source_title  = source_title,
        source_description = source_description,
        source_tags   = source_tags or [],
    )

    return {
        "script":      script,
        "title":       meta.get("title", source_title),
        "titles":      meta.get("titles", []),
        "description": meta.get("description", ""),
        "tags":        meta.get("tags", []),
        "tags_raw":    meta.get("tags_raw", ""),
    }


# ── Legacy wrappers ───────────────────────────────────────────────────────────

def rewrite(transcript: str, language: str, video_title: str) -> str:
    return _rewrite_script(transcript, language, video_title)

def generate_title(script: str, language: str, original_title: str) -> str:
    return original_title

def generate_metadata(script: str, language: str, title: str) -> dict:
    return {"description": "", "tags": []}
