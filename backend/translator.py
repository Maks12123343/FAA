"""Batch translate section texts to English for clip matching."""
import json
import os
import re
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend import api_client
from backend import languages as lang_utils


def _cache_path(project_dir: str) -> str:
    return os.path.join(project_dir, "sections_english.json")


def _is_english(language: str) -> bool:
    value = (language or "").strip().lower()
    return value.startswith("en") or value == "english"


def translate_sections(section_texts: list, language: str, project_dir: str = None, emit=None) -> list:
    """
    Translate section texts to English for clip matching.
    If language is English, returns texts as-is.
    Results are cached to project_dir/sections_english.json.
    Uses A6API (batch of ~30 sections per call).
    """
    if _is_english(language):
        return section_texts

    if not section_texts:
        return []

    # Check cache
    if project_dir:
        cp = _cache_path(project_dir)
        if os.path.exists(cp):
            try:
                with open(cp, encoding="utf-8") as f:
                    cached = json.load(f)
                fp = hashlib.md5("\n".join(section_texts[:5]).encode()).hexdigest()[:12]
                if cached.get("fp") == fp and len(cached.get("texts", [])) == len(section_texts):
                    print(f"[translator] Loaded from cache ({len(section_texts)} sections)", flush=True)
                    if emit:
                        emit("translate", f"English translations loaded from cache ({len(section_texts)} sections)")
                    return cached["texts"]
            except Exception:
                pass

    language_name = lang_utils.configured_language_name(language)
    print(f"[translator] Translating {len(section_texts)} sections from '{language_name}' to English...", flush=True)
    if emit:
        emit("translate", f"Translating {len(section_texts)} sections to English for clip matching...")

    translated = _batch_translate(section_texts, language, emit=emit)

    print(f"[translator] Done. Example: '{section_texts[0][:60]}' → '{translated[0][:60]}'", flush=True)

    # Save cache
    if project_dir:
        fp = hashlib.md5("\n".join(section_texts[:5]).encode()).hexdigest()[:12]
        try:
            os.makedirs(project_dir, exist_ok=True)
            with open(_cache_path(project_dir), "w", encoding="utf-8") as f:
                json.dump({"fp": fp, "texts": translated}, f, ensure_ascii=False)
        except Exception:
            pass

    return translated


_BATCH_SIZE = 30


def _batch_translate(texts: list, source_lang: str, emit=None) -> list:
    """Translate texts in batches using A6API."""
    source_lang_name = lang_utils.configured_language_name(source_lang)

    all_translated = []
    batches = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
        prompt = (
            f"Translate these {len(batch)} text segments from {source_lang_name} to English. "
            f"Keep the same numbering. Output ONLY the translations, one per line, numbered.\n\n"
            f"{numbered}"
        )

        result_texts = None
        try:
            raw_text, _ = api_client.call_rewrite_api(
                "You are a translator. Translate accurately and concisely. Keep numbering format: '1. translation'",
                [{"role": "user", "content": prompt}],
                timeout=120,
                max_retries=3,
                emit=emit,
                step_label="translate",
            )
            result_texts = _parse_numbered(raw_text, len(batch))
        except Exception as e:
            print(f"[translator] Batch {batch_idx+1} A6API error: {e}", flush=True)

        if result_texts and len(result_texts) == len(batch):
            all_translated.extend(result_texts)
        else:
            print(f"[translator] Batch {batch_idx+1} failed — using originals", flush=True)
            all_translated.extend(batch)

        if emit and len(batches) > 1:
            emit("translate", f"Translated batch {batch_idx+1}/{len(batches)}...")

    return all_translated


def _parse_numbered(text: str, expected: int) -> list:
    """Parse numbered translation output like '1. text\n2. text\n...'"""
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if m:
            results.append(m.group(1).strip())
        elif results:
            results[-1] += " " + line

    if len(results) == expected:
        return results

    # Fallback: try splitting by number pattern
    parts = re.split(r"\n\d+[\.\)]\s*", "\n" + text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == expected:
        return parts

    return results if len(results) >= expected * 0.8 else []
