"""Language code helpers for UI labels and LLM prompts."""

from __future__ import annotations

import re

import config


LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def _clean_profile_name(name: str) -> str:
    value = (name or "").strip()
    value = re.sub(r"\s+voice\s*$", "", value, flags=re.IGNORECASE).strip()
    return value


def _looks_like_code(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,5}", (value or "").strip().lower()))


def full_language_name(language: str, profile: dict | None = None) -> str:
    """Return a full English language name suitable for LLM prompts.

    Future custom languages work through the profile name: if a voice profile is
    named "Croatian Voice", the prompt language becomes "Croatian" even if the
    code is not in LANGUAGE_NAMES.
    """
    raw = (language or "").strip()
    code = raw.lower()
    profile_name = _clean_profile_name((profile or {}).get("name", ""))

    if code == "sw" and profile_name:
        low_name = profile_name.lower()
        if "swedish" in low_name:
            return "Swedish"
        if "swahili" in low_name:
            return "Swahili"

    if profile_name and not _looks_like_code(profile_name):
        return profile_name

    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]

    # Already a full name like "Brazilian Portuguese" or "Swedish".
    if raw and not _looks_like_code(raw):
        return _clean_profile_name(raw)

    return raw.upper() if raw else "English"


def canonical_language_code(code: str, profile: dict | None = None) -> str:
    """Normalize legacy profile codes without breaking real Swahili."""
    raw = (code or "").strip().lower()
    if raw == "sw" and full_language_name(raw, profile).lower() == "swedish":
        return "sv"
    return raw


def profile_for_language(language: str) -> dict | None:
    settings = config.load_settings()
    profiles = settings.get("voice_profiles", {}) or {}
    raw = (language or "").strip()
    code = raw.lower()
    profile = profiles.get(code)
    if profile:
        return profile
    if code == "sv":
        legacy = profiles.get("sw")
        if legacy and full_language_name("sw", legacy).lower() == "swedish":
            return legacy
    if raw and not _looks_like_code(raw):
        target = _clean_profile_name(raw).lower()
        for prof in profiles.values():
            if not isinstance(prof, dict):
                continue
            prof_name = _clean_profile_name(prof.get("name", "")).lower()
            if prof_name and prof_name == target:
                return prof
    return None


def configured_language_name(language: str) -> str:
    """Resolve language name using settings when possible, then code mapping."""
    profile = profile_for_language(language)
    return full_language_name(language, profile)
