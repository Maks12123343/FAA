"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os
import time

from backend import api_client

THUMBNAIL_ATTEMPTS = 3


def _emit(emit, msg: str):
    if emit:
        emit("thumbnail", msg)


def _image_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _call_thumbnail_step(system: str, messages: list, label: str, emit=None) -> str:
    last_err = None
    for attempt in range(1, THUMBNAIL_ATTEMPTS + 1):
        try:
            _emit(emit, f"{label} attempt {attempt}/{THUMBNAIL_ATTEMPTS}...")
            text, _ = api_client.call_pioneer(
                system,
                messages,
                timeout=90,
                max_retries=2,
                emit=emit,
                step_label="thumbnail",
                use_rewrite_model=False,
            )
            text = (text or "").strip()
            if text:
                return text
            last_err = "empty response"
            _emit(emit, f"{label} returned empty response")
        except Exception as e:
            last_err = str(e)
            _emit(emit, f"{label} failed attempt {attempt}/{THUMBNAIL_ATTEMPTS}: {e}")
            print(
                f"[thumbnail] {label} failed attempt {attempt}/{THUMBNAIL_ATTEMPTS}: {e}",
                flush=True,
            )
        if attempt < THUMBNAIL_ATTEMPTS:
            time.sleep(5 * attempt)
    raise RuntimeError(f"{label} failed after {THUMBNAIL_ATTEMPTS} attempts: {last_err}")


def analyze_and_rewrite(image_path: str, language: str, title: str = "", emit=None) -> dict:
    """Analyze a competitor thumbnail and return a prompt for a new one.

    Returns {"prompt": str, "analysis": str}. Any API failure is raised to the
    caller; the pipeline catches it and continues without a thumbnail prompt.
    """
    if not image_path or not os.path.exists(image_path):
        return {"prompt": "", "analysis": ""}

    _emit(emit, "Analyzing source thumbnail...")
    data_url = _image_data_url(image_path)

    analysis_prompt = (
        "Look at this YouTube thumbnail. Describe it for recreating a similar "
        "but unique thumbnail.\n"
        "Cover: layout, text placement, font style, colors, subject, background, "
        "mood, contrast, and any visual hooks. Be specific and practical."
    )
    analysis = _call_thumbnail_step(
        "You are a precise YouTube thumbnail design analyst.",
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": analysis_prompt},
            ],
        }],
        "Thumbnail analysis",
        emit=emit,
    )

    _emit(emit, "Writing thumbnail generation prompt...")
    rewrite_prompt = (
        "Based on this competitor thumbnail analysis, write ONE detailed prompt "
        "for generating a new YouTube thumbnail image.\n\n"
        f"VIDEO TITLE:\n{title or '(unknown)'}\n\n"
        f"TARGET LANGUAGE FOR ANY TEXT IN THE THUMBNAIL:\n{language}\n\n"
        f"COMPETITOR THUMBNAIL ANALYSIS:\n{analysis}\n\n"
        "Requirements:\n"
        "- Keep the same general concept, clarity, and emotional hook.\n"
        "- Make it visually distinct: change background, angle, secondary elements, "
        "and color accents.\n"
        "- Any visible text must be in the target language.\n"
        "- Do not mention copyrighted logos or ask for an exact copy.\n"
        "- Output only the image generation prompt, no explanation."
    )
    prompt = _call_thumbnail_step(
        "You write concise, production-ready image generation prompts.",
        [{"role": "user", "content": rewrite_prompt}],
        "Thumbnail prompt rewrite",
        emit=emit,
    )

    return {
        "prompt": (prompt or "").strip(),
        "analysis": (analysis or "").strip(),
    }
