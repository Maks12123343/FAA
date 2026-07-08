"""Thumbnail analysis and rewrite prompt generation."""

import base64
import mimetypes
import os

from backend import api_client


def _emit(emit, msg: str):
    if emit:
        emit("thumbnail", msg)


def _image_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
    analysis, _ = api_client.call_pioneer(
        "You are a precise YouTube thumbnail design analyst.",
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": analysis_prompt},
            ],
        }],
        timeout=90,
        max_retries=2,
        emit=emit,
        step_label="thumbnail",
        use_rewrite_model=False,
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
    prompt, _ = api_client.call_pioneer(
        "You write concise, production-ready image generation prompts.",
        [{"role": "user", "content": rewrite_prompt}],
        timeout=90,
        max_retries=2,
        emit=emit,
        step_label="thumbnail",
        use_rewrite_model=False,
    )

    return {
        "prompt": (prompt or "").strip(),
        "analysis": (analysis or "").strip(),
    }
