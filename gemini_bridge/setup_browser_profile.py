"""Create a persistent, separate Chrome profile for the Gemini bridge."""

from __future__ import annotations

import os

from playwright.sync_api import sync_playwright


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "FAA",
    "gemini_browser_profile",
)
PROFILE = os.path.abspath(
    os.path.expandvars(os.environ.get("GEMINI_BROWSER_PROFILE", DEFAULT_PROFILE))
)
GEMINI_URL = "https://gemini.google.com/app?hl=en"


def _chrome_path() -> str | None:
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def main() -> int:
    os.makedirs(PROFILE, exist_ok=True)
    print(f"Gemini bridge profile: {PROFILE}")
    print("A separate Chrome window will open. Sign in to Gemini in that window.")
    print("When Gemini is loaded, optionally send one test prompt, then return here.")

    with sync_playwright() as playwright:
        launch_options = {
            "user_data_dir": PROFILE,
            "headless": False,
            "viewport": {"width": 1280, "height": 900},
        }
        try:
            context = playwright.chromium.launch_persistent_context(
                channel="chrome",
                **launch_options,
            )
        except Exception as first_error:
            chrome_path = _chrome_path()
            if not chrome_path:
                raise RuntimeError(
                    "Could not start Chrome. Install Google Chrome or set up the browser manually."
                ) from first_error
            context = playwright.chromium.launch_persistent_context(
                executable_path=chrome_path,
                **launch_options,
            )

        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60000)
            page.bring_to_front()
            input("Press Enter here after you have signed in to Gemini and the page is ready...")
        finally:
            context.close()
    print("Gemini browser profile saved. The bridge can now refresh this session automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
