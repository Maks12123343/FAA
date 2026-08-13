"""Create a persistent, separate Chrome profile for the Gemini bridge."""

from __future__ import annotations

import os
import subprocess

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
    chrome_path = _chrome_path()
    if not chrome_path:
        raise RuntimeError("Google Chrome was not found. Install Chrome first.")

    print("Opening a normal Chrome window (not an automated browser).")
    print("Sign in to Gemini, then CLOSE this dedicated Chrome window.")
    chrome = subprocess.Popen([
        chrome_path,
        f"--user-data-dir={PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        GEMINI_URL,
    ])
    try:
        input("Press Enter only after the dedicated Chrome window is closed...")
    finally:
        if chrome.poll() is None:
            print("Chrome is still open. Close it before starting the bridge.")
    print("Gemini browser profile saved. The bridge can now refresh this session automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
