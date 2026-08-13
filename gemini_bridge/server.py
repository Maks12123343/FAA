"""Small, localhost-only Gemini Web image bridge for FAA.

This intentionally supports only text-to-image generation. Google session
cookies are read from this directory's .env and are never returned or logged.
The upstream Gemini Web protocol is undocumented and may change.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import re
import threading
import time
import uuid
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request


ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4981"))
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-image").strip()
LANGUAGE = os.environ.get("GEMINI_LANGUAGE", "en").strip() or "en"
REQUEST_TIMEOUT = max(30, min(900, int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "360"))))
MAX_IMAGE_BYTES = max(1, int(os.environ.get("MAX_IMAGE_BYTES", str(50 * 1024 * 1024))))
_default_browser_profile = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "FAA", "gemini_browser_profile",
)
GEMINI_BROWSER_PROFILE = os.path.abspath(
    os.path.expandvars(os.environ.get("GEMINI_BROWSER_PROFILE", _default_browser_profile))
)
GEMINI_BROWSER_MODE = os.environ.get("GEMINI_BROWSER_MODE", "auto").strip().lower()

GEMINI_APP = "https://gemini.google.com/app"
GEMINI_GENERATE = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_TOKEN_PATTERNS = {
    "at": (re.compile(r'"SNlM0e":"([^"]+)"'), re.compile(r'\["SNlM0e","([^"]+)"\]')),
    "push_id": (re.compile(r'"qKIAYe":"([^"]+)"'),),
    "build_label": (re.compile(r'"cfb2h":"([^"]+)"'),),
    "session_id": (re.compile(r'"FdrFJe":"([^"]+)"'),),
    "language": (re.compile(r'"TuX5cc":"([^"]+)"'),),
}


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return ""


def _clean_cookie(value: str) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _trusted_google_media_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return (
        host == "googleusercontent.com"
        or host.endswith(".googleusercontent.com")
        or host == "usercontent.google.com"
        or host.endswith(".usercontent.google.com")
    )


def _append_size(url: str, size: int = 2048) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path.rstrip("/")
    if "=s" in path or "=w" in path:
        return url
    parsed = parsed._replace(path=f"{path}=s{size}")
    return urlunparse(parsed)


def _build_inner(prompt: str, model: str, language: str, request_id: str) -> list:
    content = [prompt]
    metadata = ["", "", "", None, None, None, None, None, None, ""]
    inner = [None] * 69
    inner[0] = content
    inner[1] = [language]
    inner[2] = metadata
    inner[3] = model
    inner[6] = [1]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[0]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [1]
    inner[53] = 0
    inner[59] = request_id
    inner[61] = []
    inner[68] = 2
    return inner


def _extract_generated_urls(candidate: list) -> list[tuple[str, str]]:
    """Read Gemini's current dedicated generated-media slot."""
    if len(candidate) <= 12 or not isinstance(candidate[12], list):
        return []
    candidate_media = candidate[12]
    if len(candidate_media) <= 7 or not isinstance(candidate_media[7], list):
        return []
    media_groups = candidate_media[7]
    if not media_groups or not isinstance(media_groups[0], list):
        return []
    generated = media_groups[0]
    result = []
    for raw_image in generated:
        if not isinstance(raw_image, list) or not raw_image or not isinstance(raw_image[0], list):
            continue
        wrapper = raw_image[0]
        if len(wrapper) <= 3 or not isinstance(wrapper[3], list):
            continue
        metadata = wrapper[3]
        if len(metadata) <= 3 or not isinstance(metadata[3], str):
            continue
        image_url = metadata[3]
        if image_url.startswith("//"):
            image_url = "https:" + image_url
        parsed = urlparse(image_url)
        if parsed.scheme == "https" and _trusted_google_media_host(parsed.hostname or ""):
            mime = "image/png"
            for value in metadata:
                if isinstance(value, str) and value.startswith("image/"):
                    mime = value
            result.append((image_url, mime))
    return result


def _extract_images(body: str) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    seen = set()
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(")]}'"):
            line = line[4:]
        try:
            root = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(root, list):
            continue
        for item in root:
            if not isinstance(item, list) or len(item) < 3 or not isinstance(item[2], str):
                continue
            try:
                payload = json.loads(item[2])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, list) or len(payload) <= 4 or not isinstance(payload[4], list):
                continue
            for raw_candidate in payload[4]:
                if not isinstance(raw_candidate, list):
                    continue
                for url, mime in _extract_generated_urls(raw_candidate):
                    if url not in seen:
                        seen.add(url)
                        images.append((url, mime))
    return images


class GeminiWebClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.lock = threading.RLock()
        self.at = ""
        self.push_id = "feeds/mcudyrk2a4khkz"
        self.build_label = ""
        self.session_id = ""
        self.language = LANGUAGE
        self.available_models: set[str] = set()
        self.initialized_at = 0.0
        self._playwright = None
        self._browser_context = None
        self._browser_page = None

    def _browser_enabled(self) -> bool:
        if GEMINI_BROWSER_MODE in {"0", "false", "off", "cookies"}:
            return False
        if GEMINI_BROWSER_MODE in {"1", "true", "on", "browser"}:
            return True
        return os.path.isdir(GEMINI_BROWSER_PROFILE)

    def _reset_browser(self) -> None:
        try:
            if self._browser_context is not None and not self._browser_context.is_closed():
                self._browser_context.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._browser_page = None
        self._browser_context = None
        self._playwright = None

    def _browser_session(self) -> tuple[str, list[dict]]:
        """Open the persistent signed-in Gemini profile and return page/cookies."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Browser mode requires Playwright. Run setup_browser_profile.ps1 first."
            ) from exc

        if self._browser_context is not None and self._browser_context.is_closed():
            self._reset_browser()
        if self._browser_context is None:
            os.makedirs(GEMINI_BROWSER_PROFILE, exist_ok=True)
            self._playwright = sync_playwright().start()
            try:
                self._browser_context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=GEMINI_BROWSER_PROFILE,
                    channel="chrome",
                    headless=False,
                    viewport={"width": 1280, "height": 900},
                )
            except Exception:
                self._reset_browser()
                raise

        pages = self._browser_context.pages
        self._browser_page = pages[0] if pages else self._browser_context.new_page()
        self._browser_page.goto(f"{GEMINI_APP}?hl=en", wait_until="domcontentloaded", timeout=60000)
        self._browser_page.wait_for_timeout(1500)
        body = self._browser_page.content()
        cookies = self._browser_context.cookies(["https://gemini.google.com", "https://www.google.com"])
        return body, cookies

    def _apply_browser_cookies(self, cookies: list[dict]) -> None:
        self.session.cookies.clear()
        for cookie in cookies:
            name = cookie.get("name", "")
            if name:
                self.session.cookies.set(
                    name,
                    cookie.get("value", ""),
                    domain=cookie.get("domain") or ".google.com",
                    path=cookie.get("path") or "/",
                )

    @property
    def configured(self) -> bool:
        if self._browser_enabled():
            return os.path.isdir(GEMINI_BROWSER_PROFILE)
        return bool(_clean_cookie(os.environ.get("GEMINI_1PSID")) and _clean_cookie(os.environ.get("GEMINI_1PSIDTS")))

    def _set_primary_cookies(self):
        psid = _clean_cookie(os.environ.get("GEMINI_1PSID"))
        psidts = _clean_cookie(os.environ.get("GEMINI_1PSIDTS"))
        self.session.cookies.set("__Secure-1PSID", psid, domain=".google.com", path="/")
        self.session.cookies.set("__Secure-1PSIDTS", psidts, domain=".google.com", path="/")

    def _cookie_header(self) -> str:
        pairs = []
        for cookie in self.session.cookies:
            if cookie.name in {"__Secure-1PSID", "__Secure-1PSIDTS", "NID", "SOCS", "__Secure-1PSIDCC"}:
                pairs.append(f"{cookie.name}={cookie.value}")
        return "; ".join(pairs)

    def initialize(self, force: bool = False):
        with self.lock:
            if not self.configured:
                if self._browser_enabled():
                    raise RuntimeError(
                        f"Gemini browser profile is missing: {GEMINI_BROWSER_PROFILE}. "
                        "Run setup_browser_profile.ps1 and sign in once."
                    )
                raise RuntimeError("Gemini bridge is missing GEMINI_1PSID and GEMINI_1PSIDTS")
            if self.at and not force and time.time() - self.initialized_at < 900:
                return
            if self._browser_enabled():
                body, cookies = self._browser_session()
                self._apply_browser_cookies(cookies)
            else:
                self._set_primary_cookies()
                headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://gemini.google.com",
                    "Referer": "https://gemini.google.com/",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "X-Same-Domain": "1",
                    "User-Agent": USER_AGENT,
                }
                self.session.get("https://www.google.com/", headers=headers, timeout=30)
                response = self.session.get(f"{GEMINI_APP}?hl=en", headers=headers, timeout=30)
                response.raise_for_status()
                body = response.text
            at = _first_match(body, _TOKEN_PATTERNS["at"])
            if not at:
                raise RuntimeError("Gemini session token SNlM0e was not found; cookies may be expired")
            self.at = at
            self.push_id = _first_match(body, _TOKEN_PATTERNS["push_id"]) or self.push_id
            self.build_label = _first_match(body, _TOKEN_PATTERNS["build_label"])
            self.session_id = _first_match(body, _TOKEN_PATTERNS["session_id"])
            self.language = _first_match(body, _TOKEN_PATTERNS["language"]) or LANGUAGE
            self.available_models = {
                value
                for value in re.findall(r"gemini-[A-Za-z0-9._-]+", body)
                if "image" in value.lower()
            }
            if not self.available_models:
                self.available_models = {GEMINI_MODEL}
            self.initialized_at = time.time()

    def _resolve_model(self, requested: str) -> str:
        requested = requested.strip() or GEMINI_MODEL
        if requested in self.available_models:
            return requested
        matches = sorted(
            value for value in self.available_models
            if value.startswith(requested) or requested.startswith(value)
        )
        return matches[0] if matches else requested

    def _save_debug_response(self, body: str) -> None:
        """Keep a local bounded response sample for protocol changes; never log cookies."""
        path = os.path.join(ROOT, "last_response.txt")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body[:2_000_000])
        except OSError:
            pass

    def _download_image(self, raw_url: str) -> tuple[bytes, str]:
        image_url = _append_size(raw_url)
        parsed = urlparse(image_url)
        if parsed.scheme != "https" or not _trusted_google_media_host(parsed.hostname or ""):
            raise RuntimeError("Gemini returned an untrusted image host")
        headers = {"Referer": "https://gemini.google.com/", "User-Agent": USER_AGENT}
        for _ in range(6):
            response = self.session.get(image_url, headers=headers, timeout=120, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location", "")
                next_url = requests.compat.urljoin(image_url, location)
                next_parsed = urlparse(next_url)
                if next_parsed.scheme != "https" or not _trusted_google_media_host(next_parsed.hostname or ""):
                    raise RuntimeError("Gemini image redirect left the Google media allowlist")
                image_url = next_url
                continue
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status_code != 200 or not content_type.startswith("image/"):
                raise RuntimeError(f"Gemini image download returned HTTP {response.status_code} {content_type}")
            data = response.content
            if len(data) > MAX_IMAGE_BYTES:
                raise RuntimeError("Gemini image exceeded MAX_IMAGE_BYTES")
            return data, content_type.split(";", 1)[0]
        raise RuntimeError("Too many Gemini image redirects")

    def generate(self, prompt: str, model: str = "") -> tuple[bytes, str]:
        if not prompt.strip():
            raise ValueError("prompt is empty")
        with self.lock:
            last_error = None
            for attempt in range(1, 4):
                try:
                    self.initialize(force=attempt > 1)
                    request_id = str(uuid.uuid4()).upper()
                    resolved_model = self._resolve_model(model)
                    effective_prompt = prompt.strip()
                    if not re.match(r"(?i)^(generate|create|draw|make|render)\b", effective_prompt):
                        effective_prompt = "Generate an image based on this request: " + effective_prompt
                    inner = _build_inner(effective_prompt, resolved_model, self.language, request_id)
                    outer = [None, json.dumps(inner, ensure_ascii=False, separators=(",", ":"))]
                    form = {"at": self.at, "f.req": json.dumps(outer, ensure_ascii=False, separators=(",", ":"))}
                    # For text-only generation, the current Gemini Web client sends
                    # only the session token. Extra stream/upload query parameters
                    # are reserved for requests carrying uploaded files.
                    query = {"at": self.at}
                    trace_id = uuid.uuid4().hex[:16]
                    ext_header = f'[1,null,null,null,"{trace_id}",null,null,0,[4,5,6,8],null,null,2,null,null,6,1,"{uuid.uuid4().hex.upper()}"]'
                    headers = {
                        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                        "Origin": "https://gemini.google.com",
                        "Referer": "https://gemini.google.com/",
                        "X-Same-Domain": "1",
                        "X-Goog-Ext-525001261-Jspb": ext_header,
                        "Cookie": self._cookie_header(),
                        "User-Agent": USER_AGENT,
                    }
                    response = self.session.post(
                        GEMINI_GENERATE,
                        params=query,
                        data=form,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response.status_code in (401, 403):
                        raise RuntimeError(f"Gemini generation authentication failed: HTTP {response.status_code}")
                    response.raise_for_status()
                    images = _extract_images(response.text)
                    if not images:
                        self._save_debug_response(response.text)
                        raise RuntimeError("Gemini returned no generated image media")
                    return self._download_image(images[0][0])
                except Exception as exc:
                    last_error = exc
                    self.at = ""
                    if self._browser_enabled():
                        self._reset_browser()
                    if attempt < 3:
                        time.sleep(attempt * 2)
            raise RuntimeError(f"Gemini image generation failed after 3 attempts: {last_error}")


app = Flask(__name__)
client = GeminiWebClient()


def _authorized() -> bool:
    if not LOCAL_API_KEY:
        return False
    supplied = request.headers.get("X-API-Key", "")
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    return hmac.compare_digest(supplied, LOCAL_API_KEY)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "configured": client.configured,
        "initialized": bool(client.at),
        "browser_mode": client._browser_enabled(),
        "browser_profile": GEMINI_BROWSER_PROFILE if client._browser_enabled() else "",
        "model": GEMINI_MODEL,
        "bind": HOST,
        "port": PORT,
    })


@app.get("/v1/models")
def models():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"data": [{"id": GEMINI_MODEL, "object": "model", "owned_by": "google"}]})


@app.post("/v1/images/generations")
def generate_image():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "prompt is required"}), 400
    try:
        image, mime = client.generate(prompt, str(body.get("model", "")))
        return jsonify({
            "created": int(time.time()),
            "data": [{
                "b64_json": base64.b64encode(image).decode("ascii"),
                "revised_prompt": prompt.strip(),
                "mime_type": mime,
            }],
        })
    except Exception as exc:
        print(f"[gemini-bridge] generation failed: {type(exc).__name__}: {exc}", flush=True)
        return jsonify({"error": {"message": str(exc), "type": "gemini_web_error"}}), 502


if __name__ == "__main__":
    if not LOCAL_API_KEY:
        raise SystemExit("LOCAL_API_KEY is required in gemini_bridge/.env")
    print(f"[gemini-bridge] listening on http://{HOST}:{PORT}", flush=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
