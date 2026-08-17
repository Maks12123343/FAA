"""Google Flow image generation through a persistent, signed-in Chrome profile."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


FLOW_MODELS = {
    "flow-nano-pro": "nano-pro",
    "nano-pro": "nano-pro",
    "nano banana pro": "nano-pro",
    "gemini-3.1-flash-image": "nano-pro",
    "gemini-3-pro-image-preview": "nano-pro",
    "gemini-3-pro-image-preview-11-2025": "nano-pro",
    "flow-nano2": "nano2",
    "nano2": "nano2",
    "nano banana 2": "nano2",
    "flow-image4": "image4",
    "image4": "image4",
    "imagen 4": "image4",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class FlowGenerationError(RuntimeError):
    """A Flow generation failed with a concise, user-facing message."""


def resolve_flow_model(requested: str, default: str = "nano-pro") -> str:
    value = str(requested or "").strip().lower()
    if not value:
        return default
    return FLOW_MODELS.get(value, default)


def _tail(text: str, max_chars: int = 6000) -> str:
    clean = str(text or "").replace("\x00", "").strip()
    return clean[-max_chars:]


def _auth_error(text: str) -> bool:
    value = text.lower()
    markers = (
        "auth login",
        "authentication",
        "not authenticated",
        "session expired",
        "sign in",
        "signed out",
        "401",
    )
    return any(marker in value for marker in markers)


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        process.kill()


class FlowCliClient:
    """Serialize Flow requests and run the maintained browser automation driver."""

    def __init__(
        self,
        *,
        profile: str,
        home: str,
        default_model: str,
        timeout: int,
        retries: int,
        max_image_bytes: int,
    ) -> None:
        self.profile = profile.strip() or "faa"
        self.home = os.path.abspath(os.path.expandvars(home))
        self.default_model = resolve_flow_model(default_model)
        self.timeout = max(60, min(1800, int(timeout)))
        self.retries = max(1, min(5, int(retries)))
        self.max_image_bytes = max(1_000_000, int(max_image_bytes))
        self.lock = threading.Lock()
        self.last_success_at = 0.0
        self.last_error = ""

    @property
    def installed(self) -> bool:
        try:
            __import__("gflow_cli")
            return True
        except ImportError:
            return False

    @property
    def profile_dir(self) -> Path:
        return Path(self.home) / f"profile_{self.profile}"

    @property
    def configured(self) -> bool:
        profile = self.profile_dir
        return profile.is_dir() and any(
            path.exists()
            for path in (
                profile / ".gflow_account",
                profile / "Default" / "Cookies",
            )
        )

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "GFLOW_CLI_HOME": self.home,
            "GFLOW_CLI_PROFILE": self.profile,
            "GFLOW_CLI_HEADLESS": "false",
            "GFLOW_CLI_UI_MODE": "classic",
            "GFLOW_CLI_LEASE_WAIT_SECONDS": str(self.timeout),
            "GFLOW_CLI_UPDATE_CHECK": "0",
            "GFLOW_CLI_HISTORY_PROMPTS": "redacted",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        })
        return env

    def _run(self, command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            [sys.executable, "-m", "gflow_cli", *command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._environment(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lines: list[str] = []

        def read_output() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                lines.append(line)
                print(f"[flow-driver] {line.rstrip()}", flush=True)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() >= deadline:
                _kill_process_tree(process)
                reader.join(timeout=5)
                output = "".join(lines)
                raise FlowGenerationError(
                    f"Google Flow timed out after {timeout}s. Last output: {_tail(output)}"
                )
            time.sleep(0.25)
        reader.join(timeout=5)
        output = "".join(lines)
        return subprocess.CompletedProcess(command, process.returncode, output, "")

    def auth_status(self) -> tuple[bool, str]:
        if not self.installed:
            return False, "gflow-cli is not installed"
        if not self.configured:
            return False, f"Flow profile '{self.profile}' is not configured"
        result = self._run(["auth", "status", "--profile", self.profile], timeout=90)
        output = _tail(result.stdout, 3000)
        return result.returncode == 0, output

    def _find_output(self, expected: Path, output_dir: Path) -> Path:
        if expected.is_file() and expected.stat().st_size > 0:
            return expected
        candidates = sorted(
            (
                path
                for path in output_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stat().st_size > 0
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FlowGenerationError("Google Flow finished without a downloadable image file")
        return candidates[0]

    def generate(self, prompt: str, model: str = "") -> tuple[bytes, str]:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is empty")
        if not self.installed:
            raise FlowGenerationError(
                "gflow-cli is not installed. Run gemini_bridge/setup_browser_profile.ps1 once."
            )
        if not self.configured:
            raise FlowGenerationError(
                "Google Flow browser profile is not configured. "
                "Run gemini_bridge/setup_browser_profile.ps1 and sign in once."
            )

        flow_model = resolve_flow_model(model, self.default_model)
        os.makedirs(self.home, exist_ok=True)

        with self.lock:
            last_output = ""
            for attempt in range(1, self.retries + 1):
                output_dir = Path(tempfile.mkdtemp(prefix="faa_flow_"))
                output_path = output_dir / "thumbnail.jpg"
                try:
                    print(
                        f"[flow-bridge] generation attempt {attempt}/{self.retries} "
                        f"model={flow_model} aspect=16:9",
                        flush=True,
                    )
                    result = self._run(
                        [
                            "image", "t2i", prompt,
                            "--profile", self.profile,
                            "--model", flow_model,
                            "--aspect", "16:9",
                            "--count", "1",
                            "--ui-mode", "classic",
                            "--output", str(output_path),
                        ],
                        timeout=self.timeout,
                    )
                    last_output = _tail(result.stdout)
                    if result.returncode != 0:
                        if _auth_error(last_output):
                            raise FlowGenerationError(
                                "Google Flow session is signed out. Run "
                                "gemini_bridge/setup_browser_profile.ps1 to sign in again. "
                                f"Details: {last_output}"
                            )
                        raise FlowGenerationError(
                            f"Google Flow exited with code {result.returncode}: {last_output}"
                        )

                    generated = self._find_output(output_path, output_dir)
                    data = generated.read_bytes()
                    if len(data) < 20_000:
                        raise FlowGenerationError(
                            f"Google Flow returned an unexpectedly small image ({len(data)} bytes)"
                        )
                    if len(data) > self.max_image_bytes:
                        raise FlowGenerationError(
                            f"Google Flow image exceeded the {self.max_image_bytes}-byte limit"
                        )
                    mime = {
                        ".png": "image/png",
                        ".webp": "image/webp",
                    }.get(generated.suffix.lower(), "image/jpeg")
                    self.last_success_at = time.time()
                    self.last_error = ""
                    return data, mime
                except FlowGenerationError as exc:
                    self.last_error = str(exc)
                    if _auth_error(str(exc)) or attempt >= self.retries:
                        raise
                    time.sleep(min(10, attempt * 3))
                finally:
                    shutil.rmtree(output_dir, ignore_errors=True)

            raise FlowGenerationError(last_output or "Google Flow generation failed")
