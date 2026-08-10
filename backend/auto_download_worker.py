"""Background downloader for ready FAA projects.

The worker reuses the standalone downloader's validation, retry, date/language
folder layout and state-file logic. It runs on the machine that owns the output
folder, which is normally the Windows client with Google Drive mounted.
"""

from __future__ import annotations

import threading
import time
import urllib.error
from types import SimpleNamespace
from typing import Any

from download_ready_from_site import DEFAULT_LANGUAGES, _mark_existing_ready, _run_once


class AutoDownloadManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._signature: tuple[Any, ...] | None = None
        self._status: dict[str, Any] = {
            "enabled": False,
            "running": False,
            "last_check": 0.0,
            "last_error": "",
            "downloaded_total": 0,
        }

    def apply_settings(self, settings: dict) -> None:
        """Start, stop, or reconfigure the worker after settings are saved."""
        enabled = bool(settings.get("auto_download_enabled"))
        out_dir = str(settings.get("auto_download_out_dir") or "").strip()
        base_url = str(settings.get("auto_download_base_url") or "").strip()
        if not enabled or not out_dir or not base_url:
            self.stop()
            with self._lock:
                self._status.update({"enabled": False, "running": False})
            if enabled and (not out_dir or not base_url):
                print("[auto_download] disabled: base URL and output folder are required", flush=True)
            return

        languages = str(settings.get("auto_download_languages") or DEFAULT_LANGUAGES)
        signature = (
            base_url,
            out_dir,
            languages,
            float(settings.get("auto_download_interval_minutes", 1) or 1),
            bool(settings.get("auto_download_watch_new_only")),
            bool(settings.get("auto_download_all_ready", True)),
            int(settings.get("auto_download_retries", 5) or 5),
            int(settings.get("auto_download_timeout", 30) or 30),
            int(settings.get("auto_download_download_timeout", 7200) or 7200),
        )
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._signature == signature:
                return

        self.stop()
        stop_event = threading.Event()
        args = SimpleNamespace(
            base_url=base_url,
            out_dir=out_dir,
            state_file="",
            languages=[x.strip().lower() for x in languages.split(",") if x.strip()],
            project_ids=[],
            interval_minutes=max(1, float(settings.get("auto_download_interval_minutes", 1) or 1)),
            latest_per_language=not bool(settings.get("auto_download_all_ready", True)),
            dry_run=False,
            force=False,
            watch_new_only=bool(settings.get("auto_download_watch_new_only")),
            retries=max(1, int(settings.get("auto_download_retries", 5) or 5)),
            timeout=max(5, int(settings.get("auto_download_timeout", 30) or 30)),
            download_timeout=max(60, int(settings.get("auto_download_download_timeout", 7200) or 7200)),
        )
        thread = threading.Thread(
            target=self._run,
            args=(args, stop_event, signature),
            name="faa-auto-download",
            daemon=True,
        )
        with self._lock:
            self._stop_event = stop_event
            self._thread = thread
            self._signature = signature
            self._status.update({
                "enabled": True,
                "running": True,
                "last_error": "",
            })
        thread.start()
        print(f"[auto_download] started: {base_url} -> {out_dir}", flush=True)

    def stop(self) -> None:
        with self._lock:
            event = self._stop_event
            thread = self._thread
            self._stop_event = None
            self._thread = None
            self._signature = None
            self._status["running"] = False
        if event is not None:
            event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _run(self, args: SimpleNamespace, stop_event: threading.Event, signature: tuple[Any, ...]) -> None:
        interval = max(60, int(args.interval_minutes * 60))
        baseline_done = not args.watch_new_only
        try:
            print(
                f"[auto_download] watching {args.base_url} every {interval // 60} min; "
                f"output={args.out_dir}",
                flush=True,
            )
            while not stop_event.is_set():
                with self._lock:
                    self._status["last_check"] = time.time()
                try:
                    if baseline_done:
                        downloaded = _run_once(args)
                        with self._lock:
                            self._status["downloaded_total"] += int(downloaded or 0)
                        if downloaded:
                            continue
                    else:
                        marked = _mark_existing_ready(args)
                        print(f"[auto_download] ignored {marked} projects already ready at startup", flush=True)
                        baseline_done = True
                    with self._lock:
                        self._status["last_error"] = ""
                except urllib.error.URLError as exc:
                    message = str(exc)
                    print(f"[auto_download] connection error: {message}", flush=True)
                    with self._lock:
                        self._status["last_error"] = message
                except Exception as exc:
                    message = str(exc)
                    print(f"[auto_download] check failed: {message}", flush=True)
                    with self._lock:
                        self._status["last_error"] = message
                stop_event.wait(interval)
        finally:
            with self._lock:
                if self._signature == signature:
                    self._status["running"] = False
            print("[auto_download] stopped", flush=True)
