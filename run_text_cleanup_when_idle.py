"""
Wait for the FAA site job to finish, then run Qwen text cleanup.

This is meant for the server. Start it before or after launching a production
job; by default it waits until it has seen a running job at least once, then
waits for the site to become idle and runs war_cleanup_text_clips.py.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime


DEFAULT_STATUS_URL = "http://127.0.0.1:5050/api/status"


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_status(url: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run war text cleanup after FAA production becomes idle.")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--settle-seconds", type=int, default=120)
    parser.add_argument(
        "--no-wait-for-job-start",
        action="store_true",
        help="Run cleanup as soon as the site is idle, even if this watcher never saw a running job.",
    )
    parser.add_argument(
        "--cleanup-command",
        nargs=argparse.REMAINDER,
        default=None,
        help="Command to run after idle. Defaults to /venv/main/bin/python war_cleanup_text_clips.py ...",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    poll_seconds = max(5, args.poll_seconds)
    seen_running = bool(args.no_wait_for_job_start)
    last_running = None

    command = args.cleanup_command or [
        "/venv/main/bin/python",
        "war_cleanup_text_clips.py",
        "--frames",
        "0.10,0.50,0.90",
        "--min-confidence",
        "0.85",
        "--delete-severities",
        "medium,heavy",
    ]

    print(f"[watcher] {_stamp()} watching {args.status_url}", flush=True)
    print(f"[watcher] Cleanup command: {' '.join(command)}", flush=True)
    if not seen_running:
        print("[watcher] Waiting until a production job is seen running at least once.", flush=True)

    while True:
        try:
            status = _get_status(args.status_url, args.timeout)
            running = bool(status.get("job_running"))
            msg = str(status.get("last_msg") or "").strip()
            if running:
                seen_running = True

            if running != last_running:
                state = "running" if running else "idle"
                print(f"[watcher] {_stamp()} site is {state}: {msg}", flush=True)
                last_running = running
            elif msg:
                print(f"[watcher] {_stamp()} status: {msg}", flush=True)

            if seen_running and not running:
                if args.settle_seconds > 0:
                    print(f"[watcher] Site is idle. Waiting {args.settle_seconds}s before cleanup.", flush=True)
                    time.sleep(args.settle_seconds)
                    status = _get_status(args.status_url, args.timeout)
                    if bool(status.get("job_running")):
                        print("[watcher] A new job started during settle wait; continuing to watch.", flush=True)
                        continue
                print(f"[watcher] {_stamp()} starting cleanup", flush=True)
                completed = subprocess.run(command)
                print(f"[watcher] {_stamp()} cleanup exited with code {completed.returncode}", flush=True)
                return completed.returncode

        except KeyboardInterrupt:
            print("[watcher] Stopped.", flush=True)
            return 130
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[watcher] {_stamp()} status unavailable: {exc}", flush=True)
        except Exception as exc:
            print(f"[watcher] {_stamp()} error: {exc}", flush=True)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
