"""Cross-platform one-time Google Flow profile setup."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
VALUES = dotenv_values(ROOT / ".env")
PROFILE = str(VALUES.get("FLOW_PROFILE") or "faa").strip()
FLOW_HOME = os.path.abspath(os.path.expandvars(str(
    VALUES.get("FLOW_HOME")
    or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FAA" / "flow_browser"
)))


def main() -> int:
    env = os.environ.copy()
    env.update({
        "GFLOW_CLI_HOME": FLOW_HOME,
        "GFLOW_CLI_PROFILE": PROFILE,
        "GFLOW_CLI_HEADLESS": "false",
    })
    command = [
        sys.executable, "-m", "gflow_cli", "auth", "login",
        "--profile", PROFILE,
        "--browser", "chrome",
    ]
    print("Opening a dedicated Chrome profile for Google Flow.")
    result = subprocess.run(command, env=env, check=False)
    if result.returncode:
        return result.returncode
    return subprocess.run(
        [sys.executable, "-m", "gflow_cli", "auth", "status", "--profile", PROFILE],
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
