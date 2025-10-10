"""Utility script to execute Phase 3 unit tests and persist a run log."""

from __future__ import annotations

import subprocess
from pathlib import Path

LOG_PATH = Path("POSTEXECUTION/P3/07b_budget_guards_and_runner_integration.yaml-20250509-gpt5codex-runlog.txt")


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["pytest", "codex/code/work/tests", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    LOG_PATH.write_text(result.stdout + "\n" + result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
