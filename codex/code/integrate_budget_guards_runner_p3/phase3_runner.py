"""Utility script to execute Phase 3 regression suite for this branch workspace."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "codex" / "code" / "integrate_budget_guards_runner_p3" / "tests"
LOG_PATH = ROOT / "POSTEXECUTION" / "P3" / "07b_budget_guards_and_runner_integration.yaml-20250518-gpt5codex-runlog.txt"


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pytest",
        str(TEST_PATH),
        "-q",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    LOG_PATH.write_text(result.stdout + "\n" + result.stderr)
    result.check_returncode()


if __name__ == "__main__":
    main()
