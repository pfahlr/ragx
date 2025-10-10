#!/usr/bin/env python3
"""Run Phase 3 test suite and capture output for auditing."""

from __future__ import annotations

import subprocess
from pathlib import Path

LOG_PATH = Path("codex/agents/POSTEXECUTION/P3/07b_budget_guards_and_runner_integration.yaml-20250509-gpt5codex-runlog.txt")


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            ["pytest", "codex/code/work/tests", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stream.write(process.stdout)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
