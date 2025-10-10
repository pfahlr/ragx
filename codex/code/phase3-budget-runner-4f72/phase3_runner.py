#!/usr/bin/env python3
"""Utility script to execute the sandbox Phase 3 test suite."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = Path("codex/agents/POSTEXECUTION/P3/07b_budget_guards_and_runner_integration.yaml-565941d2177b4311b170222d4ecb5029-runlog.txt")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")]
    with LOG.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
