"""Utility to execute Phase 3 regression tests and capture logs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    log_path = (
        repo_root
        / "codex"
        / "agents"
        / "POSTEXECUTION"
        / "P3"
        / "07b_budget_guards_and_runner_integration.yaml-20250509T181500Z-gpt5codex-runlog.txt"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["pytest", "codex/code/work/tests", "-q"]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    log_path.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
