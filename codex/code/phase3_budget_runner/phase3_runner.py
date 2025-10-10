from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    tests_path = repo_root / "codex" / "code" / "phase3_budget_runner" / "tests"
    log_path = (
        repo_root
        / "codex"
        / "agents"
        / "POSTEXECUTION"
        / "P3"
        / "07b_budget_guards_and_runner_integration.yaml-20250518-gpt5codex-runlog.txt"
    )

    command = [sys.executable, "-m", "pytest", str(tests_path), "-q"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=repo_root)

    log_path.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
