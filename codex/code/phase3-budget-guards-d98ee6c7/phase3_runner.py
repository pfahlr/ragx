"""Helper script to execute Phase 3 sandbox tests with coverage."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
RUNLOG = ROOT.parent / "agents" / "POSTEXECUTION" / "P3" / "07b_budget_guards_and_runner_integration.yaml-47537ede-0ec2-4de1-b9cd-10e0f3c0d68c-runlog.txt"


def main() -> None:
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pytest",
        str(TESTS),
        "--maxfail=1",
        "--disable-warnings",
        "--cov",
        "codex/code/phase3-budget-guards-d98ee6c7/pkgs",
        "--cov-report",
        "term-missing",
    ]
    with RUNLOG.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, cwd=ROOT.parents[2], stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
