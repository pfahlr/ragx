from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT.parent / "POSTEXECUTION" / "P3" / "07b_budget_guards_and_runner_integration.yaml-3d661f4a-0b91-4ab8-b1b1-972c23901e92-runlog.txt"


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = ["pytest", str(ROOT / "tests"), "-q"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    LOG_PATH.write_text(completed.stdout + "\n" + completed.stderr)
    completed.check_returncode()


if __name__ == "__main__":
    main()
