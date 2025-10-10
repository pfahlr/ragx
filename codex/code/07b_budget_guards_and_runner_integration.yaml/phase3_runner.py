from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / 'codex' / 'code' / '07b_budget_guards_and_runner_integration.yaml' / 'tests'
LOG_PATH = ROOT / 'codex' / 'agents' / 'POSTEXECUTION' / 'P3' / '07b_budget_guards_and_runner_integration.yaml-20250513-gpt5codex-runlog.txt'


def run() -> int:
    command = [sys.executable, '-m', 'pytest', str(TEST_DIR)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    LOG_PATH.write_text(result.stdout)
    sys.stdout.write(result.stdout)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(run())
