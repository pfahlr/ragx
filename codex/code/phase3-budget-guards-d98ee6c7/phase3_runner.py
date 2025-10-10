"""Helpers for running Phase 3 budget guard regression suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNLOG_PATH = (
    REPO_ROOT
    / "codex"
    / "agents"
    / "POSTEXECUTION"
    / "P3"
    / "phase3_runner.runlog"
)
DEFAULT_PYTEST_TARGET = "codex/code/phase3-budget-guards-d98ee6c7/tests"


def _write_runlog(destination: Path, result: subprocess.CompletedProcess[str]) -> None:
    """Persist stdout from the pytest run to ``destination``.

    The postexecution harness expects run logs to live under ``codex/agents/POSTEXECUTION``.
    When the directory is missing we create it, mirroring historical sandbox behaviour.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.stdout)


def _run_pytest(pytest_args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "pytest", "--maxfail=1", *pytest_args]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runlog",
        type=Path,
        default=DEFAULT_RUNLOG_PATH,
        help="Location where pytest stdout should be written.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to pytest (defaults to the phase3 sandbox tests).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    pytest_args: Sequence[str]
    if args.pytest_args:
        pytest_args = args.pytest_args
    else:
        pytest_args = [DEFAULT_PYTEST_TARGET, "-q"]

    result = _run_pytest(pytest_args)
    _write_runlog(args.runlog, result)

    if result.returncode != 0:
        sys.stderr.write(result.stderr)
    else:
        sys.stdout.write(result.stdout)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
