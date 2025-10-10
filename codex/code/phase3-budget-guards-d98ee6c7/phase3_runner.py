"""Phase 3 sandbox runner for budget guards integration.

This helper executes the sandbox pytest suite and captures stdout/stderr
into the postexecution run log expected by Codex automation.  The previous
iteration attempted to build the run log path from ``ROOT.parent / "agents"``
which accidentally resolved to ``codex/code/agents`` because ``ROOT`` points at
this phase3 code directory.  Consumers (CI and documentation tooling) look for
artifacts under ``codex/agents/POSTEXECUTION`` instead, so the log was never
picked up.

The runner now derives the Codex root via ``ROOT.parents[1]`` to ensure the log
lands in ``codex/agents`` regardless of where the sandbox code itself lives.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
CODEX_ROOT = ROOT.parents[1]
AGENTS_ROOT = CODEX_ROOT / "agents"
POSTEXECUTION_ROOT = AGENTS_ROOT / "POSTEXECUTION" / "P3"
DEFAULT_RUNLOG_NAME = (
    "07b_budget_guards_and_runner_integration.yaml-"
    "47537ede-0ec2-4de1-b9cd-10e0f3c0d68c-runlog.txt"
)
DEFAULT_RUNLOG_PATH = POSTEXECUTION_ROOT / DEFAULT_RUNLOG_NAME
DEFAULT_PYTEST_COMMAND = (
    sys.executable,
    "-m",
    "pytest",
    str(ROOT / "tests"),
    "-q",
)


def _execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``command`` and capture stdout/stderr without raising on failure."""

    return subprocess.run(command, capture_output=True, text=True, check=False)


def _write_runlog(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    """Persist ``result`` output to ``path`` ensuring parent directories exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content_parts: list[str] = [result.stdout]
    if result.stderr:
        content_parts.extend(["", result.stderr])
    path.write_text("\n".join(part for part in content_parts if part), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    """Run the sandbox test suite and write the aggregated run log."""

    parser = argparse.ArgumentParser(description="Run phase3 sandbox tests")
    parser.add_argument(
        "--runlog",
        type=Path,
        default=DEFAULT_RUNLOG_PATH,
        help="Destination for the captured run log",
    )
    parser.add_argument(
        "--pytest-cmd",
        nargs=argparse.REMAINDER,
        help=(
            "Override the pytest command. Provide arguments after --pytest-cmd; "
            "defaults to the sandbox test suite."
        ),
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    command: Sequence[str]
    if parsed.pytest_cmd:
        command = tuple(parsed.pytest_cmd)
    else:
        command = DEFAULT_PYTEST_COMMAND

    result = _execute(command)
    _write_runlog(parsed.runlog, result)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
