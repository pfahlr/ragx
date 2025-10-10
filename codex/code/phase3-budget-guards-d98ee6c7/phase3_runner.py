"""Minimal phase3 runner stub for regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

DEFAULT_RUNLOG_PATH = Path("codex/agents/POSTEXECUTION/phase3/runlog.txt")


class _ResultLike(Protocol):
    stdout: str
    stderr: str | None


def _write_runlog(destination: Path, result: _ResultLike) -> None:
    """Persist stdout of a phase3 run, creating parent directories."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.stdout)
