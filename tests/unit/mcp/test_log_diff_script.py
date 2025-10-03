"""Tests for scripts/diff_core_tool_logs.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")


def _write_log(path: Path, *, modify: bool) -> None:
    payloads = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    if modify:
        payloads[0]["status"] = "mutated"
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


@pytest.mark.parametrize("modify,expected", [(False, 0), (True, 1)])
def test_diff_script_reports_exit_status(tmp_path: Path, modify: bool, expected: int) -> None:
    current = tmp_path / "current.jsonl"
    _write_log(current, modify=modify)

    cmd = [
        sys.executable,
        "scripts/diff_core_tool_logs.py",
        "--baseline",
        str(GOLDEN),
        "--current",
        str(current),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == expected, completed.stdout + completed.stderr
