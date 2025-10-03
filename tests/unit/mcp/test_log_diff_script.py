from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("deepdiff")

SCRIPT = Path("scripts/diff_core_tool_logs.py")
GOLDEN = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")


@pytest.mark.parametrize("whitelisted", [True, False])
def test_diff_script_behaviour(tmp_path: Path, whitelisted: bool) -> None:
    new_log = tmp_path / "run.jsonl"
    new_log.write_text(GOLDEN.read_text(encoding="utf-8"), encoding="utf-8")

    records = [json.loads(line) for line in new_log.read_text(encoding="utf-8").splitlines()]
    if whitelisted:
        records[0]["ts"] = "1999-01-01T00:00:00Z"
    else:
        records[0]["status"] = "err"
    new_log.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    args = [
        sys.executable,
        str(SCRIPT),
        "--new",
        str(new_log),
        "--golden",
        str(GOLDEN),
        "--whitelist",
        "ts",
        "duration_ms",
        "run_id",
        "trace_id",
        "span_id",
        "attempt_id",
        "log_path",
    ]
    result = subprocess.run(args, capture_output=True, text=True)

    if whitelisted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 1
        assert "diff" in result.stdout.lower() or "diff" in result.stderr.lower()
