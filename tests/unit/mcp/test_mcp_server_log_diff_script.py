from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("deepdiff")

SCRIPT = Path("scripts/diff_mcp_server_logs.py")
GOLDEN = Path("tests/fixtures/mcp/logs/mcp_toolpacks_transport_golden.jsonl")


def test_default_whitelist_matches_spec() -> None:
    from scripts.diff_mcp_server_logs import DEFAULT_WHITELIST

    assert DEFAULT_WHITELIST == [
        "ts",
        "traceId",
        "spanId",
        "requestId",
        "attemptId",
        "runId",
        "execution.durationMs",
    ]


@pytest.mark.parametrize("whitelisted", [True, False])
def test_server_log_diff(tmp_path: Path, whitelisted: bool) -> None:
    if not GOLDEN.exists():
        pytest.fail(f"Golden log missing: {GOLDEN}")

    new_log = tmp_path / "run.jsonl"
    new_log.write_text(GOLDEN.read_text(encoding="utf-8"), encoding="utf-8")

    records = [
        json.loads(line)
        for line in new_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "expected at least one log record"

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
        "traceId",
        "spanId",
        "execution.durationMs",
        "attemptId",
        "runId",
        "requestId",
    ]
    result = subprocess.run(args, capture_output=True, text=True)

    if whitelisted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 1
        assert "diff" in result.stdout.lower() or "diff" in result.stderr.lower()
