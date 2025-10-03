from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN_LOG = Path("tests/fixtures/mcp/server/bootstrap_golden.jsonl")
DIFF_SCRIPT = Path("scripts/diff_mcp_server_logs.py")


def test_mcp_server_once_mode_generates_deterministic_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not GOLDEN_LOG.exists():
        pytest.fail(f"Golden log missing: {GOLDEN_LOG}")

    log_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "apps.mcp_server.cli",
        "--once",
        "--deterministic-ids",
        "--log-dir",
        str(log_dir),
    ]
    env = {**dict(os.environ), "RAGX_SEED": "42"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr

    latest = log_dir / "mcp_server" / "bootstrap.latest.jsonl"
    assert latest.exists()

    diff = subprocess.run(
        [
            sys.executable,
            str(DIFF_SCRIPT),
            "--new",
            str(latest.resolve()),
            "--golden",
            str(GOLDEN_LOG.resolve()),
            "--whitelist",
            "ts",
            "durationMs",
            "traceId",
            "spanId",
            "runId",
            "attemptId",
            "requestId",
            "logPath",
        ],
        capture_output=True,
        text=True,
    )
    assert diff.returncode == 0, diff.stdout + diff.stderr
