import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("pydantic")

GOLDEN_LOG = Path("tests/fixtures/mcp/envelope_validation_golden.jsonl")
DIFF_SCRIPT = Path("scripts/diff_envelope_validation_logs.py")


@pytest.mark.slow
def test_mcp_envelope_validation_logs_match_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not GOLDEN_LOG.exists():
        pytest.fail(f"Golden log missing: {GOLDEN_LOG}")

    log_dir = tmp_path / "runs"
    env = {**os.environ, "RAGX_SEED": "42", "RAGX_MCP_ENVELOPE_VALIDATION": "shadow"}

    cmd = [
        sys.executable,
        "-m",
        "apps.mcp_server.cli",
        "--once",
        "--deterministic-ids",
        "--log-dir",
        str(log_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr

    latest = log_dir / "logs" / "mcp_server" / "envelope_validation.latest.jsonl"
    assert latest.exists(), f"Expected structured log at {latest}"

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
            "traceId",
            "spanId",
            "requestId",
            "attemptId",
            "runId",
            "execution.durationMs",
            "execution.outputBytes",
            "logPath",
            "metadata.execution.durationMs",
            "metadata.execution.outputBytes",
        ],
        capture_output=True,
        text=True,
    )
    assert diff.returncode == 0, diff.stdout + diff.stderr
