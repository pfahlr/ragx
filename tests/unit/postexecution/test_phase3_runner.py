"""Regression tests for the phase3 sandbox runner helpers."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path("codex/code/phase3-budget-guards-d98ee6c7/phase3_runner.py")
MODULE_SPEC = importlib.util.spec_from_file_location("phase3_runner", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
phase3_runner = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(phase3_runner)


def test_default_runlog_points_to_agents_postexecution():
    runlog_path = phase3_runner.DEFAULT_RUNLOG_PATH
    assert "codex/agents/POSTEXECUTION" in runlog_path.as_posix()
    assert "codex/code/agents" not in runlog_path.as_posix()


def test_write_runlog_creates_parent_directories(tmp_path):
    destination = tmp_path / "nested" / "runlog.txt"
    result = type("Result", (), {"stdout": "ok", "stderr": ""})
    phase3_runner._write_runlog(destination, result)  # type: ignore[attr-defined]
    assert destination.exists()
    assert destination.read_text() == "ok"
