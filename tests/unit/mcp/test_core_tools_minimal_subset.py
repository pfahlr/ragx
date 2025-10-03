"""Regression tests for the 06ab core tools task specification."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TASK_PATH = Path("codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, Any]:
    assert TASK_PATH.exists(), "Task definition for 06ab must exist"
    with TASK_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def test_task_metadata_and_dependencies() -> None:
    task = _load_task()
    assert task.get("version") == 1
    assert task.get("id") == "06ab_core_tools_minimal_subset"

    components = set(task.get("component_ids", []))
    assert {"core_tools", "mcp_server", "toolpacks_runtime", "observability"}.issubset(components)

    depends_on = task.get("depends_on", [])
    for dep in [
        "05b_toolpacks_executor_python_only",
        "05c_toolpacks_loader_spec_alignment",
        "05h_toolpacks_loader_metadata_validation",
    ]:
        assert dep in depends_on


def test_task_artifacts_cover_logs_and_schemas() -> None:
    task = _load_task()
    artifacts = task.get("artifacts", {})
    structured_logs = artifacts.get("structured_logs", {})
    assert structured_logs.get("format") == "jsonl"
    assert structured_logs.get("path") == "runs/core_tools/minimal.jsonl"

    schema = structured_logs.get("schema", {})
    required_fields = set(schema.get("required_fields", []))
    for field in [
        "ts",
        "agent_id",
        "task_id",
        "step_id",
        "trace_id",
        "span_id",
        "tool_id",
        "event",
        "status",
        "duration_ms",
        "attempt",
        "input_bytes",
        "output_bytes",
    ]:
        assert field in required_fields

    metadata_fields = set(schema.get("metadata_fields", []))
    for field in ["run_id", "attempt_id", "schema_version", "deterministic"]:
        assert field in metadata_fields

    log_diff = artifacts.get("log_diff", {})
    assert log_diff.get("tool") == "deepdiff.DeepDiff"
    assert log_diff.get("baseline_path") == "tests/fixtures/mcp/core_tools/minimal_golden.jsonl"


def test_task_actions_and_acceptance() -> None:
    task = _load_task()
    actions = task.get("actions", [])
    assert actions

    tests_stage = next((entry for entry in actions if entry.get("stage") == "tests"), None)
    assert tests_stage is not None
    tests_config = tests_stage.get("tests", {})
    unit_paths = {item["path"] for item in tests_config.get("unit", [])}
    assert {
        "tests/unit/mcp/test_core_tool_schemas.py",
        "tests/unit/mcp/test_core_tool_logging.py",
        "tests/unit/mcp/test_core_tool_invocations.py",
        "tests/unit/mcp/test_core_tool_log_diff.py",
        "tests/unit/mcp/test_log_diff_script.py",
    }.issubset(unit_paths)

    fixtures = {item["path"] for item in tests_config.get("fixtures", [])}
    assert {
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "tests/fixtures/mcp/core_tools/docs/example.md",
        "tests/fixtures/mcp/core_tools/docs/example.json",
    }.issubset(fixtures)

    acceptance = task.get("acceptance", [])
    assert any("deepdiff" in entry for entry in acceptance)
    assert any("runs/core_tools/minimal.jsonl" in entry for entry in acceptance)
    assert any("scripts/diff_core_tool_logs.py" in entry for entry in acceptance)
