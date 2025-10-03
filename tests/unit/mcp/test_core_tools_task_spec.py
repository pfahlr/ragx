"""Validates codex task definition for 06ab core tools minimal subset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TASK_PATH = Path("codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, Any]:
    assert TASK_PATH.exists(), "Task definition for 06ab must exist"
    with TASK_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), "Task file must deserialize to a mapping"
    return data


def test_task_metadata_and_dependencies() -> None:
    task = _load_task()
    assert task.get("version") == 1
    assert task.get("id") == "06ab_core_tools_minimal_subset"

    components = task.get("component_ids")
    assert isinstance(components, list), "component_ids must be a list"
    expected = {"core_tools", "toolpacks_runtime", "mcp_server", "observability"}
    assert expected.issubset(set(components))

    depends_on = task.get("depends_on")
    assert isinstance(depends_on, list) and depends_on
    for required in [
        "05b_toolpacks_executor_python_only",
        "05c_toolpacks_loader_spec_alignment",
        "05h_toolpacks_loader_metadata_validation",
    ]:
        assert required in depends_on


def test_task_artifacts_and_logging_contract() -> None:
    task = _load_task()
    artifacts = task.get("artifacts")
    assert isinstance(artifacts, dict)

    structured_logs = artifacts.get("structured_logs")
    assert isinstance(structured_logs, dict)
    assert structured_logs["format"] == "jsonl"
    assert structured_logs["path"] == "runs/core_tools/minimal.jsonl"

    schema = structured_logs.get("schema")
    assert isinstance(schema, dict)
    required_fields = schema.get("required_fields", [])
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

    metadata_fields = schema.get("metadata_fields", [])
    for meta in ["run_id", "attempt_id", "schema_version", "deterministic"]:
        assert meta in metadata_fields

    log_diff = artifacts.get("log_diff")
    assert isinstance(log_diff, dict)
    assert str(log_diff.get("tool", "")).startswith("deepdiff")
    whitelist = log_diff.get("whitelist_fields")
    assert isinstance(whitelist, list) and whitelist
    for allowed in ["ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"]:
        assert allowed in whitelist
    baseline = log_diff.get("baseline_path")
    assert baseline == "tests/fixtures/mcp/core_tools/minimal_golden.jsonl"


def test_task_actions_and_acceptance_criteria() -> None:
    task = _load_task()
    actions = task.get("actions", [])
    assert actions, "actions must be listed"

    tests_stage = next((entry for entry in actions if entry.get("stage") == "tests"), None)
    assert tests_stage, "tests stage must exist"

    tests_config = tests_stage.get("tests")
    assert isinstance(tests_config, dict)
    unit_tests = {item["path"] for item in tests_config.get("unit", [])}
    assert {
        "tests/unit/mcp/test_core_tool_schemas.py",
        "tests/unit/mcp/test_core_tool_logging.py",
        "tests/unit/mcp/test_core_tool_log_diff.py",
        "tests/unit/mcp/test_core_tool_invocations.py",
        "tests/unit/mcp/test_log_diff_script.py",
    }.issubset(unit_tests)

    e2e_tests = {item["path"] for item in tests_config.get("e2e", [])}
    assert "tests/e2e/test_mcp_minimal_core_tools.py" in e2e_tests

    fixtures = {item["path"] for item in tests_config.get("fixtures", [])}
    assert {
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "tests/fixtures/mcp/core_tools/docs/example.md",
        "tests/fixtures/mcp/core_tools/docs/example.json",
    }.issubset(fixtures)

    acceptance = task.get("acceptance", [])
    assert acceptance, "acceptance must be listed"
    assert any("deepdiff" in entry for entry in acceptance)
    assert any("runs/core_tools/minimal.jsonl" in entry for entry in acceptance)
    assert any("scripts/diff_core_tool_logs.py" in entry for entry in acceptance)

    observability_stage = next(
        (entry for entry in actions if entry.get("stage") == "observability"),
        None,
    )
    assert observability_stage, "observability stage must exist"
    summary = observability_stage.get("summary", "").lower()
    assert "logging" in summary and "diff" in summary

    text = TASK_PATH.read_text(encoding="utf-8")
    for snippet in [
        "apps/mcp_server/schemas/tools/exports_render_markdown.input.schema.json",
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "deepdiff.DeepDiff",
    ]:
        assert snippet in text
