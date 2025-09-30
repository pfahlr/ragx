from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

TASK_PATH = Path("codex/agents/TASKS/06a_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, Any]:
    assert TASK_PATH.exists(), "Task definition for 06a must exist"
    with TASK_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), "Task file must deserialize to a mapping"
    return data


def test_metadata_and_dependencies() -> None:
    task = _load_task()
    assert task.get("version") == 1, "Task version should be explicitly set to 1"
    assert task.get("id") == "06a_core_tools_minimal_subset"

    components = task.get("component_ids")
    assert isinstance(components, list), "component_ids must be a list"
    expected_components = {"core_tools", "toolpacks_runtime", "mcp_server", "observability_ci"}
    assert expected_components.issubset(set(components)), "Task must list all relevant components"

    depends_on = task.get("depends_on")
    assert isinstance(depends_on, list) and depends_on, "depends_on must enumerate prerequisites"
    for required in [
        "05b_toolpacks_executor_python_only",
        "05c_toolpacks_loader_spec_alignment",
        "05h_toolpacks_loader_metadata_validation",
    ]:
        assert required in depends_on, f"Missing dependency {required}"

    actions = task.get("actions")
    assert isinstance(actions, list) and actions, "actions must be a non-empty list"
    first_stage = actions[0]
    assert first_stage.get("stage") == "tests", "First stage must enforce tests-first development"


@pytest.mark.parametrize(
    "field",
    ["ts", "agentId", "taskId", "stepId", "event", "status", "retries", "durationMs", "metadata"],
)
def test_structured_logging_artifacts_define_required_fields(field: str) -> None:
    task = _load_task()
    artifacts = task.get("artifacts")
    assert isinstance(artifacts, dict) and "structured_logs" in artifacts

    logs = artifacts["structured_logs"]
    assert logs.get("format") == "jsonl"
    schema = logs.get("schema")
    assert isinstance(schema, dict) and "required_fields" in schema
    required_fields = schema["required_fields"]
    assert isinstance(required_fields, list)
    assert field in required_fields, f"Structured logs must require '{field}'"

    metadata_fields = schema.get("metadata_fields")
    assert isinstance(metadata_fields, list), "metadata_fields must be enumerated"
    for meta_key in ["requestId", "runId", "mcpTransport", "toolpackId"]:
        assert meta_key in metadata_fields, f"metadata_fields missing {meta_key}"

    log_diff = artifacts.get("log_diff")
    assert isinstance(log_diff, dict), "log_diff configuration must exist"
    assert str(log_diff.get("tool", "")).startswith("deepdiff"), "Log diff tool must leverage deepdiff"
    whitelist = log_diff.get("whitelist_fields")
    assert isinstance(whitelist, list) and whitelist, "whitelist_fields must be configured"
    for allowed in ["ts", "durationMs", "requestId", "runId", "pid"]:
        assert allowed in whitelist, f"Whitelist must contain nondeterministic field '{allowed}'"
    baseline = log_diff.get("baseline_path")
    assert isinstance(baseline, str) and baseline.endswith("minimal_golden.jsonl"), "Baseline path must target golden logs"


def test_tests_stage_outlines_unit_e2e_and_fixtures() -> None:
    task = _load_task()
    actions = task["actions"]
    tests_stage = next((entry for entry in actions if entry.get("stage") == "tests"), None)
    assert tests_stage, "Tests stage must be defined"

    tests_section = tests_stage.get("tests")
    assert isinstance(tests_section, dict), "tests_stage.tests must be a mapping"

    unit_tests = tests_section.get("unit")
    assert isinstance(unit_tests, list) and unit_tests, "unit tests must be listed"
    unit_paths = {item.get("path") for item in unit_tests}
    assert {
        "tests/unit/test_core_tools_schemas.py",
        "tests/unit/test_core_tools_logging_minimal.py",
        "tests/unit/test_core_tools_log_diff.py",
    }.issubset(unit_paths), "Unit tests must cover schemas, logging, and log diff"

    e2e_tests = tests_section.get("e2e")
    assert isinstance(e2e_tests, list) and e2e_tests, "e2e tests must be listed"
    e2e_paths = {item.get("path") for item in e2e_tests}
    assert "tests/e2e/test_mcp_minimal_core_tools.py" in e2e_paths, "E2E test path missing"

    fixtures = tests_section.get("fixtures")
    assert isinstance(fixtures, list) and fixtures, "fixtures must be enumerated"
    fixture_paths = {item.get("path") for item in fixtures}
    assert {
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "tests/fixtures/mcp/core_tools/docs/example.md",
    }.issubset(fixture_paths), "Fixture files must include golden log and markdown fixture"

    logging_test = next((item for item in unit_tests if "logging" in item.get("path", "")), None)
    assert logging_test and "structured" in logging_test.get("description", "").lower()
    diff_test = next((item for item in unit_tests if "log_diff" in item.get("path", "")), None)
    assert diff_test and "deepdiff" in diff_test.get("description", "").lower()


def test_acceptance_and_actions_cover_observability() -> None:
    task = _load_task()
    acceptance = task.get("acceptance")
    assert isinstance(acceptance, list) and acceptance, "acceptance criteria must be listed"
    assert any("deepdiff" in item for item in acceptance), "Acceptance must require deepdiff diff check"
    assert any("runs/core_tools/minimal.jsonl" in item for item in acceptance), "Acceptance must reference structured logs"
    assert any("scripts/diff_core_tool_logs.py" in item for item in acceptance), "Acceptance must mention diff script"

    observability_stage = next((entry for entry in task.get("actions", []) if entry.get("stage") == "observability"), None)
    assert observability_stage is not None, "Observability stage must be explicitly defined"
    summary = observability_stage.get("summary", "").lower()
    assert "logging" in summary and "diff" in summary, "Observability summary must mention logging and diff"

    text = TASK_PATH.read_text(encoding="utf-8")
    for snippet in [
        "apps/mcp_server/schemas/tools/",
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "deepdiff.DeepDiff",
    ]:
        assert snippet in text, f"Task spec must mention '{snippet}'"
