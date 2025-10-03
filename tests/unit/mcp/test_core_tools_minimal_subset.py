from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

TASK_PATH = Path("codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, Any]:
    assert TASK_PATH.exists(), "Task definition for 06a must exist"
    with TASK_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), "Task file must deserialize to a mapping"
    return data


def test_metadata_and_dependencies() -> None:
    task = _load_task()
    assert task.get("version") == 1, "Task version should be explicitly set to 1"
    assert task.get("id") == "06ab_core_tools_minimal_subset"

    components = task.get("components")
    assert isinstance(components, list), "components must be enumerated"
    expected_components = {"core_tools", "toolpacks_runtime", "mcp_server", "observability"}
    assert expected_components.issubset(set(components)), "Task must list all relevant components"

    arg_spec = task.get("arg_spec")
    assert isinstance(arg_spec, list) and arg_spec == ["mcp_server", "task_runner"]

    test_plan = task.get("test_plan")
    assert isinstance(test_plan, list) and any("deepdiff" in item.lower() for item in test_plan)

    actions = task.get("actions")
    assert isinstance(actions, list) and actions, "actions must be a non-empty list"
    first_stage = actions[0]
    assert first_stage.get("stage") == "tests", "First stage must enforce tests-first development"


@pytest.mark.parametrize(
    "field",
    ["ts", "agent_id", "task_id", "step_id", "event", "status", "duration_ms", "metadata"],
)
def test_structured_logging_artifacts_define_required_fields(field: str) -> None:
    task = _load_task()
    structured_logs = next(
        (item for item in task.get("artifacts", []) if item.get("name") == "structured_logs"),
        None,
    )
    assert structured_logs, "structured_logs artifact must be declared"
    assert structured_logs.get("path") == "runs/core_tools/minimal.jsonl"

    contract = task.get("structured_logging_contract")
    assert isinstance(contract, dict)
    assert contract.get("format") == "jsonl"
    assert contract.get("storage_path") == "runs/core_tools/minimal.jsonl"
    fields = contract.get("event_fields")
    assert isinstance(fields, list) and field in fields

    diff_strategy = task.get("log_diff_strategy")
    assert isinstance(diff_strategy, dict)
    assert "deepdiff" in diff_strategy.get("tool", "")
    whitelist = diff_strategy.get("whitelist_fields")
    assert isinstance(whitelist, list) and whitelist, "whitelist_fields must be configured"
    assert {"ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"}.issubset(set(whitelist))
    baseline = diff_strategy.get("baseline_path")
    assert baseline == "tests/fixtures/mcp/core_tools/minimal_golden.jsonl"


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
        "tests/unit/mcp/test_core_tool_schemas_minimal.py",
        "tests/unit/mcp/test_core_tool_logging.py",
        "tests/unit/mcp/test_core_tool_log_diff.py",
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
    assert any("pytest -k \"core_tools_minimal\"" in item for item in acceptance)

    observability_requirements = task.get("observability_requirements")
    assert isinstance(observability_requirements, list)
    assert any("timestamp" in item for item in observability_requirements)

    text = TASK_PATH.read_text(encoding="utf-8")
    for snippet in [
        "apps/mcp_server/schemas/tools/",
        "tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        "deepdiff.DeepDiff",
    ]:
        assert snippet in text, f"Task spec must mention '{snippet}'"
