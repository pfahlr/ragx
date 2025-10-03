from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


TASK_PATH = Path("codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, Any]:
    assert TASK_PATH.exists(), "Task specification must exist"
    raw = TASK_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "Task specification must deserialize to a mapping"
    return data


def test_task_metadata_matches_spec() -> None:
    task = _load_task()

    assert task.get("version") == 1
    assert task.get("id") == "06ab_core_tools_minimal_subset"

    component_ids = task.get("component_ids")
    assert isinstance(component_ids, list)
    expected_components = {"core_tools", "mcp_server", "toolpacks_runtime", "observability"}
    assert expected_components.issubset(set(component_ids))

    dependencies = task.get("depends_on")
    assert isinstance(dependencies, list) and dependencies
    for required in (
        "05a_toolpacks_loader_minimal",
        "05b_toolpacks_executor_python_only",
        "05c_toolpacks_loader_spec_alignment",
        "05h_toolpacks_loader_metadata_validation",
    ):
        assert required in dependencies

    arg_spec = task.get("arg_spec")
    assert arg_spec == ["mcp_server", "task_runner"]

    test_plan = task.get("test_plan")
    assert isinstance(test_plan, list) and len(test_plan) >= 4
    assert any("Draft2020-12" in item for item in test_plan)
    assert any("DeepDiff" in item for item in test_plan)


def test_task_logging_contracts_and_requirements() -> None:
    task = _load_task()

    observability = task.get("observability_requirements")
    assert isinstance(observability, list)
    assert any("timestamp" in item for item in observability)
    assert any("runs/core_tools/minimal.jsonl" in item for item in observability)

    logging_contract = task.get("structured_logging_contract")
    assert isinstance(logging_contract, dict)
    assert logging_contract.get("format") == "jsonl"
    assert logging_contract.get("storage_path") == "runs/core_tools/minimal.jsonl"
    assert logging_contract.get("retention") == "keep-last-5"

    event_fields = logging_contract.get("event_fields")
    assert isinstance(event_fields, list)
    for field in (
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
        "error",
        "metadata",
    ):
        assert field in event_fields

    log_diff = task.get("log_diff_strategy")
    assert isinstance(log_diff, dict)
    assert log_diff.get("tool") == "deepdiff.DeepDiff"
    assert log_diff.get("baseline_path") == "tests/fixtures/mcp/core_tools/minimal_golden.jsonl"
    whitelist = log_diff.get("whitelist_fields")
    assert isinstance(whitelist, list) and whitelist
    for field in ("ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"):
        assert field in whitelist


def test_task_artifacts_and_acceptance_entries() -> None:
    task = _load_task()

    artifacts = task.get("artifacts")
    assert isinstance(artifacts, list)

    def _paths_for(name: str) -> list[str]:
        for entry in artifacts:
            if entry.get("name") == name:
                paths = entry.get("paths") or entry.get("path")
                if isinstance(paths, list):
                    return [str(item) for item in paths]
                if isinstance(paths, str):
                    return [paths]
        return []

    schema_paths = _paths_for("schemas")
    assert schema_paths
    assert "apps/mcp_server/schemas/tools/exports_render_markdown.input.schema.json" in schema_paths
    assert "apps/mcp_server/schemas/tools/exports_render_markdown.output.schema.json" in schema_paths

    toolpack_paths = _paths_for("toolpacks")
    assert "apps/mcp_server/toolpacks/core/vector.query.search.tool.yaml" in toolpack_paths

    python_paths = _paths_for("python_modules")
    assert "apps/toolpacks/python/core/vector/query_search.py" in python_paths

    assert _paths_for("structured_logs") == ["runs/core_tools/minimal.jsonl"]
    assert _paths_for("golden_fixture") == ["tests/fixtures/mcp/core_tools/minimal_golden.jsonl"]
    assert _paths_for("doc_fixture") == ["tests/fixtures/mcp/core_tools/docs/example.md"]
    assert _paths_for("documentation") == ["docs/core_tools/minimal.md"]
    assert _paths_for("log_diff_script") == ["scripts/diff_core_tool_logs.py"]

    acceptance = task.get("acceptance")
    assert isinstance(acceptance, list)
    assert any("pytest -k \"core_tools_minimal\"" in item for item in acceptance)
    assert any("DeepDiff" in item for item in acceptance)

