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


def test_task_core_tools_minimal_metadata() -> None:
    task = _load_task()
    assert task.get("id") == "06ab_core_tools_minimal_subset"
    assert task.get("version") == 2

    components = set(task.get("component_ids", []))
    assert {"core_tools", "toolpacks_runtime", "mcp_server", "observability"}.issubset(components)

    dependencies = set(task.get("depends_on", []))
    required = {
        "05b_toolpacks_executor_python_only",
        "05c_toolpacks_loader_spec_alignment",
        "05h_toolpacks_loader_metadata_validation",
        "06a_core_tools_minimal_subset",
    }
    assert required.issubset(dependencies)


def test_task_core_tools_minimal_artifacts_and_logging() -> None:
    task = _load_task()
    artifacts = task.get("artifacts")
    assert isinstance(artifacts, dict)

    assert "schemas" in artifacts
    assert artifacts["structured_logs"]["path"] == "runs/core_tools/minimal.jsonl"
    assert artifacts["log_diff"]["tool"].startswith("deepdiff")
    whitelist = set(artifacts["log_diff"]["whitelist_fields"])
    for field in {"ts", "duration_ms", "trace_id", "span_id", "run_id", "attempt_id"}:
        assert field in whitelist
    assert artifacts["log_diff"]["baseline_path"].endswith("minimal_golden.jsonl")

    structured_contract = task.get("structured_logging_contract")
    assert structured_contract["storage_path"] == "runs/core_tools/minimal.jsonl"
    assert structured_contract["retention"] == "keep-last-5"
    assert structured_contract["event_fields"] == [
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
    ]


def test_task_core_tools_minimal_observability_actions() -> None:
    task = _load_task()
    actions = task.get("actions")
    assert isinstance(actions, list) and actions
    assert actions[0]["stage"] == "tests"

    acceptance = task.get("acceptance", [])
    assert any("deepdiff" in item for item in acceptance)
    assert any("runs/core_tools/minimal.jsonl" in item for item in acceptance)
    assert any("scripts/diff_core_tool_logs.py" in item for item in acceptance)

    observability = next((step for step in actions if step.get("stage") == "observability"), None)
    assert observability is not None
    summary = observability.get("summary", "").lower()
    assert "logging" in summary and "diff" in summary
