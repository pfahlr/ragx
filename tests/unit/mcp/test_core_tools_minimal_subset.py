from __future__ import annotations

from pathlib import Path

import yaml

TASK_PATH = Path("codex/agents/TASKS/06a_core_tools_minimal_subset.yaml")


def _load_task() -> dict[str, object]:
    assert TASK_PATH.exists(), "Task definition for 06a must exist"
    with TASK_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), "Task file must deserialize to a mapping"
    return data


def test_task_describes_core_tools_subset() -> None:
    task = _load_task()
    assert task.get("id") == "06a_core_tools_minimal_subset"
    assert isinstance(task.get("title"), str) and task["title"]
    description = task.get("description")
    assert isinstance(description, str) and "toolpacks" in description.lower()


def test_task_acceptance_references_key_tests() -> None:
    task = _load_task()
    acceptance = task.get("acceptance")
    assert isinstance(acceptance, list) and acceptance, "acceptance criteria must be listed"
    entries = {str(item) for item in acceptance}
    assert any("tests/unit/test_core_tools_schemas.py" in item for item in entries)
    assert any("tests/e2e/test_mcp_minimal_core_tools.py" in item for item in entries)


def test_task_steps_cover_schemas_and_toolpacks() -> None:
    task = _load_task()
    steps = task.get("steps")
    assert isinstance(steps, list) and steps, "steps must be defined"
    text = "\n".join(str(item) for item in steps)
    assert "apps/mcp_server/schemas/tools/" in text
    assert "toolpacks" in text.lower()
