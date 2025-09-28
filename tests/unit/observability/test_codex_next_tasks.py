from __future__ import annotations

import json
import subprocess
import sys
from importlib import util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "codex_next_tasks.py"

spec = util.spec_from_file_location("codex_next_tasks", SCRIPT_PATH)
assert spec is not None and spec.loader is not None, "Unable to load codex_next_tasks module"
codex_next_tasks = util.module_from_spec(spec)
sys.modules.setdefault("codex_next_tasks", codex_next_tasks)
spec.loader.exec_module(codex_next_tasks)


def test_task_loading_is_sorted_and_enriched() -> None:
    tasks = codex_next_tasks.load_tasks()
    assert tasks, "expected to discover at least one task definition"

    ids = [task.task_id for task in tasks[:3]]
    assert ids == [
        "00_scaffold_directories",
        "01_ci_and_tooling",
        "02_glue_makefile_and_scripts",
    ]

    for task in tasks:
        assert Path(task.path).is_file()
        assert task.title


def test_cli_plain_output_lists_tasks() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.codex_next_tasks", "--limit", "5"],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("Next tasks (showing")
    assert any(
        "02_glue_makefile_and_scripts" in line
        and "Add Makefile and codex helper scripts" in line
        for line in lines[1:]
    )


def test_cli_plain_limit_zero_shows_all_tasks() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.codex_next_tasks", "--limit", "0"],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    header = lines[0]
    assert header.startswith("Next tasks (showing")
    assert len(lines) - 1 >= 10, "expected multiple tasks when limit is zero"


def test_cli_json_output() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.codex_next_tasks",
            "--format",
            "json",
            "--limit",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert [item["id"] for item in payload] == [
        "00_scaffold_directories",
        "01_ci_and_tooling",
    ]
    assert all(Path(item["path"]).is_file() for item in payload)
