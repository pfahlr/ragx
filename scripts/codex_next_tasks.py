from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

import yaml

TASK_PATTERN = "codex/agents/TASKS/*.yaml"


def iter_task_files(pattern: str = TASK_PATTERN) -> Iterable[str]:
    return glob.glob(pattern)


def main() -> None:
    task_files = sorted(iter_task_files())
    print("Next tasks:")
    for task_file in task_files:
        path = Path(task_file)
        with path.open(encoding="utf-8") as handle:
            yaml.safe_load(handle)
        print(f"- {task_file}")


if __name__ == "__main__":
    main()
