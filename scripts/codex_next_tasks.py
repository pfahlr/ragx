"""CLI helper that surfaces the next Codex automation tasks.

This module scans ``codex/agents/TASKS`` (the single source of truth for
automation work items) and exposes a simple CLI for humans and bots.  The CLI
supports a concise plain-text view as well as a machine-readable JSON mode so
that other tooling can easily integrate with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS_ROOT = Path("codex/agents/TASKS")


@dataclass(slots=True)
class TaskRecord:
    """Lightweight representation of a Codex task entry."""

    task_id: str
    title: str
    component_ids: tuple[str, ...]
    path: str

    def to_dict(self) -> dict[str, object]:
        """Render the task record as a plain ``dict`` suitable for JSON."""

        return {
            "id": self.task_id,
            "title": self.title,
            "component_ids": list(self.component_ids),
            "path": self.path,
        }


def _coerce_title(raw: object, fallback: str) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def _coerce_component_ids(raw: object) -> tuple[str, ...]:
    if isinstance(raw, Iterable) and not isinstance(raw, str | bytes):
        return tuple(str(item) for item in raw)
    return ()


def _task_sort_key(task_id: str, path: Path) -> tuple[int, str]:
    prefix, _, remainder = task_id.partition("_")
    try:
        numeric = int(prefix)
    except ValueError:
        numeric = 1_000_000
    return (numeric, remainder or path.name)


def load_tasks(directory: Path | None = None) -> list[TaskRecord]:
    """Load and sort every task definition under ``codex/agents/TASKS``."""

    base = directory or TASKS_ROOT
    if not base.exists():
        return []

    records: list[TaskRecord] = []
    for task_path in sorted(base.glob("*.yaml")):
        data = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
        task_id = str(data.get("id") or task_path.stem)
        title = _coerce_title(data.get("title"), fallback="(missing title)")
        component_ids = _coerce_component_ids(data.get("component_ids"))
        records.append(
            TaskRecord(
                task_id=task_id,
                title=title,
                component_ids=component_ids,
                path=str(task_path),
            )
        )

    records.sort(key=lambda record: _task_sort_key(record.task_id, Path(record.path)))
    return records


def _format_task_line(record: TaskRecord, show_path: bool) -> str:
    component_text = ", ".join(record.component_ids) if record.component_ids else "unassigned"
    extra_parts: list[str] = []
    if show_path:
        extra_parts.append(record.path)
    suffix = f" ({'; '.join(extra_parts)})" if extra_parts else ""
    return f"- {record.task_id} [{component_text}] — {record.title}{suffix}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of tasks to show (defaults to 10). Use 0 to show all.",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "json"),
        default="plain",
        help="Output mode: human-readable plain text or JSON array.",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Include absolute task file paths in the plain-text output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    records = load_tasks()
    total = len(records)

    if args.limit and args.limit > 0:
        visible = records[: args.limit]
    else:
        visible = records

    if args.format == "json":
        payload = [record.to_dict() for record in visible]
        json.dump(payload, fp=sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    header_limit = len(visible)
    print(f"Next tasks (showing {header_limit} of {total} total):")
    for record in visible:
        print(_format_task_line(record, show_path=args.show_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI in tests
    raise SystemExit(main(sys.argv[1:]))
