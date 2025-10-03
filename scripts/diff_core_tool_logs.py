#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepdiff import DeepDiff

WHITELIST = {"ts", "duration_ms", "trace_id", "span_id", "run_id", "attempt_id"}


def _load_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for field in WHITELIST:
            record.pop(field, None)
        events.append(record)
    return sorted(events, key=lambda event: (event.get("step_id", 0), event.get("attempt", 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff core tool structured logs against the golden fixture")
    parser.add_argument("--actual", default="runs/core_tools/minimal.jsonl", help="Path to the produced log file")
    parser.add_argument(
        "--expected",
        default="tests/fixtures/mcp/core_tools/minimal_golden.jsonl",
        help="Path to the golden fixture",
    )
    args = parser.parse_args()

    actual_path = Path(args.actual)
    expected_path = Path(args.expected)

    actual = _load_events(actual_path)
    expected = _load_events(expected_path)

    diff = DeepDiff(expected, actual, ignore_order=True)
    if diff:
        print("Structured log differences detected:")
        print(diff)
        return 1

    print("Structured logs match the golden fixture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
