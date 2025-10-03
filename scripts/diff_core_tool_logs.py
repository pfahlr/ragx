#!/usr/bin/env python3
"""Diff core tool structured logs against the golden fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from deepdiff import DeepDiff

DEFAULT_WHITELIST = {"ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"}


def _load_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalise(events: Iterable[dict[str, object]], whitelist: set[str]) -> list[dict[str, object]]:
    normalised: list[dict[str, object]] = []
    for event in events:
        filtered = {key: value for key, value in event.items() if key not in whitelist}
        normalised.append(filtered)
    return sorted(normalised, key=lambda item: (item.get("step_id"), item.get("attempt")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actual",
        default="runs/core_tools/minimal.jsonl",
        type=Path,
        help="Path to the structured log produced by the runtime.",
    )
    parser.add_argument(
        "--golden",
        default="tests/fixtures/mcp/logs/core_tools_minimal_golden.jsonl",
        type=Path,
        help="Path to the golden structured log fixture.",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=sorted(DEFAULT_WHITELIST),
        help="Fields to ignore when diffing logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    whitelist = set(args.ignore)
    actual = _normalise(_load_events(args.actual), whitelist)
    golden = _normalise(_load_events(args.golden), whitelist)
    diff = DeepDiff(golden, actual, ignore_order=True)
    if diff:
        print("Structured log differences detected:")
        print(diff)
        return 1
    print("Structured logs match the golden baseline.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

