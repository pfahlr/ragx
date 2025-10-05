#!/usr/bin/env python
"""Diff MCP envelope validation logs against the golden fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepdiff import DeepDiff

DEFAULT_BASELINE = Path("tests/fixtures/mcp/envelope_validation_golden.jsonl")
DEFAULT_NEW = Path("runs/mcp_server/envelope_validation.latest.jsonl")
VOLATILE_FIELDS = {"ts", "traceId", "spanId", "durationMs", "requestId"}


def _load_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _normalise(event: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in event.items() if key not in VOLATILE_FIELDS}
    error = payload.get("error")
    if isinstance(error, dict):
        payload["error"] = {k: v for k, v in error.items() if k not in VOLATILE_FIELDS}
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--new", dest="new_log", type=Path, default=DEFAULT_NEW)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    baseline = [_normalise(event) for event in _load_events(args.baseline)]
    candidate = [_normalise(event) for event in _load_events(args.new_log)]
    diff = DeepDiff(baseline, candidate, ignore_order=True)
    if diff:
        print("Envelope validation logs differ:")
        print(diff.to_json(indent=2))
        return 1
    print("[diff] logs match baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
