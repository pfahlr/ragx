#!/usr/bin/env python3
"""Compare structured log files using DeepDiff while ignoring volatile fields."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepdiff import DeepDiff

VOLATILE_FIELDS = {"ts", "trace_id", "span_id", "duration_ms", "run_id", "attempt_id"}


def _load(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        for field in VOLATILE_FIELDS:
            payload.pop(field, None)
        records.append(payload)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    current = _load(args.current)
    diff = DeepDiff(baseline, current, ignore_order=True)
    if diff:
        sys.stdout.write(json.dumps(diff, indent=2, sort_keys=True) + "\n")
        return 1
    sys.stdout.write("Structured logs match baseline.\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
