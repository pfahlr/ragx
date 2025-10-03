from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from deepdiff import DeepDiff


def _load(path: Path, whitelist: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        for key in whitelist:
            data.pop(key, None)
        records.append(data)
    return records


def compare_logs(produced: Path, baseline: Path, *, whitelist: Iterable[str]) -> DeepDiff:
    """Compare structured logs using DeepDiff with a whitelist of volatile fields."""

    whitelist_set = set(whitelist)
    produced_records = _load(produced, whitelist_set)
    baseline_records = _load(baseline, whitelist_set)
    return DeepDiff(baseline_records, produced_records, ignore_order=True)


def main() -> int:  # pragma: no cover - CLI convenience
    parser = argparse.ArgumentParser(description="Diff core tool logs against a golden fixture")
    parser.add_argument("produced", type=Path, help="Path to produced JSONL log file")
    parser.add_argument("baseline", type=Path, help="Path to golden JSONL log file")
    parser.add_argument(
        "--whitelist",
        nargs="*",
        default=["ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"],
        help="Fields to ignore during comparison",
    )
    args = parser.parse_args()

    diff = compare_logs(args.produced, args.baseline, whitelist=args.whitelist)
    if diff:
        print(diff)  # noqa: T201 - CLI output
        return 1
    print("No differences detected")  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
