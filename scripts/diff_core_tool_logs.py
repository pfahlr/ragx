#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from deepdiff import DeepDiff

DEFAULT_NEW = Path("runs/core_tools/minimal.jsonl")
DEFAULT_GOLDEN = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")
DEFAULT_WHITELIST = ["ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id", "log_path"]


def _load_log(path: Path, whitelist: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    whitelist_set = set(whitelist)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        metadata = dict(payload.get("metadata", {}))
        for field in whitelist_set:
            payload.pop(field, None)
            metadata.pop(field, None)
        payload["metadata"] = metadata
        records.append(payload)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff core tool logs using DeepDiff")
    parser.add_argument("--new", dest="new_log", default=str(DEFAULT_NEW), help="Path to the newly generated log")
    parser.add_argument(
        "--golden",
        dest="golden_log",
        default=str(DEFAULT_GOLDEN),
        help="Path to the golden fixture",
    )
    parser.add_argument(
        "--whitelist",
        nargs="*",
        default=DEFAULT_WHITELIST,
        help="Fields to ignore during comparison",
    )
    args = parser.parse_args(argv)

    new_log = Path(args.new_log)
    golden_log = Path(args.golden_log)

    new_records = _load_log(new_log, args.whitelist)
    golden_records = _load_log(golden_log, args.whitelist)

    diff = DeepDiff(golden_records, new_records, ignore_order=True)
    if diff:
        print("Differences detected between logs:")
        print(diff)
        return 1
    print("Logs match the golden fixture.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
