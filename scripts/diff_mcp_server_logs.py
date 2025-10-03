from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from deepdiff import DeepDiff

DEFAULT_NEW = Path("runs/mcp_server/bootstrap.latest.jsonl")
DEFAULT_GOLDEN = Path("tests/fixtures/mcp/server/bootstrap_golden.jsonl")
DEFAULT_WHITELIST = [
    "ts",
    "durationMs",
    "traceId",
    "spanId",
    "runId",
    "attemptId",
    "requestId",
    "logPath",
]


def _load_log(path: Path, whitelist: Iterable[str]) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    whitelist_set = set(whitelist)
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = dict(record.get("metadata", {}))
        for field in whitelist_set:
            record.pop(field, None)
            metadata.pop(field, None)
        record["metadata"] = metadata
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff MCP server logs against golden fixture")
    parser.add_argument("--new", dest="new_log", default=str(DEFAULT_NEW))
    parser.add_argument("--golden", dest="golden_log", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--whitelist", nargs="*", default=DEFAULT_WHITELIST)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
