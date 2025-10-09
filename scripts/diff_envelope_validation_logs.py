import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deepdiff import DeepDiff

DEFAULT_NEW = Path("runs/mcp_server/envelope_validation.latest.jsonl")
DEFAULT_GOLDEN = Path("tests/fixtures/mcp/envelope_validation_golden.jsonl")
DEFAULT_WHITELIST = [
    "ts",
    "traceId",
    "spanId",
    "execution.durationMs",
    "requestId",
    "metadata.runId",
    "metadata.attemptId",
    "metadata.logPath",
]


def _remove_path(container: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    target = container
    for key in parts[:-1]:
        if not isinstance(target, dict) or key not in target:
            return
        next_value = target[key]
        if not isinstance(next_value, dict):
            return
        target = next_value
    if isinstance(target, dict):
        target.pop(parts[-1], None)


def _load_log(path: Path, whitelist: Iterable[str]) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    whitelist_set = set(whitelist)
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for field in whitelist_set:
            _remove_path(record, field)
            if "." not in field:
                _remove_path(record, f"metadata.{field}")
        metadata = dict(record.get("metadata", {}))
        record["metadata"] = metadata
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff envelope validation logs against golden fixture"
    )
    parser.add_argument("--new", dest="new_log", default=str(DEFAULT_NEW))
    parser.add_argument("--golden", dest="golden_log", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--whitelist", nargs="*", default=DEFAULT_WHITELIST)
    args = parser.parse_args(argv)

    new_records = _load_log(Path(args.new_log), args.whitelist)
    golden_records = _load_log(Path(args.golden_log), args.whitelist)

    diff = DeepDiff(golden_records, new_records, ignore_order=True)
    if diff:
        print("Differences detected between logs:")
        print(diff)
        return 1
    print("Logs match the golden fixture.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
