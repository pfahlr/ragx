from __future__ import annotations

import json
from pathlib import Path

from deepdiff import DeepDiff

VOLATILE_ROOT_FIELDS = {"ts", "traceId", "spanId", "requestId", "durationMs"}
TRANSPORT_SPECIFIC_FIELDS = {"transport", "route"}
FIXTURE_PATH = Path("tests/fixtures/mcp/envelope_validation_golden.jsonl")


def _load_events() -> tuple[dict[str, object], dict[str, object]]:
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    http_event = json.loads(lines[0])
    stdio_event = json.loads(lines[1])
    return http_event, stdio_event


def _strip_volatile(event: dict[str, object]) -> dict[str, object]:
    filtered = {
        k: v
        for k, v in event.items()
        if k not in VOLATILE_ROOT_FIELDS and k not in TRANSPORT_SPECIFIC_FIELDS
    }
    metadata = filtered.get("metadata")
    if isinstance(metadata, dict):
        filtered["metadata"] = dict(metadata)
    return filtered


def test_transport_parity_after_normalisation() -> None:
    http_event, stdio_event = _load_events()
    http_normalised = _strip_volatile(http_event)
    stdio_normalised = _strip_volatile(stdio_event)
    diff = DeepDiff(http_normalised, stdio_normalised, ignore_order=True)
    assert diff == {}
