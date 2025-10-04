"""Integration-style executable specs for transport parity and logging."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.mcp_server.service.errors_stub import CanonicalError

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_LOG = REPO_ROOT / "tests" / "fixtures" / "mcp" / "envelope_validation_golden.jsonl"
EXPECTED_EVENT_FIELDS = {
    "ts",
    "agentId",
    "taskId",
    "stepId",
    "transport",
    "route",
    "method",
    "status",
    "durationMs",
    "attempt",
    "inputBytes",
    "outputBytes",
    "requestId",
    "traceId",
    "spanId",
    "metadata",
    "error",
}


def _load_golden_events() -> list[dict]:
    contents = GOLDEN_LOG.read_text().splitlines()
    return [json.loads(line) for line in contents if line.strip()]


def test_golden_log_events_include_expected_fields() -> None:
    """Structured logging goldens should match the documented event layout."""
    events = _load_golden_events()
    assert events, "golden log must include at least one event"
    for event in events:
        assert EXPECTED_EVENT_FIELDS.issubset(event)
        assert isinstance(event["metadata"], dict)
        assert {"schemaVersion", "deterministic"}.issubset(event["metadata"])


@pytest.mark.xfail(reason="HTTP/STDIO parity not yet enforced", strict=True)
def test_http_and_stdio_error_payloads_share_canonical_surface() -> None:
    """Both transports must emit the same canonical code and message in errors."""
    code = "INVALID_ARGUMENT"
    http_status = CanonicalError.to_http_status(code)
    jsonrpc_payload = CanonicalError.to_jsonrpc_error(code)
    assert jsonrpc_payload["data"]["canonical"] == code
    assert jsonrpc_payload["data"]["httpStatus"] == http_status
    assert jsonrpc_payload["message"].lower().startswith("invalid")
