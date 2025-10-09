"""Integration-style executable specs for transport parity and logging."""
from __future__ import annotations

import json
from pathlib import Path

from apps.mcp_server.service.errors import CanonicalError

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
    "attempt",
    "requestId",
    "traceId",
    "spanId",
    "metadata",
    "execution",
    "idempotency",
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
        assert set(event["execution"]).issuperset({"durationMs", "inputBytes", "outputBytes"})
        assert set(event["idempotency"]).issuperset({"cacheHit"})
        assert isinstance(event["metadata"], dict)
        assert {"schemaVersion", "deterministic"}.issubset(event["metadata"])


def test_http_and_stdio_error_payloads_share_canonical_surface() -> None:
    """Both transports must emit the same canonical code and message in errors."""
    code = "INVALID_INPUT"
    http_status = CanonicalError.to_http_status(code)
    jsonrpc_payload = CanonicalError.to_jsonrpc_error(code)
    assert jsonrpc_payload["data"]["canonical"] == code
    assert jsonrpc_payload["data"]["httpStatus"] == http_status
    assert "invalid" in jsonrpc_payload["message"].lower()
