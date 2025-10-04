"""Executable spec for canonical error mappings used by the MCP server."""
from __future__ import annotations

import pytest

from apps.mcp_server.service.errors_stub import CanonicalError

EXPECTED_CODES = {
    "OK",
    "INVALID_ARGUMENT",
    "NOT_FOUND",
    "FAILED_PRECONDITION",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "DEADLINE_EXCEEDED",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "UNAVAILABLE",
}


def test_canonical_error_declares_expected_codes() -> None:
    """The stub must enumerate the canonical error codes described in the spec."""
    assert EXPECTED_CODES == set(CanonicalError.codes())


@pytest.mark.xfail(reason="HTTP status mapping not implemented", strict=True)
def test_http_status_mapping_aligns_with_spec() -> None:
    """Canonical errors must map to deterministic HTTP status codes."""
    assert CanonicalError.to_http_status("INVALID_ARGUMENT") == 400
    assert CanonicalError.to_http_status("NOT_FOUND") == 404
    assert CanonicalError.to_http_status("FAILED_PRECONDITION") == 412
    assert CanonicalError.to_http_status("INTERNAL") == 500
    assert CanonicalError.to_http_status("UNAVAILABLE") == 503


@pytest.mark.xfail(reason="JSON-RPC error mapping not implemented", strict=True)
def test_jsonrpc_error_mapping_includes_metadata() -> None:
    """JSON-RPC errors must expose canonical metadata for downstream tooling."""
    payload = CanonicalError.to_jsonrpc_error("INVALID_ARGUMENT")
    assert payload["code"] == -32602
    assert payload["message"].lower().startswith("invalid")
    assert payload["data"]["canonical"] == "INVALID_ARGUMENT"
    assert payload["data"]["httpStatus"] == 400


@pytest.mark.parametrize("code", sorted(EXPECTED_CODES - {"OK"}))
def test_unknown_mappings_raise_key_error_when_missing(code: str) -> None:
    """Stubs should clearly communicate missing mappings during development."""
    with pytest.raises(KeyError):
        CanonicalError._lookup(code, mapping={})
