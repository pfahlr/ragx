"""Executable spec for canonical MCP error mapping.

These tests describe the expected HTTP status codes and JSON-RPC error payloads
for canonical MCP error identifiers. They are marked ``xfail`` because the
production implementation will arrive in a subsequent task.
"""
from __future__ import annotations

import pytest

from apps.mcp_server.service.errors_stub import CanonicalError

_CANONICAL_CASES = {
    "INVALID_ENVELOPE": {
        "http": 400,
        "jsonrpc": {"code": -32600, "message": "Invalid MCP envelope"},
    },
    "INVALID_TOOL_INPUT": {
        "http": 422,
        "jsonrpc": {"code": -32602, "message": "Invalid tool input"},
    },
    "TOOL_NOT_FOUND": {
        "http": 404,
        "jsonrpc": {"code": -32001, "message": "Unknown tool"},
    },
    "INTERNAL_ERROR": {
        "http": 500,
        "jsonrpc": {"code": -32603, "message": "Internal error"},
    },
}


@pytest.mark.parametrize("code", sorted(_CANONICAL_CASES))
@pytest.mark.xfail(
    reason="Canonical error mapping not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_http_status_mapping_contract(code: str) -> None:
    """Each canonical error maps to a deterministic HTTP status code."""

    expected = _CANONICAL_CASES[code]["http"]
    assert CanonicalError.to_http_status(code) == expected


@pytest.mark.parametrize("code", sorted(_CANONICAL_CASES))
@pytest.mark.xfail(
    reason="Canonical error mapping not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_jsonrpc_error_payload_contract(code: str) -> None:
    """Canonical errors should expose JSON-RPC 2.0 compatible payloads."""

    expected = _CANONICAL_CASES[code]["jsonrpc"]
    assert CanonicalError.to_jsonrpc_error(code) == expected


@pytest.mark.xfail(
    reason="Canonical error mapping not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_unknown_error_code_is_rejected() -> None:
    """Unknown codes should raise to avoid silently returning the wrong mapping."""

    with pytest.raises(KeyError):
        CanonicalError.to_http_status("NOT_REAL")

    with pytest.raises(KeyError):
        CanonicalError.to_jsonrpc_error("NOT_REAL")
