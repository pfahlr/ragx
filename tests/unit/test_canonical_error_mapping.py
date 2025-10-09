"""Executable spec for canonical error mappings used by the MCP server."""
from __future__ import annotations

import pytest

from apps.mcp_server.service.errors import CanonicalError

EXPECTED_CODES = [
    "INVALID_INPUT",
    "INVALID_OUTPUT",
    "NOT_FOUND",
    "TIMEOUT",
    "UNAUTHORIZED",
    "INTERNAL_ERROR",
]

HTTP_STATUS_EXPECTATIONS = {
    "INVALID_INPUT": 400,
    "INVALID_OUTPUT": 502,
    "NOT_FOUND": 404,
    "TIMEOUT": 504,
    "UNAUTHORIZED": 401,
    "INTERNAL_ERROR": 500,
}

JSONRPC_EXPECTATIONS = {
    "INVALID_INPUT": (-32602, "invalid input"),
    "INVALID_OUTPUT": (-32002, "invalid output"),
    "NOT_FOUND": (-32004, "not found"),
    "TIMEOUT": (-32005, "invocation timed out"),
    "UNAUTHORIZED": (-32001, "unauthorized"),
    "INTERNAL_ERROR": (-32603, "internal server error"),
}


def test_canonical_error_declares_expected_codes() -> None:
    """The stub must enumerate the canonical error codes described in the spec."""
    assert set(EXPECTED_CODES) == set(CanonicalError.codes())


def test_http_status_mapping_aligns_with_spec() -> None:
    """Canonical errors must map to deterministic HTTP status codes."""
    for code, expected in HTTP_STATUS_EXPECTATIONS.items():
        assert CanonicalError.to_http_status(code) == expected


def test_jsonrpc_error_mapping_includes_metadata() -> None:
    """JSON-RPC errors must expose canonical metadata for downstream tooling."""
    for code, (expected_code, snippet) in JSONRPC_EXPECTATIONS.items():
        payload = CanonicalError.to_jsonrpc_error(code)
        assert payload["code"] == expected_code
        assert snippet in payload["message"].lower()
        metadata = payload["data"]
        assert metadata["canonical"] == code
        assert metadata["httpStatus"] == HTTP_STATUS_EXPECTATIONS[code]
        assert isinstance(metadata["message"], str) and metadata["message"].strip()
        assert isinstance(metadata.get("retryable"), bool)


@pytest.mark.parametrize("code", sorted(EXPECTED_CODES))
def test_unknown_mappings_raise_key_error_when_missing(code: str) -> None:
    """Stubs should clearly communicate missing mappings during development."""
    with pytest.raises(KeyError):
        CanonicalError._lookup(code, mapping={})
