"""Canonical error mapping contract tests for the MCP server."""

from __future__ import annotations

import pytest

from apps.mcp_server.service.errors_stub import CanonicalError


@pytest.mark.xfail(strict=True, reason="Canonical error mapping not implemented yet")
@pytest.mark.parametrize(
    "code,http_status,jsonrpc_code,jsonrpc_message",
    [
        ("INVALID_INPUT", 400, -32602, "Invalid params"),
        ("INVALID_OUTPUT", 500, -32002, "Invalid output"),
        ("NOT_FOUND", 404, -32004, "Not found"),
        ("UNAUTHORIZED", 401, -32001, "Unauthorized"),
        ("INTERNAL_ERROR", 500, -32603, "Internal error"),
    ],
)
def test_canonical_error_maps_to_http_and_jsonrpc(
    code: str,
    http_status: int,
    jsonrpc_code: int,
    jsonrpc_message: str,
) -> None:
    assert CanonicalError.to_http_status(code) == http_status
    error_payload = CanonicalError.to_jsonrpc_error(code)
    assert error_payload["code"] == jsonrpc_code
    assert error_payload["message"] == jsonrpc_message
    assert error_payload["data"]["canonicalCode"] == code


@pytest.mark.xfail(strict=True, reason="Canonical error mapping not implemented yet")
def test_canonical_error_requires_known_code() -> None:
    with pytest.raises(KeyError):
        CanonicalError.to_http_status("NOT_A_REAL_CODE")
    with pytest.raises(KeyError):
        CanonicalError.to_jsonrpc_error("NOT_A_REAL_CODE")


@pytest.mark.xfail(strict=True, reason="Canonical error mapping not implemented yet")
def test_jsonrpc_mapping_includes_transport_metadata() -> None:
    payload = CanonicalError.to_jsonrpc_error("INVALID_INPUT")
    assert payload["jsonrpc"] == "2.0"
    assert payload["data"]["transport"] == "stdio"
