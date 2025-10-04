from __future__ import annotations

import pytest

from apps.mcp_server.service.errors_stub import CanonicalError

HTTP_EXPECTATIONS: dict[str, int] = {
    "INVALID_INPUT": 400,
    "INVALID_OUTPUT": 502,
    "NOT_FOUND": 404,
    "UNAUTHORIZED": 401,
    "INTERNAL_ERROR": 500,
}


JSONRPC_EXPECTATIONS: dict[str, tuple[int, str]] = {
    "INVALID_INPUT": (-32602, "Invalid params"),
    "INVALID_OUTPUT": (-32002, "Invalid tool output"),
    "NOT_FOUND": (-32004, "Resource not found"),
    "UNAUTHORIZED": (-32001, "Unauthorized"),
    "INTERNAL_ERROR": (-32603, "Internal error"),
}


@pytest.mark.spec_xfail(strict=True, reason="CanonicalError.to_http_status not implemented")
@pytest.mark.xfail(strict=True, reason="CanonicalError.to_http_status not implemented")
@pytest.mark.parametrize("code", sorted(HTTP_EXPECTATIONS))
def test_canonical_error_http_mapping(code: str) -> None:
    expected_status = HTTP_EXPECTATIONS[code]
    assert CanonicalError.to_http_status(code) == expected_status


@pytest.mark.spec_xfail(strict=True, reason="CanonicalError.to_jsonrpc_error not implemented")
@pytest.mark.xfail(strict=True, reason="CanonicalError.to_jsonrpc_error not implemented")
@pytest.mark.parametrize("code", sorted(JSONRPC_EXPECTATIONS))
def test_canonical_error_jsonrpc_mapping(code: str) -> None:
    rpc_code, message = JSONRPC_EXPECTATIONS[code]
    expected_jsonrpc = {
        "code": rpc_code,
        "message": message,
        "data": {"canonical_code": code},
    }
    assert CanonicalError.to_jsonrpc_error(code) == expected_jsonrpc


@pytest.mark.spec_xfail(strict=True, reason="CanonicalError should guard against unknown codes")
@pytest.mark.xfail(strict=True, reason="CanonicalError should guard against unknown codes")
def test_canonical_error_unknown_code() -> None:
    with pytest.raises(KeyError):
        CanonicalError.to_http_status("TOTALLY_UNKNOWN")
