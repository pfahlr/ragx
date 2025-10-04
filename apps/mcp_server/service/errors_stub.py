from __future__ import annotations

__all__ = ["CanonicalError"]


class CanonicalError:
    """Stub for canonical error mapping utilities."""

    _HTTP_STATUS_MAP = {
        "INVALID_INPUT": 400,
        "INVALID_OUTPUT": 502,
        "NOT_FOUND": 404,
        "UNAUTHORIZED": 401,
        "INTERNAL_ERROR": 500,
    }

    _JSONRPC_ERROR_MAP = {
        "INVALID_INPUT": {"code": -32602, "message": "Invalid params"},
        "INVALID_OUTPUT": {"code": -32002, "message": "Invalid tool output"},
        "NOT_FOUND": {"code": -32004, "message": "Resource not found"},
        "UNAUTHORIZED": {"code": -32001, "message": "Unauthorized"},
        "INTERNAL_ERROR": {"code": -32603, "message": "Internal error"},
    }

    @staticmethod
    def to_http_status(code: str) -> int:
        """Return the HTTP status for a canonical error code."""

        raise NotImplementedError("CanonicalError.to_http_status will be implemented in Part B")

    @staticmethod
    def to_jsonrpc_error(code: str) -> dict[str, object]:
        """Return the JSON-RPC error payload for a canonical code."""

        raise NotImplementedError("CanonicalError.to_jsonrpc_error will be implemented in Part B")
