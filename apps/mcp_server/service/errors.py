"""Canonical error enumeration and mapping helpers for the MCP server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["CanonicalError"]


@dataclass(frozen=True)
class _CanonicalErrorSpec:
    code: str
    description: str


class CanonicalError:
    """Expose canonical error codes alongside HTTP/JSON-RPC mappings."""

    _ERRORS: tuple[_CanonicalErrorSpec, ...] = (
        _CanonicalErrorSpec("INVALID_INPUT", "Request failed schema validation"),
        _CanonicalErrorSpec("INVALID_OUTPUT", "Tool output failed schema validation"),
        _CanonicalErrorSpec("NOT_FOUND", "Requested resource was not found"),
        _CanonicalErrorSpec("UNAUTHORIZED", "Caller is not authorised to perform the action"),
        _CanonicalErrorSpec("INTERNAL_ERROR", "Unexpected internal server error"),
    )

    _HTTP_STATUS_MAP: Mapping[str, int] = {
        "INVALID_INPUT": 400,
        "INVALID_OUTPUT": 502,
        "NOT_FOUND": 404,
        "UNAUTHORIZED": 401,
        "INTERNAL_ERROR": 500,
    }

    _JSONRPC_ERROR_MAP: Mapping[str, dict[str, object]] = {
        "INVALID_INPUT": {
            "code": -32602,
            "message": "Invalid input parameters",
            "data": {"canonical": "INVALID_INPUT", "httpStatus": 400},
        },
        "INVALID_OUTPUT": {
            "code": -32000,
            "message": "Invalid tool output",
            "data": {"canonical": "INVALID_OUTPUT", "httpStatus": 502},
        },
        "NOT_FOUND": {
            "code": -32601,
            "message": "Resource not found",
            "data": {"canonical": "NOT_FOUND", "httpStatus": 404},
        },
        "UNAUTHORIZED": {
            "code": -32001,
            "message": "Unauthorized",
            "data": {"canonical": "UNAUTHORIZED", "httpStatus": 401},
        },
        "INTERNAL_ERROR": {
            "code": -32603,
            "message": "Internal error",
            "data": {"canonical": "INTERNAL_ERROR", "httpStatus": 500},
        },
    }

    @classmethod
    def codes(cls) -> Sequence[str]:
        """Return the supported canonical error codes."""

        return tuple(spec.code for spec in cls._ERRORS)

    @staticmethod
    def _lookup(code: str, mapping: Mapping[str, object], *, context: str | None = None) -> object:
        if code not in mapping:
            raise KeyError(f"{code} does not have a mapping for {context or 'requested lookup'}")
        return mapping[code]

    @classmethod
    def to_http_status(cls, code: str) -> int:
        """Translate a canonical error code into an HTTP status code."""

        value = cls._lookup(code, cls._HTTP_STATUS_MAP, context="HTTP status")
        if not isinstance(value, int):  # pragma: no cover - defensive
            raise TypeError("HTTP status mapping must be an integer")
        return value

    @classmethod
    def to_jsonrpc_error(cls, code: str) -> dict[str, object]:
        """Translate a canonical error code into a JSON-RPC error payload."""

        value = cls._lookup(code, cls._JSONRPC_ERROR_MAP, context="JSON-RPC error")
        if not isinstance(value, dict):  # pragma: no cover - defensive
            raise TypeError("JSON-RPC mapping must be a dictionary")
        return value.copy()
