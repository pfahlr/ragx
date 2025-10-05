"""Canonical error helpers for MCP transports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["CanonicalError"]


@dataclass(frozen=True)
class _CanonicalErrorSpec:
    code: str
    description: str


class CanonicalError:
    """Utility for working with canonical MCP error codes."""

    _ERRORS: tuple[_CanonicalErrorSpec, ...] = (
        _CanonicalErrorSpec("INVALID_INPUT", "Request failed schema validation"),
        _CanonicalErrorSpec("INVALID_OUTPUT", "Tool returned data failing output schema"),
        _CanonicalErrorSpec("NOT_FOUND", "Requested resource does not exist"),
        _CanonicalErrorSpec("UNAUTHORIZED", "Authentication or authorization failed"),
        _CanonicalErrorSpec("INTERNAL_ERROR", "Unhandled server-side exception"),
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
            "message": "Invalid request payload",
        },
        "INVALID_OUTPUT": {
            "code": -32002,
            "message": "Tool produced invalid output",
        },
        "NOT_FOUND": {
            "code": -32601,
            "message": "Requested entity was not found",
        },
        "UNAUTHORIZED": {
            "code": -32001,
            "message": "Unauthorized",
        },
        "INTERNAL_ERROR": {
            "code": -32603,
            "message": "Internal error",
        },
    }

    @classmethod
    def codes(cls) -> Sequence[str]:
        """Return the canonical error codes in deterministic order."""

        return tuple(spec.code for spec in cls._ERRORS)

    @staticmethod
    def _lookup(code: str, mapping: Mapping[str, object], *, context: str | None = None) -> object:
        if code not in mapping:
            raise KeyError(f"{code} does not have a mapping for {context or 'requested lookup'}")
        return mapping[code]

    @classmethod
    def to_http_status(cls, code: str) -> int:
        """Translate a canonical code to an HTTP status code."""

        value = cls._lookup(code, cls._HTTP_STATUS_MAP, context="HTTP status")
        if not isinstance(value, int):  # pragma: no cover - defensive programming
            raise TypeError("HTTP status mapping must be an integer")
        return value

    @classmethod
    def to_jsonrpc_error(cls, code: str) -> dict[str, object]:
        """Translate a canonical code to a JSON-RPC compliant error payload."""

        base = cls._lookup(code, cls._JSONRPC_ERROR_MAP, context="JSON-RPC error")
        if not isinstance(base, dict):  # pragma: no cover - defensive programming
            raise TypeError("JSON-RPC mapping must be a dictionary")
        payload = dict(base)
        http_status = cls.to_http_status(code)
        payload.setdefault("data", {})
        data_field = dict(payload["data"])
        data_field.setdefault("canonical", code)
        data_field.setdefault("httpStatus", http_status)
        payload["data"] = data_field
        return payload

