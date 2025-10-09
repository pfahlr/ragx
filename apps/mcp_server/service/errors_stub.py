"""Canonical error enumeration and mapping stubs for the MCP server."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["CanonicalError"]


@dataclass(frozen=True)
class _CanonicalErrorSpec:
    code: str
    description: str


class CanonicalError:
    """Stub exposing canonical error codes and mapping helpers."""

    _ERRORS: tuple[_CanonicalErrorSpec, ...] = (
        _CanonicalErrorSpec("OK", "Success"),
        _CanonicalErrorSpec("INVALID_ARGUMENT", "Request failed schema validation"),
        _CanonicalErrorSpec("NOT_FOUND", "Resource not found"),
        _CanonicalErrorSpec("FAILED_PRECONDITION", "Preconditions not met"),
        _CanonicalErrorSpec("PERMISSION_DENIED", "Operation not permitted"),
        _CanonicalErrorSpec("UNAUTHENTICATED", "Authentication required"),
        _CanonicalErrorSpec("DEADLINE_EXCEEDED", "Deadline exceeded"),
        _CanonicalErrorSpec("TIMEOUT", "Tool execution exceeded timeout"),
        _CanonicalErrorSpec("RESOURCE_EXHAUSTED", "Rate or quota exceeded"),
        _CanonicalErrorSpec("INTERNAL", "Internal server error"),
        _CanonicalErrorSpec("UNAVAILABLE", "Service unavailable"),
    )

    _HTTP_STATUS_MAP: Mapping[str, int] = {}
    _JSONRPC_ERROR_MAP: Mapping[str, dict[str, object]] = {}

    @classmethod
    def codes(cls) -> Sequence[str]:
        """Return the ordered set of canonical error codes."""
        return tuple(spec.code for spec in cls._ERRORS)

    @staticmethod
    def _lookup(code: str, mapping: Mapping[str, object], *, context: str | None = None) -> object:
        if code not in mapping:
            raise KeyError(f"{code} does not have a mapping for {context or 'requested lookup'}")
        return mapping[code]

    @classmethod
    def to_http_status(cls, code: str) -> int:
        """Translate a canonical code to an HTTP status (not implemented)."""
        value = cls._lookup(code, cls._HTTP_STATUS_MAP, context="HTTP status")
        if not isinstance(value, int):  # pragma: no cover - defensive
            raise TypeError("HTTP status mapping must be an integer")
        return value

    @classmethod
    def to_jsonrpc_error(cls, code: str) -> dict[str, object]:
        """Translate a canonical code to a JSON-RPC error payload (not implemented)."""
        value = cls._lookup(code, cls._JSONRPC_ERROR_MAP, context="JSON-RPC error")
        if not isinstance(value, dict):  # pragma: no cover - defensive
            raise TypeError("JSON-RPC mapping must be a dictionary")
        return value.copy()
