from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["CanonicalError"]


@dataclass(frozen=True)
class _CanonicalSpec:
    code: str
    description: str
    retryable: bool
    http_status: int
    jsonrpc_code: int
    message: str


@dataclass(frozen=True)
class _JsonRpcErrorTemplate:
    """Immutable template describing JSON-RPC error payload fields."""

    code: int
    message: str
    retryable: bool

    def build_payload(self, *, canonical_code: str, http_status: int) -> dict[str, object]:
        """Materialise a JSON-RPC error payload without sharing state."""

        return {
            "code": self.code,
            "message": self.message,
            "data": {
                "canonical": canonical_code,
                "httpStatus": http_status,
                "message": self.message,
                "retryable": bool(self.retryable),
            },
        }


class CanonicalError:
    """Canonical error codes used across transports."""

    _SPECS: tuple[_CanonicalSpec, ...] = (
        _CanonicalSpec(
            "INVALID_INPUT",
            "Request payload failed validation",
            False,
            400,
            -32602,
            "Invalid input payload",
        ),
        _CanonicalSpec(
            "INVALID_OUTPUT",
            "Tool produced output that failed validation",
            False,
            502,
            -32002,
            "Invalid output payload",
        ),
        _CanonicalSpec(
            "NOT_FOUND",
            "Requested resource was not found",
            False,
            404,
            -32004,
            "Resource not found",
        ),
        _CanonicalSpec(
            "TIMEOUT",
            "Tool invocation exceeded allotted time",
            True,
            504,
            -32003,
            "Tool execution timed out",
        ),
        _CanonicalSpec(
            "UNAUTHORIZED",
            "Authentication or authorization failed",
            False,
            401,
            -32001,
            "Unauthorized",
        ),
        _CanonicalSpec(
            "INTERNAL_ERROR",
            "Unexpected server-side failure",
            True,
            500,
            -32603,
            "Internal server error",
        ),
    )

    _HTTP_STATUS_MAP: dict[str, int] = {spec.code: spec.http_status for spec in _SPECS}
    _JSONRPC_MAP: dict[str, _JsonRpcErrorTemplate] = {
        spec.code: _JsonRpcErrorTemplate(
            code=spec.jsonrpc_code,
            message=spec.message,
            retryable=spec.retryable,
        )
        for spec in _SPECS
    }

    @classmethod
    def codes(cls) -> Sequence[str]:
        return tuple(spec.code for spec in cls._SPECS)

    @staticmethod
    def _lookup(code: str, mapping: Mapping[str, object], *, context: str | None = None) -> object:
        if code not in mapping:
            raise KeyError(f"{code} does not have a mapping for {context or 'requested lookup'}")
        return mapping[code]

    @classmethod
    def to_http_status(cls, code: str) -> int:
        value = cls._lookup(code, cls._HTTP_STATUS_MAP, context="HTTP status")
        if not isinstance(value, int):  # pragma: no cover - defensive
            raise TypeError("HTTP status mapping must be an integer")
        return value

    @classmethod
    def to_jsonrpc_error(cls, code: str) -> dict[str, object]:
        template = cls._lookup(code, cls._JSONRPC_MAP, context="JSON-RPC error")
        if not isinstance(template, _JsonRpcErrorTemplate):  # pragma: no cover - defensive
            raise TypeError("JSON-RPC mapping must be a _JsonRpcErrorTemplate")
        http_status = cls.to_http_status(code)
        return template.build_payload(canonical_code=code, http_status=http_status)
