"""Stub canonical error helpers for Part A tests."""

from __future__ import annotations

from typing import Any, ClassVar


class CanonicalError:
    """Static helpers mapping canonical codes to transport-specific payloads."""

    ALLOWED_CODES: ClassVar[set[str]] = {
        "INVALID_INPUT",
        "INVALID_OUTPUT",
        "NOT_FOUND",
        "UNAUTHORIZED",
        "INTERNAL_ERROR",
    }

    @staticmethod
    def to_http_status(code: str) -> int:  # pragma: no cover - stub
        raise NotImplementedError("CanonicalError.to_http_status will be implemented in Part B")

    @staticmethod
    def to_jsonrpc_error(code: str) -> dict[str, Any]:  # pragma: no cover - stub
        raise NotImplementedError("CanonicalError.to_jsonrpc_error will be implemented in Part B")
