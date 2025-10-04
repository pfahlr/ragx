"""Canonical error mapping stub for MCP services.

The executable specs added in task 06cV2A import this module to define the
expected behaviour for HTTP and JSON-RPC error mapping. The real implementation
will populate these helpers in the next task.
"""
from __future__ import annotations

_CANONICAL_CODES = {
    "INVALID_ENVELOPE",
    "INVALID_TOOL_INPUT",
    "TOOL_NOT_FOUND",
    "INTERNAL_ERROR",
}


class CanonicalError:
    """Stub exposing the canonical error mapping helpers."""

    @staticmethod
    def to_http_status(code: str) -> int:
        """Return the HTTP status code for ``code``.

        Raises ``NotImplementedError`` until the concrete mapping is provided.
        """

        raise NotImplementedError("HTTP status mapping not implemented (06cV2A scaffold)")

    @staticmethod
    def to_jsonrpc_error(code: str) -> dict[str, object]:
        """Return the JSON-RPC error payload for ``code``."""

        raise NotImplementedError(
            "JSON-RPC error mapping not implemented (06cV2A scaffold)"
        )


__all__ = ["CanonicalError", "_CANONICAL_CODES"]
