"""Service layer for the MCP server (lazy exports)."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "Envelope",
    "EnvelopeError",
    "EnvelopeMeta",
    "McpService",
    "RequestContext",
    "ServerLogManager",
]

_EXPORT_MAP = {
    "Envelope": "apps.mcp_server.service.envelope",
    "EnvelopeError": "apps.mcp_server.service.envelope",
    "EnvelopeMeta": "apps.mcp_server.service.envelope",
    "McpService": "apps.mcp_server.service.mcp_service",
    "RequestContext": "apps.mcp_server.service.mcp_service",
    "ServerLogManager": "apps.mcp_server.service.mcp_service",
}

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .envelope import Envelope, EnvelopeError, EnvelopeMeta
    from .mcp_service import McpService, RequestContext, ServerLogManager


def __getattr__(name: str):  # pragma: no cover - thin loader
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MAP[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - thin loader
    return sorted(set(globals()) | set(__all__))
