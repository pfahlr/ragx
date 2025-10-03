"""Structured logging utilities for the MCP server."""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "ragx.mcp_server"

logger = logging.getLogger(LOGGER_NAME)


def log_event(*, trace_id: str, transport: str, status: str, **extra: Any) -> None:
    """Emit a JSON log line describing an MCP request."""

    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "transport": transport,
        "status": status,
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    logger.info(json.dumps(payload, sort_keys=True))
