"""Sample callables used by Toolpack-related unit tests."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


def echo(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of the payload for deterministic assertions."""

    return {"echo": dict(payload)}


def uppercase(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an uppercased version of a ``text`` field."""

    text = str(payload.get("text", ""))
    return {"original": text, "upper": text.upper()}


def delayed_echo(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sleep for ``delayMs`` before mirroring the payload."""

    delay_ms = float(payload.get("delayMs", 0))
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    return {"echo": dict(payload)}
