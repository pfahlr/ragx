"""Sample callables used by Toolpack-related unit tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def echo(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of the payload for deterministic assertions."""

    return {"echo": dict(payload)}


def uppercase(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an uppercased version of a ``text`` field."""

    text = str(payload.get("text", ""))
    return {"original": text, "upper": text.upper()}
