"""Deterministic test runtime that sums provided values."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sum of ``values`` with optional delay and text payload."""

    values = payload.get("values", [])
    if not isinstance(values, Sequence):
        raise TypeError("values must be a sequence")
    try:
        total = sum(float(value) for value in values)
    except TypeError as exc:  # pragma: no cover - defensive guard for invalid payloads
        raise TypeError("values must be numeric") from exc

    delay_ms = payload.get("delayMs", 0)
    if delay_ms:
        time.sleep(float(delay_ms) / 1000.0)

    text = payload.get("text", "")
    if not isinstance(text, str):  # pragma: no cover - schema should reject, guard for safety
        raise TypeError("text must be a string")

    response: dict[str, Any] = {"sum": total}

    output_size = payload.get("outputSize")
    if output_size is not None:
        response["text"] = "y" * int(output_size)
    elif text:
        response["text"] = text
    return response
