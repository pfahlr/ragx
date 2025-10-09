from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    numbers = payload.get("numbers", [])
    if not isinstance(numbers, Iterable):
        raise TypeError("numbers must be iterable")
    total = 0
    coerced: list[float] = []
    for value in numbers:
        coerced.append(float(value))
        total += float(value)
    sleep_ms = payload.get("sleep_ms")
    if isinstance(sleep_ms, (int | float)) and sleep_ms > 0:
        time.sleep(float(sleep_ms) / 1000.0)
    output_bytes = payload.get("output_bytes")
    blob: str | None = None
    if isinstance(output_bytes, (int | float)) and output_bytes > 0:
        blob = "x" * int(output_bytes)
    result: dict[str, Any] = {"total": total, "numbers": coerced}
    if blob is not None:
        result["blob"] = blob
    return result
