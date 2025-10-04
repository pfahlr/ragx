"""Property-based executable spec for envelope validation robustness."""

from __future__ import annotations

import functools
import importlib
import inspect
import itertools
import sys
import types
from pathlib import Path

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

try:
    _hypothesis_module = importlib.import_module("hypothesis")
except ModuleNotFoundError:  # pragma: no cover - fallback path
    _hypothesis_module = types.ModuleType("hypothesis")
    sys.modules.setdefault("hypothesis", _hypothesis_module)
_given = getattr(_hypothesis_module, "given", None)
_strategies = getattr(_hypothesis_module, "strategies", None)

if _given is None or _strategies is None:  # pragma: no cover - fallback path
    class _Strategy:
        def __init__(self, samples: list[str]) -> None:
            self._samples = samples

        def filter(self, predicate):
            filtered = [value for value in self._samples if predicate(value)]
            return _Strategy(filtered or self._samples[:1])

    def _text(*, min_size: int = 0, max_size: int | None = None) -> _Strategy:
        candidates = [
            "id",
            "jsonrpc",
            "method",
            "params",
            "missing",
            "invalid",
            "",
        ]
        samples = [value for value in candidates if len(value) >= min_size]
        if max_size is not None:
            samples = [value for value in samples if len(value) <= max_size]
        return _Strategy(samples or ["invalid"])

    def _given_fallback(*strategies: _Strategy):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                pools = [strategy._samples for strategy in strategies]
                for values in itertools.product(*pools):
                    fn(*values, *args, **kwargs)

            original_params = list(inspect.signature(fn).parameters.values())
            retained_params = original_params[len(strategies) :]
            wrapper.__signature__ = inspect.Signature(parameters=retained_params)
            return wrapper

        return decorator

    _strategies = types.SimpleNamespace(text=_text)
    _given = _given_fallback
    _hypothesis_module.given = _given  # type: ignore[attr-defined]
    _hypothesis_module.strategies = _strategies  # type: ignore[attr-defined]

given = _given
st = _strategies

SCHEMAS = Path("codex/specs/schemas")
_REQUIRED_FIELDS = ("id", "jsonrpc", "method", "params")


def _mutate_payload(field: str, marker: str) -> dict[str, object]:
    """Create an invalid envelope payload based on the selected field."""

    payload: dict[str, object] = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "jsonrpc": "2.0",
        "method": "mcp.tool.invoke",
        "params": {"echo": "hello"},
    }

    if field == "id":
        if len(marker) % 2 == 0:
            payload.pop("id")
        else:
            payload["id"] = ""
    elif field == "jsonrpc":
        candidate = marker.strip() or "1.0"
        if candidate == "2.0":
            candidate = "1.0"
        payload["jsonrpc"] = candidate
    elif field == "method":
        payload["method"] = marker.strip()
        if payload["method"]:
            payload.pop("method")
    elif field == "params":
        payload["params"] = marker
    else:  # pragma: no cover - defensive fallback
        payload.pop(field, None)

    return payload


@pytest.mark.xfail(
    reason="Envelope schema validation not implemented",
    raises=NotImplementedError,
    strict=True,
)
@given(
    st.text(min_size=1).filter(lambda name: name in _REQUIRED_FIELDS),
    st.text(max_size=32),
)
def test_envelope_validator_rejects_fuzzed_payloads(field: str, marker: str) -> None:
    """The schema validator must reject malformed payloads under fuzzing."""

    payload = _mutate_payload(field, marker)
    registry = SchemaRegistry(schema_root=SCHEMAS)
    validator = registry.load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)
