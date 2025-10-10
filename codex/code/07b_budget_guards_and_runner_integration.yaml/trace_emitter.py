"""Schema-validating trace event emitter."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


class TraceSchemaError(ValueError):
    """Raised when a trace payload violates the schema."""


_TRACE_SCHEMA: Dict[str, Dict[str, Iterable[str]]] = {
    "policy_push": {"required": ("policy_id", "node_id")},
    "policy_resolved": {"required": ("policy_id", "node_id", "status")},
    "policy_violation": {"required": ("policy_id", "node_id", "reason")},
    "budget_preflight": {
        "required": (
            "scope",
            "scope_id",
            "stage",
            "attempted_ms",
            "remaining_ms",
            "action",
        ),
        "optional": ("allowed", "limit_ms", "breach"),
    },
    "budget_charge": {
        "required": (
            "scope",
            "scope_id",
            "stage",
            "attempted_ms",
            "remaining_ms",
            "action",
            "allowed",
        ),
        "optional": ("limit_ms", "breach"),
    },
    "budget_breach": {
        "required": (
            "scope",
            "scope_id",
            "stage",
            "attempted_ms",
            "remaining_ms",
            "action",
        ),
        "optional": ("limit_ms", "breach"),
    },
}


def _validate_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise TraceSchemaError(f"Field '{field}' must be numeric")
    return float(value)


def _validate_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TraceSchemaError(f"Field '{field}' must be a string")
    return value


def _validate_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TraceSchemaError(f"Field '{field}' must be a boolean")
    return value


def _normalize_breach(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceSchemaError("Breach payload must be a mapping")
    required = {"scope", "scope_id", "attempted_ms", "limit_ms", "remaining_ms", "action"}
    if not required.issubset(value.keys()):
        missing = sorted(required.difference(value.keys()))
        raise TraceSchemaError(f"Breach payload missing fields: {missing}")
    normalized: Dict[str, Any] = {
        "scope": _validate_string(value["scope"], "breach.scope"),
        "scope_id": _validate_string(value["scope_id"], "breach.scope_id"),
        "attempted_ms": _validate_number(value["attempted_ms"], "breach.attempted_ms"),
        "limit_ms": _validate_number(value["limit_ms"], "breach.limit_ms"),
        "remaining_ms": _validate_number(value["remaining_ms"], "breach.remaining_ms"),
        "action": _validate_string(value["action"], "breach.action"),
    }
    return MappingProxyType(normalized)


class TraceEventEmitter:
    """Collects trace events while enforcing schema compliance."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Mapping[str, Any]]] = []

    @property
    def events(self) -> List[Mapping[str, Any]]:
        return list(self._events)

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if event_type not in _TRACE_SCHEMA:
            raise TraceSchemaError(f"Unknown trace event type: {event_type}")
        normalized_payload = self._normalize_payload(event_type, payload)
        record: Dict[str, Mapping[str, Any]] = {
            "type": event_type,
            "payload": MappingProxyType(normalized_payload),
        }
        self._events.append(record)
        return record

    def clear(self) -> None:
        self._events.clear()

    def _normalize_payload(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> MutableMapping[str, Any]:
        schema = _TRACE_SCHEMA[event_type]
        normalized: Dict[str, Any] = {}
        for field in schema.get("required", ()):  # type: ignore[arg-type]
            if field not in payload:
                raise TraceSchemaError(f"Missing required field '{field}' for {event_type}")
            normalized[field] = self._coerce(field, payload[field])
        for field in schema.get("optional", ()):  # type: ignore[arg-type]
            if field in payload:
                normalized[field] = self._coerce(field, payload[field])
        # Preserve ordering for deterministic assertions.
        ordered = dict((key, normalized[key]) for key in normalized)
        return ordered

    def _coerce(self, field: str, value: Any) -> Any:
        if field in {"scope", "scope_id", "stage", "action", "policy_id", "node_id", "status", "reason"}:
            return _validate_string(value, field)
        if field in {"attempted_ms", "remaining_ms", "limit_ms"}:
            return _validate_number(value, field)
        if field == "allowed":
            return _validate_boolean(value, field)
        if field == "breach":
            return _normalize_breach(value)
        return value
