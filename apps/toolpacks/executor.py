from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import validators
from jsonschema.exceptions import SchemaError, ValidationError

from apps.toolpacks.loader import Toolpack, ToolpackValidationError

__all__ = ["Executor", "ExecutionStats", "ToolpackExecutionError"]


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """Immutable snapshot of the most recent tool invocation."""

    duration_ms: float
    input_bytes: int
    output_bytes: int
    cache_hit: bool


class ToolpackExecutionError(Exception):
    """Raised when executing a Toolpack fails."""


class Executor:
    """Execute Toolpack definitions for supported execution kinds."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_stats: ExecutionStats | None = None

    def run_toolpack(self, toolpack: Toolpack, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute ``toolpack`` with ``payload`` and return the validated output."""

        execution_kind = toolpack.execution.get("kind")
        if execution_kind != "python":
            raise ToolpackExecutionError(
                f"Unsupported execution kind '{execution_kind}' for toolpack {toolpack.id}"
            )

        input_payload = _ensure_mapping(payload, stage="input", toolpack=toolpack)
        cache_key = self._cache_key(toolpack, input_payload)
        input_bytes = _canonical_size(input_payload)
        _validate_instance(
            schema=toolpack.input_schema,
            instance=input_payload,
            stage="input",
            toolpack=toolpack,
        )

        if toolpack.deterministic:
            cached = self._cache.get(cache_key)
            if cached is not None:
                output_bytes = _canonical_size(cached)
                self._last_stats = ExecutionStats(
                    duration_ms=0.0,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    cache_hit=True,
                )
                return copy.deepcopy(cached)

        runner = self._resolve_python_callable(toolpack)
        start = time.perf_counter()
        try:
            result = runner(copy.deepcopy(input_payload))
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
        except Exception as exc:  # pragma: no cover - execution failure path
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} execution failed: {exc}"
            ) from exc

        output_payload = _ensure_mapping(result, stage="output", toolpack=toolpack)
        _validate_instance(
            schema=toolpack.output_schema,
            instance=output_payload,
            stage="output",
            toolpack=toolpack,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        materialised = copy.deepcopy(output_payload)
        output_bytes = _canonical_size(materialised)

        stats = ExecutionStats(
            duration_ms=duration_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            cache_hit=False,
        )
        self._last_stats = stats

        if toolpack.deterministic:
            self._cache[cache_key] = copy.deepcopy(materialised)
            return copy.deepcopy(materialised)
        return materialised

    def last_run_stats(self) -> ExecutionStats | None:
        """Return metrics from the most recent invocation, if any."""

        return self._last_stats

    def _resolve_python_callable(self, toolpack: Toolpack) -> Callable[[Mapping[str, Any]], Any]:
        execution = toolpack.execution
        entrypoint = execution.get("module")
        if not isinstance(entrypoint, str) or not entrypoint:
            message = (
                f"Toolpack {toolpack.id} python execution requires module entrypoint "
                "'pkg.mod:func'"
            )
            raise ToolpackExecutionError(message)

        module_name, sep, attr_name = entrypoint.partition(":")
        if not sep or not module_name or not attr_name:
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} python module entrypoint must use 'module:callable'"
            )

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import errors rely on Python
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} failed to import module '{module_name}': {exc}"
            ) from exc

        try:
            func = getattr(module, attr_name)
        except AttributeError as exc:
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} module '{module_name}' has no attribute '{attr_name}'"
            ) from exc

        if not callable(func):
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} attribute '{attr_name}' is not callable"
            )

        return func

    def _cache_key(self, toolpack: Toolpack, payload: Mapping[str, Any]) -> str:
        envelope = {
            "id": toolpack.id,
            "version": toolpack.version,
            "payload": payload,
        }
        try:
            serialised = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        except TypeError as exc:
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} input is not JSON serialisable for caching"
            ) from exc
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _ensure_mapping(value: Any, *, stage: str, toolpack: Toolpack) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolpackExecutionError(
            f"Toolpack {toolpack.id} {stage} payload must be a mapping"
        )
    return dict(value)


def _canonical_size(payload: Mapping[str, Any]) -> int:
    try:
        serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError as exc:  # pragma: no cover - payload already validated
        raise ToolpackExecutionError("Payload is not JSON serialisable") from exc
    return len(serialised)


def _validate_instance(
    *,
    schema: Mapping[str, Any],
    instance: Mapping[str, Any],
    stage: str,
    toolpack: Toolpack,
) -> None:
    try:
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        validator.validate(instance)
    except SchemaError as exc:  # pragma: no cover - schema already validated by loader
        raise ToolpackValidationError(
            f"Toolpack {toolpack.id} schema failed validation: {exc.message}"
        ) from exc
    except ValidationError as exc:
        raise ToolpackExecutionError(
            f"Toolpack {toolpack.id} {stage} failed JSON schema validation: {exc.message}"
        ) from exc
