from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import time
from contextvars import ContextVar
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import validators
from jsonschema.exceptions import SchemaError, ValidationError

from apps.toolpacks.loader import Toolpack, ToolpackValidationError

__all__ = ["ExecutionStats", "Executor", "ToolpackExecutionError"]


@dataclass(slots=True)
class ExecutionStats:
    """Structured execution metrics recorded for the last tool invocation."""

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
        self._last_stats: ContextVar[ExecutionStats | None] = ContextVar(
            f"toolpack_executor_last_stats_{id(self)}",
            default=None,
        )

    def last_run_stats(self) -> ExecutionStats | None:
        """Return metrics for the most recent invocation."""

        return self._last_stats.get()

    def run_toolpack(self, toolpack: Toolpack, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute ``toolpack`` with ``payload`` and return the validated output."""

        result, _ = self.run_toolpack_with_stats(toolpack, payload)
        return result

    def run_toolpack_with_stats(
        self,
        toolpack: Toolpack,
        payload: Mapping[str, Any],
        *,
        use_cache: bool = True,
    ) -> tuple[dict[str, Any], ExecutionStats]:
        """Execute ``toolpack`` with ``payload`` and return result + metrics."""

        start_time = time.perf_counter()
        execution_kind = toolpack.execution.get("kind")
        if execution_kind != "python":
            raise ToolpackExecutionError(
                f"Unsupported execution kind '{execution_kind}' for toolpack {toolpack.id}"
            )

        input_payload = _ensure_mapping(payload, stage="input", toolpack=toolpack)
        _validate_instance(
            schema=toolpack.input_schema,
            instance=input_payload,
            stage="input",
            toolpack=toolpack,
        )

        cache_key = self._cache_key(toolpack, input_payload)
        input_bytes = _payload_size(input_payload)
        cache_enabled = use_cache and toolpack.deterministic
        if cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                output_bytes = _payload_size(cached)
                duration_ms = _elapsed_ms(start_time)
                stats = ExecutionStats(
                    duration_ms=duration_ms,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    cache_hit=True,
                )
                self._last_stats.set(stats)
                return copy.deepcopy(cached), stats

        runner = self._resolve_python_callable(toolpack)
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

        materialised = copy.deepcopy(output_payload)
        output_bytes = _payload_size(materialised)
        duration_ms = _elapsed_ms(start_time)
        stats = ExecutionStats(
            duration_ms=duration_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            cache_hit=False,
        )
        self._last_stats.set(stats)
        if toolpack.deterministic:
            if cache_enabled:
                self._cache[cache_key] = copy.deepcopy(materialised)
            return copy.deepcopy(materialised), stats
        return materialised, stats

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


def _payload_size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
