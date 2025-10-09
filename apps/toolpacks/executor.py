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


@dataclass(frozen=True)
class ExecutionStats:
    cache_hit: bool
    duration_ms: float
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class _CacheEntry:
    payload: dict[str, Any]
    output_bytes: int


class ToolpackExecutionError(Exception):
    """Raised when executing a Toolpack fails."""


class Executor:
    """Execute Toolpack definitions for supported execution kinds."""

    def __init__(self) -> None:
        self._cache: dict[str, _CacheEntry] = {}
        self._last_stats: ExecutionStats | None = None

    @property
    def last_run_stats(self) -> ExecutionStats | None:
        return self._last_stats

    def run_toolpack(self, toolpack: Toolpack, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute ``toolpack`` with ``payload`` and return the validated output."""

        self._last_stats = None
        execution_kind = toolpack.execution.get("kind")
        if execution_kind != "python":
            raise ToolpackExecutionError(
                f"Unsupported execution kind '{execution_kind}' for toolpack {toolpack.id}"
            )

        input_payload = _ensure_mapping(payload, stage="input", toolpack=toolpack)
        input_bytes = _payload_size(input_payload, stage="input", toolpack=toolpack)
        max_input_bytes = int(toolpack.limits.get("maxInputBytes", 0))
        if max_input_bytes and input_bytes > max_input_bytes:
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=0.0,
                input_bytes=input_bytes,
                output_bytes=0,
            )
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} input size {input_bytes} exceeds "
                f"maxInputBytes {max_input_bytes}"
            )
        try:
            _validate_instance(
                schema=toolpack.input_schema,
                instance=input_payload,
                stage="input",
                toolpack=toolpack,
            )
        except ToolpackExecutionError:
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=0.0,
                input_bytes=input_bytes,
                output_bytes=0,
            )
            raise

        cache_key = self._cache_key(toolpack, input_payload)
        if toolpack.deterministic:
            entry = self._cache.get(cache_key)
            if entry is not None:
                self._last_stats = ExecutionStats(
                    cache_hit=True,
                    duration_ms=0.0,
                    input_bytes=input_bytes,
                    output_bytes=entry.output_bytes,
                )
                return copy.deepcopy(entry.payload)

        runner = self._resolve_python_callable(toolpack)
        start_time = time.perf_counter()
        try:
            result = runner(copy.deepcopy(input_payload))
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
        except Exception as exc:  # pragma: no cover - execution failure path
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=duration_ms,
                input_bytes=input_bytes,
                output_bytes=0,
            )
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} execution failed: {exc}"
            ) from exc

        output_payload = _ensure_mapping(result, stage="output", toolpack=toolpack)
        try:
            _validate_instance(
                schema=toolpack.output_schema,
                instance=output_payload,
                stage="output",
                toolpack=toolpack,
            )
        except ToolpackExecutionError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=duration_ms,
                input_bytes=input_bytes,
                output_bytes=0,
            )
            raise
        try:
            output_bytes = _payload_size(output_payload, stage="output", toolpack=toolpack)
        except ToolpackExecutionError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=duration_ms,
                input_bytes=input_bytes,
                output_bytes=0,
            )
            raise

        max_output_bytes = int(toolpack.limits.get("maxOutputBytes", 0))
        if max_output_bytes and output_bytes > max_output_bytes:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_stats = ExecutionStats(
                cache_hit=False,
                duration_ms=duration_ms,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
            )
            raise ToolpackExecutionError(
                f"Toolpack {toolpack.id} output size {output_bytes} exceeds "
                f"maxOutputBytes {max_output_bytes}"
            )

        materialised = copy.deepcopy(output_payload)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_stats = ExecutionStats(
            cache_hit=False,
            duration_ms=duration_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
        if toolpack.deterministic:
            self._cache[cache_key] = _CacheEntry(
                payload=copy.deepcopy(materialised),
                output_bytes=output_bytes,
            )
            return materialised
        return materialised

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


def _payload_size(
    payload: Mapping[str, Any], *, stage: str, toolpack: Toolpack
) -> int:
    try:
        serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ToolpackExecutionError(
            f"Toolpack {toolpack.id} {stage} payload is not JSON serialisable"
        ) from exc
    return len(serialised.encode("utf-8"))
