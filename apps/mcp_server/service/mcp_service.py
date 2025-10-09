from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

from jsonschema import Draft202012Validator, ValidationError, validators

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.logging import resolve_tool_invocation_log_paths
from apps.mcp_server.validation import SchemaRegistry
from apps.mcp_server.validation.logging import (
    EnvelopeValidationEvent,
    EnvelopeValidationLogManager,
)
from apps.toolpacks.executor import ExecutionStats, Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack, ToolpackLoader

from .envelope import Envelope, EnvelopeError, EnvelopeMeta

__all__ = ["McpService", "RequestContext", "ServerLogManager", "ServiceLimits"]

_AGENT_ID = "mcp_server"
_TASK_ID = "06b_mcp_server_bootstrap"
_SCHEMA_VERSION_DEFAULT = "0.1.0"
_DETERMINISTIC_NAMESPACE = UUID("c1fd1c20-77b7-4f73-b39c-8ed2dd2f2d8c")


class ValidationMode(Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @classmethod
    def from_str(cls, raw: str | None) -> ValidationMode:
        if not raw:
            return cls.SHADOW
        normalised = raw.strip().lower()
        for mode in cls:
            if mode.value == normalised:
                return mode
        return cls.SHADOW


@dataclass(slots=True)
class RequestContext:
    """Per-request metadata supplied by transports."""

    transport: str
    route: str
    method: str
    deterministic_ids: bool = False
    request_payload: Mapping[str, Any] | None = None
    start_time: float = field(default_factory=time.perf_counter)
    attempt: int = 0
    _cached_ids: dict[str, Any] | None = field(default=None, repr=False, init=False)

    def clone_with_payload(self, payload: Mapping[str, Any]) -> RequestContext:
        return RequestContext(
            transport=self.transport,
            route=self.route,
            method=self.method,
            deterministic_ids=self.deterministic_ids,
            request_payload=payload,
            start_time=time.perf_counter(),
            attempt=self.attempt,
        )


@dataclass(frozen=True, slots=True)
class ServiceLimits:
    """Global guardrail configuration for tool invocations."""

    max_input_bytes: int = 1_048_576
    max_output_bytes: int = 2_097_152
    timeout_ms: int = 15_000

    def __post_init__(self) -> None:
        if self.max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive integer")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")


_DEFAULT_SERVICE_LIMITS = ServiceLimits()


@dataclass(slots=True)
class Prompt:
    prompt_id: str
    name: str
    description: str
    major: int
    tags: list[str]
    messages: list[dict[str, str]]

    def to_discovery_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "version": {"major": self.major},
            "tags": list(self.tags),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "version": {"major": self.major},
            "messages": list(self.messages),
        }


class PromptRepository:
    """Load prompt definitions from disk."""

    def __init__(self, base_dir: Path) -> None:
        self._prompts: dict[str, Prompt] = {}
        self._load(base_dir)

    def _load(self, base_dir: Path) -> None:
        directory = Path(base_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Prompts directory not found: {directory}")
        prompts: dict[str, Prompt] = {}
        for path in sorted(directory.rglob("*.prompt.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            prompt = _prompt_from_payload(data, path)
            if prompt.prompt_id in prompts:
                raise ValueError(f"Duplicate prompt id '{prompt.prompt_id}' in {path}")
            prompts[prompt.prompt_id] = prompt
        self._prompts = prompts

    def list(self) -> list[Prompt]:
        return list(self._prompts.values())

    def get(self, prompt_id: str) -> Prompt:
        if prompt_id not in self._prompts:
            raise KeyError(f"Unknown prompt '{prompt_id}'")
        return self._prompts[prompt_id]


class SchemaStore:
    """JSON schema loader with caching."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._cache: dict[Path, Draft202012Validator] = {}

    def validator(self, name: str) -> Draft202012Validator:
        path = self._base_dir / name
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        self._cache[path] = validator
        return validator


@dataclass(slots=True)
class ServerLogEvent:
    ts: datetime
    request_id: str
    trace_id: str
    span_id: str
    transport: str
    route: str
    method: str
    status: str
    attempt: int
    execution: dict[str, Any]
    idempotency: dict[str, Any]
    metadata: dict[str, Any]
    step_id: int
    error: dict[str, Any] | None = None

    def to_serialisable(self) -> dict[str, Any]:
        return {
            "ts": self.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "agentId": _AGENT_ID,
            "taskId": _TASK_ID,
            "stepId": self.step_id,
            "transport": self.transport,
            "route": self.route,
            "method": self.method,
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "requestId": self.request_id,
            "status": self.status,
            "attempt": self.attempt,
            "execution": dict(self.execution),
            "idempotency": dict(self.idempotency),
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class ServerLogManager:
    """Wrapper around :class:`JsonLogWriter` for MCP server events."""

    def __init__(
        self,
        *,
        log_dir: Path,
        schema_version: str,
        deterministic: bool,
        retention: int = 5,
    ) -> None:
        paths = resolve_tool_invocation_log_paths(Path(log_dir))
        paths.ensure_directories()
        self._paths = paths
        self._writer = JsonLogWriter(
            agent_id=_AGENT_ID,
            task_id=_TASK_ID,
            storage_prefix=self._paths.storage_prefix,
            latest_symlink=self._paths.latest_symlink,
            schema_version=schema_version,
            deterministic=deterministic,
            root_dir=self._paths.root,
            retention=retention,
        )
        self._step_counter = 0

    @property
    def latest_symlink(self) -> Path:
        return self._paths.latest_symlink

    @property
    def writer(self) -> JsonLogWriter:
        return self._writer

    def next_step_id(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def emit(self, event: ServerLogEvent) -> None:
        attempt_id = self._writer.new_attempt_id()
        self._writer.write(event, attempt_id=attempt_id)


class McpService:
    """High-level service implementing discovery, prompts, and tool invocation."""

    def __init__(
        self,
        *,
        toolpacks: dict[str, Toolpack],
        executor: Executor,
        prompts: PromptRepository,
        schema_store: SchemaStore,
        log_manager: ServerLogManager,
        schema_version: str,
        validation_registry: SchemaRegistry,
        validation_log: EnvelopeValidationLogManager,
        validation_mode: ValidationMode,
        limits: ServiceLimits,
    ) -> None:
        self._toolpacks = toolpacks
        self._executor = executor
        self._prompts = prompts
        self._schemas = schema_store
        self._log_manager = log_manager
        self._schema_version = schema_version
        self._validation_registry = validation_registry
        self._validation_log = validation_log
        self._validation_mode = validation_mode
        self._limits = limits

    @property
    def log_manager(self) -> ServerLogManager:
        return self._log_manager

    @classmethod
    def create(
        cls,
        *,
        toolpacks_dir: Path,
        prompts_dir: Path,
        schema_dir: Path,
        log_dir: Path,
        schema_version: str = _SCHEMA_VERSION_DEFAULT,
        deterministic_logs: bool = True,
        logger: ServerLogManager | None = None,
        max_input_bytes: int | None = None,
        max_output_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> McpService:
        loader = ToolpackLoader()
        loader.load_dir(toolpacks_dir)
        toolpacks = {pack.id: pack for pack in loader.list()}
        executor = Executor()
        prompts = PromptRepository(prompts_dir)
        schema_store = SchemaStore(schema_dir)
        log_manager = logger or ServerLogManager(
            log_dir=log_dir,
            schema_version=schema_version,
            deterministic=deterministic_logs,
        )
        validation_registry = SchemaRegistry()
        validation_log = EnvelopeValidationLogManager(
            log_dir=log_dir,
            schema_version=schema_version,
            deterministic=deterministic_logs,
        )
        validation_mode = ValidationMode.from_str(
            os.getenv("RAGX_MCP_ENVELOPE_VALIDATION")
        )
        base_limits = _DEFAULT_SERVICE_LIMITS
        limits = ServiceLimits(
            max_input_bytes=(
                max_input_bytes
                if max_input_bytes is not None
                else base_limits.max_input_bytes
            ),
            max_output_bytes=(
                max_output_bytes
                if max_output_bytes is not None
                else base_limits.max_output_bytes
            ),
            timeout_ms=(
                timeout_ms if timeout_ms is not None else base_limits.timeout_ms
            ),
        )
        return cls(
            toolpacks=toolpacks,
            executor=executor,
            prompts=prompts,
            schema_store=schema_store,
            log_manager=log_manager,
            schema_version=schema_version,
            validation_registry=validation_registry,
            validation_log=validation_log,
            validation_mode=validation_mode,
            limits=limits,
        )

    def discover(self, context: RequestContext | None = None) -> Envelope:
        payload: dict[str, Any] = {}
        ctx = self._normalise_context(context, "discover", "mcp.discover", payload)
        tools = [self._tool_to_dict(tool) for tool in self._toolpacks.values()]
        prompts = [prompt.to_discovery_dict() for prompt in self._prompts.list()]
        data = {"tools": tools, "prompts": prompts}
        self._schemas.validator("discover.response.schema.json").validate(data)
        envelope = self._finalise_envelope(data, ctx, tool_id=None, prompt_id=None)
        return envelope

    def get_prompt(self, prompt_id: str, context: RequestContext | None = None) -> Envelope:
        payload = {"promptId": prompt_id}
        ctx = self._normalise_context(context, "prompt", "mcp.prompt.get", payload)
        try:
            prompt = self._prompts.get(prompt_id)
        except KeyError:
            return self._error_response(
                code="NOT_FOUND",
                message=f"Prompt '{prompt_id}' not found",
                context=ctx,
                payload=payload,
                prompt_id=prompt_id,
            )
        data = prompt.to_payload()
        self._schemas.validator("prompt.response.schema.json").validate(data)
        return self._finalise_envelope(data, ctx, prompt_id=prompt_id)

    def invoke_tool(
        self,
        *,
        tool_id: str,
        arguments: Mapping[str, Any],
        context: RequestContext | None = None,
    ) -> Envelope:
        payload = {"toolId": tool_id, "arguments": dict(arguments)}
        ctx = self._normalise_context(context, "tool", "mcp.tool.invoke", payload)
        ids = self._request_ids(ctx)
        if tool_id not in self._toolpacks:
            return self._error_response(
                code="NOT_FOUND",
                message=f"Tool '{tool_id}' not found",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
            )
        toolpack = self._toolpacks[tool_id]
        tool_input_limit = int(
            toolpack.limits.get("maxInputBytes", self._limits.max_input_bytes)
        )
        effective_input_limit = min(self._limits.max_input_bytes, tool_input_limit)
        if ids["input_bytes"] > effective_input_limit:
            return self._error_response(
                code="INVALID_INPUT",
                message="Input payload exceeds configured maxInputBytes",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
            )
        validators_bundle = None
        if self._validation_mode is not ValidationMode.OFF:
            try:
                validators_bundle = self._validation_registry.load_tool_io(tool_id)
            except KeyError:
                validators_bundle = None
            if validators_bundle is not None:
                try:
                    validators_bundle.input.validate(
                        {"tool": tool_id, "input": dict(arguments)}
                    )
                except ValidationError as exc:
                    return self._error_response(
                        code="INVALID_INPUT",
                        message=str(exc).splitlines()[0],
                        context=ctx,
                        payload=payload,
                        tool_id=tool_id,
                    )
        try:
            result = self._executor.run_toolpack(toolpack, arguments)
        except ToolpackExecutionError as exc:
            return self._error_response(
                code="INTERNAL_ERROR",
                message=str(exc),
                context=ctx,
                payload=payload,
                tool_id=tool_id,
            )
        stats = self._executor.last_run_stats()
        if stats is None:
            stats = ExecutionStats(
                duration_ms=_duration_ms(ctx.start_time),
                input_bytes=ids["input_bytes"],
                output_bytes=_payload_size(result),
                cache_hit=False,
            )
        effective_timeout = min(self._limits.timeout_ms, toolpack.timeout_ms)
        if stats.duration_ms > effective_timeout:
            return self._error_response(
                code="TIMEOUT",
                message=f"Tool execution exceeded timeout of {effective_timeout}ms",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                execution_stats=stats,
            )
        tool_output_limit = int(
            toolpack.limits.get("maxOutputBytes", self._limits.max_output_bytes)
        )
        effective_output_limit = min(self._limits.max_output_bytes, tool_output_limit)
        if stats.output_bytes > effective_output_limit:
            return self._error_response(
                code="INVALID_OUTPUT",
                message="Tool output exceeded configured maxOutputBytes",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                execution_stats=stats,
                cache_hit=stats.cache_hit,
            )
        if self._validation_mode is not ValidationMode.OFF and validators_bundle is not None:
            try:
                validators_bundle.output.validate(
                    {"tool": tool_id, "output": dict(result)}
                )
            except ValidationError as exc:
                return self._error_response(
                    code="INVALID_OUTPUT",
                    message=str(exc).splitlines()[0],
                    context=ctx,
                    payload=payload,
                    tool_id=tool_id,
                    execution_stats=stats,
                    cache_hit=stats.cache_hit,
                )
        data = {
            "toolId": tool_id,
            "result": dict(result),
            "metadata": {
                "toolpack": {
                    "id": toolpack.id,
                    "version": toolpack.version,
                    "deterministic": toolpack.deterministic,
                    "timeoutMs": toolpack.timeout_ms,
                }
            },
        }
        self._schemas.validator("tool.response.schema.json").validate(data)
        return self._finalise_envelope(
            data,
            ctx,
            tool_id=tool_id,
            execution_stats=stats,
            cache_hit=stats.cache_hit,
        )

    def health(self, context: RequestContext | None = None) -> dict[str, Any]:
        _ = self._normalise_context(context, "health", "mcp.health", {})
        return {"status": "ok"}

    # Internal helpers -------------------------------------------------

    def _execution_payload(
        self,
        *,
        duration_ms: float,
        input_bytes: int,
        output_bytes: int,
    ) -> dict[str, Any]:
        return {
            "durationMs": max(float(duration_ms), 0.0),
            "inputBytes": max(int(input_bytes), 0),
            "outputBytes": max(int(output_bytes), 0),
        }

    @staticmethod
    def _idempotency_payload(cache_hit: bool) -> dict[str, Any]:
        return {"cacheHit": bool(cache_hit)}

    def _finalise_envelope(
        self,
        data: Mapping[str, Any],
        context: RequestContext,
        *,
        tool_id: str | None = None,
        prompt_id: str | None = None,
        execution_stats: ExecutionStats | None = None,
        cache_hit: bool | None = None,
    ) -> Envelope:
        duration_ms = _duration_ms(context.start_time)
        ids = self._request_ids(context)
        step_id = self._log_manager.next_step_id()
        payload_output_bytes = _payload_size(data)
        if execution_stats is not None:
            exec_duration = execution_stats.duration_ms
            exec_input_bytes = execution_stats.input_bytes
            exec_output_bytes = execution_stats.output_bytes
            cache_flag = execution_stats.cache_hit if cache_hit is None else cache_hit
        else:
            exec_duration = duration_ms
            exec_input_bytes = ids["input_bytes"]
            exec_output_bytes = payload_output_bytes
            cache_flag = cache_hit if cache_hit is not None else False
        execution_payload = self._execution_payload(
            duration_ms=exec_duration,
            input_bytes=exec_input_bytes,
            output_bytes=exec_output_bytes,
        )
        idempotency_payload = self._idempotency_payload(cache_flag)
        meta = EnvelopeMeta.from_ids(
            request_id=ids["request_id"],
            trace_id=ids["trace_id"],
            span_id=ids["span_id"],
            schema_version=self._schema_version,
            deterministic=context.deterministic_ids,
            transport=context.transport,
            route=context.route,
            method=context.method,
            status="ok",
            attempt=context.attempt,
            execution=execution_payload,
            idempotency=idempotency_payload,
            tool_id=tool_id,
            prompt_id=prompt_id,
        )
        envelope = Envelope.success(data=dict(data), meta=meta)
        envelope_dict = envelope.model_dump(by_alias=True)
        metadata_payload = {
            key: value
            for key, value in {"toolId": tool_id, "promptId": prompt_id}.items()
            if value is not None
        }
        self._log_manager.emit(
            ServerLogEvent(
                ts=datetime.now(UTC),
                request_id=meta.request_id,
                trace_id=meta.trace_id,
                span_id=meta.span_id,
                transport=context.transport,
                route=context.route,
                method=context.method,
                status="ok",
                attempt=context.attempt,
                execution=execution_payload,
                idempotency=idempotency_payload,
                metadata=metadata_payload,
                step_id=step_id,
            )
        )
        self._validate_envelope_and_log(
            envelope_dict=envelope_dict,
            context=context,
            ids=ids,
            tool_id=tool_id,
            prompt_id=prompt_id,
            execution=execution_payload,
            idempotency=idempotency_payload,
            status="ok",
            error=None,
        )
        return envelope

    def _error_response(
        self,
        *,
        code: str,
        message: str,
        context: RequestContext,
        payload: Mapping[str, Any],
        tool_id: str | None = None,
        prompt_id: str | None = None,
        execution_stats: ExecutionStats | None = None,
        cache_hit: bool | None = None,
    ) -> Envelope:
        ids = self._request_ids(context)
        step_id = self._log_manager.next_step_id()
        duration_ms = _duration_ms(context.start_time)
        if execution_stats is not None:
            exec_duration = execution_stats.duration_ms
            exec_input_bytes = execution_stats.input_bytes
            exec_output_bytes = execution_stats.output_bytes
            cache_flag = execution_stats.cache_hit if cache_hit is None else cache_hit
        else:
            exec_duration = duration_ms
            exec_input_bytes = ids["input_bytes"]
            exec_output_bytes = 0
            cache_flag = cache_hit if cache_hit is not None else False
        execution_payload = self._execution_payload(
            duration_ms=exec_duration,
            input_bytes=exec_input_bytes,
            output_bytes=exec_output_bytes,
        )
        idempotency_payload = self._idempotency_payload(cache_flag)
        meta = EnvelopeMeta.from_ids(
            request_id=ids["request_id"],
            trace_id=ids["trace_id"],
            span_id=ids["span_id"],
            schema_version=self._schema_version,
            deterministic=context.deterministic_ids,
            transport=context.transport,
            route=context.route,
            method=context.method,
            status="error",
            attempt=context.attempt,
            execution=execution_payload,
            idempotency=idempotency_payload,
            tool_id=tool_id,
            prompt_id=prompt_id,
        )
        envelope = Envelope.failure(error=EnvelopeError(code=code, message=message), meta=meta)
        envelope_dict = envelope.model_dump(by_alias=True)
        error_payload = {"canonical": code, "message": message}
        metadata_payload = {
            key: value
            for key, value in {"toolId": tool_id, "promptId": prompt_id}.items()
            if value is not None
        }
        self._log_manager.emit(
            ServerLogEvent(
                ts=datetime.now(UTC),
                request_id=meta.request_id,
                trace_id=meta.trace_id,
                span_id=meta.span_id,
                transport=context.transport,
                route=context.route,
                method=context.method,
                status="error",
                attempt=context.attempt,
                execution=execution_payload,
                idempotency=idempotency_payload,
                error={"code": code, "message": message},
                metadata=metadata_payload,
                step_id=step_id,
            )
        )
        self._validate_envelope_and_log(
            envelope_dict=envelope_dict,
            context=context,
            ids=ids,
            tool_id=tool_id,
            prompt_id=prompt_id,
            execution=execution_payload,
            idempotency=idempotency_payload,
            status="error",
            error=error_payload,
        )
        return envelope

    def _normalise_context(
        self,
        context: RequestContext | None,
        route: str,
        method: str,
        payload: Mapping[str, Any],
    ) -> RequestContext:
        if context is None:
            context = RequestContext(
                transport="http",
                route=route,
                method=method,
                deterministic_ids=False,
            )
        return context.clone_with_payload(payload)

    def _validate_envelope_and_log(
        self,
        *,
        envelope_dict: Mapping[str, Any],
        context: RequestContext,
        ids: Mapping[str, Any],
        tool_id: str | None,
        prompt_id: str | None,
        execution: Mapping[str, Any],
        idempotency: Mapping[str, Any],
        status: str,
        error: dict[str, Any] | None,
    ) -> None:
        if self._validation_mode is ValidationMode.OFF:
            return
        validator = self._validation_registry.load_envelope()
        try:
            validator.validate(envelope_dict)
        except ValidationError as exc:
            canonical_code = "INVALID_OUTPUT" if tool_id else "INTERNAL_ERROR"
            fallback_error = {
                "canonical": canonical_code,
                "message": str(exc).splitlines()[0],
            }
            fallback_execution = self._execution_payload(
                duration_ms=float(execution.get("durationMs", 0.0)),
                input_bytes=int(execution.get("inputBytes", ids.get("input_bytes", 0))),
                output_bytes=0,
            )
            self._log_validation_event(
                context=context,
                ids=ids,
                status="error",
                execution=fallback_execution,
                idempotency=self._idempotency_payload(False),
                tool_id=tool_id,
                prompt_id=prompt_id,
                error=fallback_error,
            )
            if self._validation_mode is ValidationMode.ENFORCE:
                raise
            return
        output_bytes = _payload_size(envelope_dict)
        execution_payload = dict(execution)
        execution_payload["outputBytes"] = max(int(output_bytes), 0)
        execution_payload.setdefault(
            "durationMs", float(execution_payload.get("durationMs", 0.0))
        )
        execution_payload.setdefault(
            "inputBytes", int(execution_payload.get("inputBytes", ids.get("input_bytes", 0)))
        )
        idempotency_payload = dict(idempotency)
        idempotency_payload.setdefault("cacheHit", False)
        self._log_validation_event(
            context=context,
            ids=ids,
            status=status,
            execution=execution_payload,
            idempotency=idempotency_payload,
            tool_id=tool_id,
            prompt_id=prompt_id,
            error=error,
        )

    def _log_validation_event(
        self,
        *,
        context: RequestContext,
        ids: Mapping[str, Any],
        status: str,
        execution: Mapping[str, Any],
        idempotency: Mapping[str, Any],
        tool_id: str | None,
        prompt_id: str | None,
        error: dict[str, Any] | None,
    ) -> None:
        if self._validation_mode is ValidationMode.OFF:
            return
        metadata = {
            "schemaVersion": self._schema_version,
            "deterministic": context.deterministic_ids,
        }
        if tool_id:
            metadata["toolId"] = tool_id
        if prompt_id:
            metadata["promptId"] = prompt_id
        event = EnvelopeValidationEvent(
            ts=datetime.now(UTC),
            request_id=str(ids["request_id"]),
            trace_id=str(ids["trace_id"]),
            span_id=str(ids["span_id"]),
            transport=context.transport,
            route=context.route,
            method=context.method,
            status=status,
            execution=dict(execution),
            idempotency=dict(idempotency),
            attempt=context.attempt,
            metadata=metadata,
            error=error,
            step_id=self._validation_log.next_step_id(),
        )
        self._validation_log.emit(event)

    def _tool_to_dict(self, toolpack: Toolpack) -> dict[str, Any]:
        return {
            "id": toolpack.id,
            "version": toolpack.version,
            "deterministic": toolpack.deterministic,
            "timeoutMs": toolpack.timeout_ms,
            "limits": dict(toolpack.limits),
            "caps": dict(toolpack.caps),
        }

    def _request_ids(self, context: RequestContext) -> dict[str, Any]:
        if context._cached_ids is not None:
            return context._cached_ids
        payload = context.request_payload or {}
        fingerprint = _fingerprint({
            "route": context.route,
            "method": context.method,
            "payload": payload,
        })
        if context.deterministic_ids:
            request_id = uuid5(_DETERMINISTIC_NAMESPACE, f"req:{fingerprint}")
            trace_id = uuid5(_DETERMINISTIC_NAMESPACE, f"trace:{fingerprint}")
            span_id = uuid5(_DETERMINISTIC_NAMESPACE, f"span:{fingerprint}")
        else:
            request_id = uuid4()
            trace_id = uuid4()
            span_id = uuid4()
        computed = {
            "request_id": str(request_id),
            "trace_id": str(trace_id),
            "span_id": str(span_id),
            "input_bytes": _payload_size(payload),
        }
        context._cached_ids = computed
        return computed


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_size(payload: Mapping[str, Any] | Any) -> int:
    if isinstance(payload, Mapping):
        return len(_fingerprint(payload).encode("utf-8"))
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _prompt_from_payload(data: Mapping[str, Any], path: Path) -> Prompt:
    try:
        identifier = str(data["id"])
        major = int(data.get("major", 1))
        name = str(data.get("name", identifier))
        description = str(data.get("description", ""))
        tags = [str(tag) for tag in data.get("tags", [])]
        messages_field = data.get("messages", [])
        if not isinstance(messages_field, Iterable):
            raise TypeError("messages must be an iterable")
        messages: list[dict[str, str]] = []
        for entry in messages_field:
            if not isinstance(entry, Mapping):
                raise TypeError("prompt message must be a mapping")
            role = str(entry.get("role"))
            content = str(entry.get("content", ""))
            messages.append({"role": role, "content": content})
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid prompt file {path}: {exc}") from exc
    prompt_id = f"{identifier}@{major}"
    return Prompt(
        prompt_id=prompt_id,
        name=name,
        description=description,
        major=major,
        tags=tags,
        messages=messages,
    )


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
