"""Resource constrained execution sandbox for Python transforms."""

from __future__ import annotations

import contextlib
import copy
import io
import math
import os
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import connection, get_context
from typing import Any

try:  # pragma: no cover - platform guard
    import resource
except ImportError:  # pragma: no cover - resource is POSIX only
    resource = None  # type: ignore[assignment]


@dataclass(slots=True)
class SandboxResult:
    """Outcome of a sandboxed execution."""

    result: Any
    stdout: str
    stderr: str
    stdout_snippet: str
    stderr_snippet: str
    stdout_truncated: bool
    stderr_truncated: bool
    exit_code: int
    duration_ms: float


class SandboxError(RuntimeError):
    """Base error for sandbox execution failures."""

    def __init__(self, message: str, result: SandboxResult) -> None:
        super().__init__(message)
        self.result = result


class SandboxExecutionError(SandboxError):
    """Raised when the sandboxed code raises an exception."""


class SandboxTimeoutError(SandboxError):
    """Raised when execution exceeds the configured time budget."""


class SandboxMemoryError(SandboxError):
    """Raised when execution exceeds the configured memory budget."""


def _truncate_for_log(text: str, limit_bytes: int) -> tuple[str, bool]:
    if limit_bytes <= 0:
        return "", bool(text)
    encoded = text.encode("utf-8")
    if len(encoded) <= limit_bytes:
        return text, False
    trimmed = encoded[:limit_bytes]
    safe = trimmed.decode("utf-8", errors="ignore")
    suffix = "… [truncated]"
    return safe + suffix, True


def _apply_limits(mem_bytes: int | None, cpu_seconds: float | None) -> None:
    if resource is None:  # pragma: no cover - handled in tests via platform guard
        return
    if mem_bytes and mem_bytes > 0:
        for limit_name in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(limit_name, (mem_bytes, mem_bytes))
            except (ValueError, OSError):  # pragma: no cover - depends on kernel policy
                continue
    if cpu_seconds and cpu_seconds > 0:
        cpu_cap = max(int(math.ceil(cpu_seconds)), 1)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap))
        except (ValueError, OSError):  # pragma: no cover - depends on kernel policy
            return


def _python_worker(
    conn: connection.Connection,
    code: str,
    inputs: Mapping[str, Any],
    mem_bytes: int | None,
    cpu_seconds: float | None,
) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exit_code = 0
    try:
        _apply_limits(mem_bytes, cpu_seconds)
        local_inputs = copy.deepcopy(dict(inputs))
        namespace: dict[str, Any] = {"__name__": "__sandbox__", "inputs": local_inputs}
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, namespace)
            transform = namespace.get("transform")
            if callable(transform):
                result = transform(namespace["inputs"])
            else:
                result = namespace.get("result")
        conn.send(
            {
                "status": "ok",
                "result": result,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - sandbox must capture everything
        exit_code = 1
        conn.send(
            {
                "status": "error",
                "exc_type": exc.__class__.__name__,
                "exc_repr": repr(exc),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )
    finally:
        conn.close()
        os._exit(exit_code)


class Sandbox:
    """Execute Python transforms inside an isolated subprocess."""

    def __init__(self, *, cpu_ms: int, mem_mb: int, net_enabled: bool) -> None:
        if cpu_ms <= 0:
            raise ValueError("cpu_ms must be positive")
        if mem_mb <= 0:
            raise ValueError("mem_mb must be positive")
        self.cpu_ms = int(cpu_ms)
        self.mem_mb = int(mem_mb)
        self.net_enabled = bool(net_enabled)

    def run_python(
        self,
        code: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        timeout_ms: int | None = None,
        log_max_bytes: int = 4096,
    ) -> SandboxResult:
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Python source")
        if log_max_bytes <= 0:
            raise ValueError("log_max_bytes must be positive")

        payload: Mapping[str, Any] = inputs or {}
        mem_bytes = self.mem_mb * 1024 * 1024
        cpu_seconds = self.cpu_ms / 1000.0
        wall_timeout = (timeout_ms if timeout_ms is not None else self.cpu_ms) / 1000.0
        wall_timeout = max(wall_timeout, 0.05)

        ctx = get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_python_worker,
            args=(child_conn, code, payload, mem_bytes, cpu_seconds),
        )
        start = time.perf_counter()
        proc.start()
        child_conn.close()

        message: dict[str, Any] | None = None
        try:
            if parent_conn.poll(wall_timeout):
                message = parent_conn.recv()
            else:
                proc.terminate()
                proc.join()
                duration_ms = (time.perf_counter() - start) * 1000.0
                result = self._build_result(
                    result=None,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    duration_ms=duration_ms,
                    log_max_bytes=log_max_bytes,
                )
                raise SandboxTimeoutError("Python transform timed out", result)
        finally:
            parent_conn.close()

        proc.join()
        duration_ms = (time.perf_counter() - start) * 1000.0
        exit_code = proc.exitcode if proc.exitcode is not None else -1

        if message is None:
            result = self._build_result(
                result=None,
                stdout="",
                stderr="",
                exit_code=exit_code,
                duration_ms=duration_ms,
                log_max_bytes=log_max_bytes,
            )
            raise SandboxExecutionError(
                "Python transform exited without reporting a result",
                result,
            )

        stdout = message.get("stdout", "")
        stderr = message.get("stderr", "")

        if message.get("status") == "ok":
            return self._build_result(
                result=message.get("result"),
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code if exit_code is not None else 0,
                duration_ms=duration_ms,
                log_max_bytes=log_max_bytes,
            )

        exc_type = message.get("exc_type", "SandboxError")
        exc_repr = message.get("exc_repr", "execution failed")
        result_obj = self._build_result(
            result=None,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code if exit_code != 0 else 1,
            duration_ms=duration_ms,
            log_max_bytes=log_max_bytes,
        )
        message_text = f"Python transform raised {exc_type}: {exc_repr}"
        if exc_type == "MemoryError" or exit_code == -9:
            raise SandboxMemoryError(message_text, result_obj)
        raise SandboxExecutionError(message_text, result_obj)

    def _build_result(
        self,
        *,
        result: Any,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: float,
        log_max_bytes: int,
    ) -> SandboxResult:
        stdout_snippet, stdout_truncated = _truncate_for_log(stdout, log_max_bytes)
        stderr_snippet, stderr_truncated = _truncate_for_log(stderr, log_max_bytes)
        return SandboxResult(
            result=result,
            stdout=stdout,
            stderr=stderr,
            stdout_snippet=stdout_snippet,
            stderr_snippet=stderr_snippet,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )
