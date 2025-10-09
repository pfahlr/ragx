"""Tests for CLI flag plumbing into :mod:`apps.mcp_server.cli`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.mcp_server import cli


class _SentinelService:
    """Lightweight stand-in returned from ``McpService.create``."""


def test_run_server_forwards_limit_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI limit flags propagate to ``McpService.create``."""

    captured: dict[str, object] = {}

    async def _fake_run_once(service: object, *, deterministic_ids: bool) -> None:
        captured["run_once_service"] = service
        captured["run_once_deterministic"] = deterministic_ids

    def _fake_create(
        *,
        toolpacks_dir: object,
        prompts_dir: object,
        schema_dir: object,
        log_dir: object,
        schema_version: str = "0.1.0",
        deterministic_logs: bool,
        logger: object | None = None,
        max_input_bytes: int | None = None,
        max_output_bytes: int | None = None,
        timeout_ms: int | None = None,
    ) -> _SentinelService:
        captured.update(
            {
                "max_input_bytes": max_input_bytes,
                "max_output_bytes": max_output_bytes,
                "timeout_ms": timeout_ms,
                "deterministic_logs": deterministic_logs,
                "log_dir": log_dir,
            }
        )
        return _SentinelService()

    monkeypatch.setattr(cli, "_run_once", _fake_run_once)
    monkeypatch.setattr(cli, "McpService", SimpleNamespace(create=_fake_create))

    args = cli.parse_args(
        [
            "--once",
            "--deterministic-ids",
            "--log-dir",
            str(tmp_path),
            "--max-input-bytes",
            "2048",
            "--max-output-bytes",
            "8192",
            "--timeout-ms",
            "1234",
        ]
    )

    asyncio.run(cli._run_server(args))

    assert captured["max_input_bytes"] == 2048
    assert captured["max_output_bytes"] == 8192
    assert captured["timeout_ms"] == 1234
    assert captured["deterministic_logs"] is True
    assert captured["log_dir"] == tmp_path
    assert isinstance(captured["run_once_service"], _SentinelService)
    assert captured["run_once_deterministic"] is True
