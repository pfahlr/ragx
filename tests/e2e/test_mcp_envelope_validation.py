from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


def _service(tmp_path: Path) -> McpService:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=log_dir,
    )


def test_invalid_tool_payload_returns_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAGX_SEED", "1337")
    service = _service(tmp_path)
    client = TestClient(create_app(service))

    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"encoding": "utf-8"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
    assert payload["meta"]["status"] == "error"
