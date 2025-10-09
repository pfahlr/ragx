from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from apps.mcp_server.http import create_app
from apps.mcp_server.service.mcp_service import McpService


class _InMemoryLogManager:
    def __init__(self) -> None:
        self.events: list = []
        self._step = 0

    def next_step_id(self) -> int:
        self._step += 1
        return self._step

    def emit(self, event) -> None:  # pragma: no cover - passthrough
        self.events.append(event)


TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
PROMPT_DIR = Path("apps/mcp_server/prompts")
SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
DOC_FIXTURE = Path("tests/fixtures/mcp/docs/sample_article.md")
META_FIXTURE = Path("tests/fixtures/mcp/docs/sample_metadata.json")


def test_http_python_toolpack_execution_and_cache_logging(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "123")
    log_manager = _InMemoryLogManager()
    service = McpService.create(
        toolpacks_dir=TOOLPACK_DIR,
        prompts_dir=PROMPT_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path,
        deterministic_logs=True,
        logger=log_manager,
    )

    app = create_app(service, deterministic_ids=True)
    with TestClient(app) as client:
        payload = {
            "arguments": {
                "path": str(DOC_FIXTURE),
                "metadataPath": str(META_FIXTURE),
            }
        }
        first_response = client.post("/mcp/tool/mcp.tool:docs.load.fetch", json=payload)
        second_response = client.post("/mcp/tool/mcp.tool:docs.load.fetch", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_body = first_response.json()
    second_body = second_response.json()

    assert first_body["ok"] is True
    assert second_body["ok"] is True
    assert first_body["data"]["result"] == second_body["data"]["result"]
    assert first_body["data"]["metadata"]["toolpack"]["deterministic"] is True

    assert len(log_manager.events) == 2
    assert log_manager.events[0].metadata.get("idempotencyCache") == "miss"
    assert log_manager.events[1].metadata.get("idempotencyCache") == "hit"
