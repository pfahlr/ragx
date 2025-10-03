from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
FIXTURE_DIR = Path("tests/fixtures/mcp/core_tools/docs")


@pytest.fixture()
def runtime(tmp_path: Path) -> Iterable[CoreToolsRuntime]:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    logger = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)
    yield runtime
    logger.close()


def test_core_tools_minimal_exports_render(runtime: CoreToolsRuntime) -> None:
    payload = {
        "title": "Core Tools Demo",
        "template": "# {{ title }}\n\n{{ body }}",
        "body": "Hello from deterministic stub.",
        "front_matter": {"authors": ["RAGX"], "tags": ["demo"]},
    }

    result = runtime.invoke("mcp.tool:exports.render.markdown", payload)
    assert "markdown" in result
    assert result["markdown"].startswith("---")
    assert result["metadata"]["title"] == "Core Tools Demo"
    keys = set(result["metadata"]["front_matter_keys"])
    assert {"title", "authors", "tags"}.issubset(keys)


def test_core_tools_minimal_docs_load(runtime: CoreToolsRuntime) -> None:
    payload = {
        "path": str(FIXTURE_DIR / "example.md"),
        "metadata_path": str(FIXTURE_DIR / "example.json"),
    }
    result = runtime.invoke("mcp.tool:docs.load.fetch", payload)
    assert "document" in result and "metadata" in result
    assert result["metadata"]["title"] == "Example Document"
    assert result["metadata"]["checksum"].startswith("sha256:")


def test_core_tools_minimal_vector_query(runtime: CoreToolsRuntime) -> None:
    result = runtime.invoke(
        "mcp.tool:vector.query.search",
        {"query": "retrieval pipeline", "top_k": 2},
    )
    assert "hits" in result and len(result["hits"]) == 2
    assert result["hits"][0]["score"] >= result["hits"][1]["score"]


def test_core_tools_minimal_retry_flow(runtime: CoreToolsRuntime) -> None:
    result = runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {
            "title": "Retry",
            "template": "{{ title }}",
            "body": "will fail",
            "simulate_error": True,
        },
        max_attempts=2,
    )
    assert result["metadata"]["title"] == "Retry"

    log_entries = runtime.log_writer.path.read_text(encoding="utf-8").splitlines()
    assert any("invocation_retry" in entry for entry in log_entries)
    assert any("invocation_success" in entry for entry in log_entries)
