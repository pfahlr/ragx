from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
DOC_FIXTURE_DIR = Path("tests/fixtures/mcp/docs")
SCHEMA_VERSION = "0.1.0"


@pytest.fixture()
def runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CoreToolsRuntime, JsonLogWriter]:
    monkeypatch.setenv("RAGX_SEED", "1234")
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest_symlink = tmp_path / "runs/core_tools/minimal.jsonl"
    writer = JsonLogWriter(
        agent_id="test-agent",
        task_id="core-tools",
        storage_prefix=storage_prefix,
        latest_symlink=latest_symlink,
        schema_version=SCHEMA_VERSION,
        deterministic=True,
    )

    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=writer,
        agent_id="test-agent",
        task_id="core-tools",
    )
    return runtime, writer


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_exports_render_markdown_is_deterministic(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, writer = runtime
    payload = {
        "title": "Core Tools Demo",
        "template": "# {{ title }}\n\n{{ body }}",
        "body": "Hello from deterministic stub.",
        "frontMatter": {"authors": ["RAGX"], "tags": ["demo"]},
    }

    result = runtime_instance.invoke("mcp.tool:exports.render.markdown", payload)

    expected_markdown = (
        "---\n"
        "authors:\n"
        "- RAGX\n"
        "tags:\n"
        "- demo\n"
        "title: Core Tools Demo\n"
        "---\n"
        "# Core Tools Demo\n\n"
        "Hello from deterministic stub.\n"
    )
    assert result["markdown"] == expected_markdown
    digest = hashlib.sha256(expected_markdown.encode("utf-8")).hexdigest()
    assert result["contentHash"] == digest
    assert result["metadata"]["title"] == "Core Tools Demo"

    events = _read_events(writer.path)
    invoke, success = events[-2:]
    assert invoke["event"] == "tool.invoke"
    assert success["event"] == "tool.ok"
    assert success["status"] == "ok"
    assert success["execution"]["outputBytes"] >= len(
        expected_markdown.encode("utf-8")
    )


def test_vector_query_search_returns_sorted_hits(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, writer = runtime
    payload = {"query": "retrieval quality", "topK": 2}
    first = runtime_instance.invoke("mcp.tool:vector.query.search", payload)
    second = runtime_instance.invoke("mcp.tool:vector.query.search", payload)

    assert first == second, "Deterministic tool should cache identical responses"
    hits = first["hits"]
    assert len(hits) == 2
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["document"]["title"] != hits[1]["document"]["title"]

    # ensure logging captured byte counts for cached response as well
    events = _read_events(writer.path)
    last_event = events[-1]
    assert last_event["toolId"] == "mcp.tool:vector.query.search"
    assert last_event["event"] == "tool.ok"
    assert last_event["status"] == "ok"
    assert last_event["attempt"] == 0


def test_docs_load_fetch_reads_fixture(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, writer = runtime
    payload = {
        "path": str(DOC_FIXTURE_DIR / "sample_article.md"),
        "metadataPath": str(DOC_FIXTURE_DIR / "sample_metadata.json"),
    }
    result = runtime_instance.invoke("mcp.tool:docs.load.fetch", payload)

    assert "document" in result and "metadata" in result
    assert result["document"]["path"].endswith("sample_article.md")
    assert "Sample Article" in result["document"]["content"]
    assert result["metadata"]["title"] == "Sample Article"

    events = _read_events(writer.path)
    assert any(evt["toolId"] == "mcp.tool:docs.load.fetch" for evt in events)


def test_runtime_raises_for_unknown_tool(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, _ = runtime
    with pytest.raises(KeyError):
        runtime_instance.invoke("mcp.tool:unknown", {})
