from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _schema(**properties: dict[str, object]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _build_toolpack(
    tmp_path: Path,
    *,
    tool_slug: str,
    module_name: str,
    module_source: str,
    input_schema: dict[str, object],
    output_schema: dict[str, object],
    timeout_ms: int,
    max_input: int,
    max_output: int,
) -> tuple[Path, str]:
    module_root = tmp_path / "toolsrc"
    module_root.mkdir(exist_ok=True)
    _write(module_root / "__init__.py", "")
    module_path = module_root / f"{module_name}.py"
    _write(module_path, textwrap.dedent(module_source))

    schema_root = tmp_path / "schemas"
    input_schema_path = schema_root / f"{module_name}.input.schema.json"
    output_schema_path = schema_root / f"{module_name}.output.schema.json"
    _write(input_schema_path, json.dumps(input_schema))
    _write(output_schema_path, json.dumps(output_schema))

    toolpacks_root = tmp_path / "toolpacks"
    toolpacks_root.mkdir(exist_ok=True)
    tool_id = f"tests.tool:{tool_slug}"
    toolpack_payload = {
        "id": tool_id,
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": timeout_ms,
        "limits": {
            "maxInputBytes": max_input,
            "maxOutputBytes": max_output,
        },
        "inputSchema": {"$ref": str(Path("../schemas") / input_schema_path.name)},
        "outputSchema": {"$ref": str(Path("../schemas") / output_schema_path.name)},
        "execution": {"kind": "python", "module": f"toolsrc.{module_name}:run"},
    }
    _write(
        toolpacks_root / f"{module_name}.tool.yaml",
        yaml.safe_dump(toolpack_payload, sort_keys=False),
    )
    return toolpacks_root, tool_id


def _service_for_toolpack(tmp_path: Path, toolpacks_dir: Path) -> McpService:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return McpService.create(
        toolpacks_dir=toolpacks_dir,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=log_dir,
        schema_version="0.1.0",
        deterministic_logs=True,
    )


def _load_server_log(service: McpService) -> list[dict[str, object]]:
    log_path = service.log_manager.writer.path
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clear_toolsrc_modules() -> None:
    for name in list(sys.modules):
        if name == "toolsrc" or name.startswith("toolsrc."):
            sys.modules.pop(name)


def test_python_toolpack_executes_and_logs_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_source = """
    CALL_COUNT = 0

    def run(payload):
        global CALL_COUNT
        CALL_COUNT += 1
        return {"result": payload["value"] + 1}
    """
    toolpacks_dir, tool_id = _build_toolpack(
        tmp_path,
        tool_slug="adderexec",
        module_name="adder_exec",
        module_source=module_source,
        input_schema=_schema(value={"type": "integer"}),
        output_schema=_schema(result={"type": "integer"}),
        timeout_ms=5000,
        max_input=8192,
        max_output=8192,
    )
    _clear_toolsrc_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    service = _service_for_toolpack(tmp_path, toolpacks_dir)
    client = TestClient(create_app(service, deterministic_ids=True))

    payload = {"arguments": {"value": 7}}
    first = client.post(f"/mcp/tool/{tool_id}", json=payload)
    assert first.status_code == 200
    data_first = first.json()
    assert data_first["ok"] is True
    assert data_first["data"]["result"]["result"] == 8

    second = client.post(f"/mcp/tool/{tool_id}", json=payload)
    assert second.status_code == 200
    data_second = second.json()
    assert data_second["ok"] is True
    assert data_second["data"]["result"] == data_first["data"]["result"]

    module = importlib.import_module("toolsrc.adder_exec")
    assert module.CALL_COUNT == 1

    server_events = _load_server_log(service)
    tool_events = [event for event in server_events if event["route"] == "tool"]
    assert len(tool_events) == 2
    cache_markers = [event["metadata"].get("idempotencyCache") for event in tool_events]
    assert cache_markers == ["miss", "hit"], cache_markers
    assert all(event["inputBytes"] > 0 for event in tool_events)
    assert all(event["outputBytes"] > 0 for event in tool_events)


def test_toolpack_input_size_limit_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_source = """
    CALL_COUNT = 0

    def run(payload):
        raise AssertionError("execution should not be reached")
    """
    toolpacks_dir, tool_id = _build_toolpack(
        tmp_path,
        tool_slug="limitsexec",
        module_name="limits_exec",
        module_source=module_source,
        input_schema=_schema(payload={"type": "string", "maxLength": 512}),
        output_schema=_schema(result={"type": "string"}),
        timeout_ms=5000,
        max_input=32,
        max_output=8192,
    )
    _clear_toolsrc_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    service = _service_for_toolpack(tmp_path, toolpacks_dir)
    client = TestClient(create_app(service, deterministic_ids=True))

    response = client.post(
        f"/mcp/tool/{tool_id}",
        json={"arguments": {"payload": "x" * 128}},
    )

    assert response.status_code == 400
    envelope = response.json()
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_INPUT"
    assert "maxInputBytes" in envelope["error"]["message"]

    module = importlib.import_module("toolsrc.limits_exec")
    assert module.CALL_COUNT == 0


def test_toolpack_timeout_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_source = """
    import time

    def run(payload):
        time.sleep(0.2)
        return {"result": payload.get("value", 0)}
    """
    toolpacks_dir, tool_id = _build_toolpack(
        tmp_path,
        tool_slug="timeoutexec",
        module_name="timeout_exec",
        module_source=module_source,
        input_schema=_schema(value={"type": "integer"}),
        output_schema=_schema(result={"type": "integer"}),
        timeout_ms=50,
        max_input=8192,
        max_output=8192,
    )
    _clear_toolsrc_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    service = _service_for_toolpack(tmp_path, toolpacks_dir)
    client = TestClient(create_app(service, deterministic_ids=True))

    response = client.post(
        f"/mcp/tool/{tool_id}",
        json={"arguments": {"value": 5}},
    )

    assert response.status_code == 504
    envelope = response.json()
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "TIMEOUT"
    assert "timeout" in envelope["error"]["message"].lower()

    server_events = _load_server_log(service)
    tool_events = [event for event in server_events if event["route"] == "tool"]
    assert tool_events, "expected tool event logged"
    duration = max(event["durationMs"] for event in tool_events)
    assert duration >= 50
