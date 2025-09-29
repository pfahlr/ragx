from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import ToolpackLoader


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _schema(**properties: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def test_execute_python_toolpack_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema_dir = tmp_path / "schemas"
    input_schema = _schema(value={"type": "integer", "minimum": 0})
    output_schema = _schema(result={"type": "integer"})
    input_schema_path = schema_dir / "calc.input.schema.json"
    output_schema_path = schema_dir / "calc.output.schema.json"
    _write(input_schema_path, json.dumps(input_schema))
    _write(output_schema_path, json.dumps(output_schema))

    module_root = tmp_path / "toolsrc"
    module_root.mkdir()
    _write(module_root / "__init__.py", "")
    module_code = """
def run(payload):
    value = payload["value"]
    return {"result": value * 3}
"""
    _write(module_root / "calc.py", module_code)
    monkeypatch.syspath_prepend(str(tmp_path))

    packs_dir = tmp_path / "toolpacks"
    packs_dir.mkdir()
    toolpack_data = {
        "id": "calc.multiply",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 2000,
        "limits": {"maxInputBytes": 1024, "maxOutputBytes": 1024},
        "inputSchema": {"$ref": os.path.relpath(input_schema_path, packs_dir)},
        "outputSchema": {"$ref": os.path.relpath(output_schema_path, packs_dir)},
        "execution": {"kind": "python", "module": "toolsrc.calc:run"},
    }
    _write(packs_dir / "calc.multiply.tool.yaml", yaml.safe_dump(toolpack_data, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(packs_dir)
    toolpack = loader.get("calc.multiply")

    executor = Executor()
    result_one = executor.run_toolpack(toolpack, {"value": 7})
    result_two = executor.run_toolpack(toolpack, {"value": 7})

    assert result_one == result_two == {"result": 21}
    result_one["result"] = 0
    assert executor.run_toolpack(toolpack, {"value": 7}) == {"result": 21}

    with pytest.raises(ToolpackExecutionError):
        executor.run_toolpack(toolpack, {"value": -1})
