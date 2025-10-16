from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "codex" / "specs" / "dsl" / "v1"
FLOW = ROOT / "flows" / "examples" / "multishot_smoke.yaml"
FLOW_SCHEMA = SCHEMA_DIR / "flow.schema.json"


@lru_cache(maxsize=1)
def _load_flow_validator() -> jsonschema.protocols.Validator:
    with FLOW_SCHEMA.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    store: dict[str, dict[str, Any]] = {}
    for schema_path in SCHEMA_DIR.glob("*.schema.json"):
        with schema_path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        schema_id = doc.get("$id")
        if schema_id:
            store[schema_id] = doc

    resolver = jsonschema.RefResolver.from_schema(schema, store=store)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema, resolver=resolver)


def _validate_flow(instance: dict[str, Any]) -> None:
    validator = _load_flow_validator()
    validator.validate(instance)


@pytest.mark.parametrize("path", [FLOW, FLOW_SCHEMA])
def test_files_exist(path: Path) -> None:
    assert os.path.exists(path), f"Missing required file: {path}"


def test_flow_schema_strict_accepts_valid() -> None:
    with FLOW.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    _validate_flow(data)


def test_flow_schema_strict_rejects_unknown_fields() -> None:
    with FLOW.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    data_with_extra = copy.deepcopy(data)
    data_with_extra["unknown_field"] = True
    with pytest.raises(jsonschema.ValidationError):
        _validate_flow(data_with_extra)

    data_with_bad_node = copy.deepcopy(data)
    data_with_bad_node["nodes"][0]["mystery"] = 123
    with pytest.raises(jsonschema.ValidationError):
        _validate_flow(data_with_bad_node)
