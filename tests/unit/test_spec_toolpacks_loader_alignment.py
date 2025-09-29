from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from apps.toolpacks.loader import Toolpack, ToolpackLoader

SPEC_PATH = Path("codex/specs/ragx_master_spec.yaml")


def _load_spec() -> dict[str, Any]:
    if not SPEC_PATH.exists():
        pytest.skip("Master spec missing at codex/specs/ragx_master_spec.yaml")
    with SPEC_PATH.open(encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - spec still being authored
            pytest.xfail(f"Master spec not yet valid YAML: {exc}")
    if not isinstance(data, dict):  # pragma: no cover - guardrail
        pytest.fail("Master spec is not a mapping")
    return data


def _toolpacks_component() -> dict[str, Any]:
    spec = _load_spec()
    components = spec.get("components", [])
    for component in components:
        if isinstance(component, dict) and component.get("id") == "toolpacks_runtime":
            return component
    pytest.fail("toolpacks_runtime component missing from master spec")


def _classes_by_name(component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    interfaces = component.get("interfaces", {})
    if not isinstance(interfaces, dict):
        pytest.fail("toolpacks_runtime.interfaces must be a mapping")
    classes = interfaces.get("classes", [])
    by_name: dict[str, dict[str, Any]] = {}
    for entry in classes:
        if isinstance(entry, dict) and "name" in entry:
            by_name[str(entry["name"])] = entry
    return by_name


def _to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def test_toolpack_fields_align_with_spec() -> None:
    component = _toolpacks_component()
    classes = _classes_by_name(component)
    toolpack_spec = classes.get("Toolpack")
    assert toolpack_spec is not None, (
        "Toolpack class missing in toolpacks_runtime.interfaces.classes"
    )

    spec_fields = toolpack_spec.get("fields", [])
    assert spec_fields, "Toolpack fields missing from spec"

    expected = {_to_snake(str(field)) for field in spec_fields}
    actual = {field.name for field in dataclasses.fields(Toolpack)}

    assert expected <= actual, f"Toolpack dataclass missing fields: {sorted(expected - actual)}"
    extras = actual - expected
    assert extras == {"source_path"}, "Unexpected extra fields in Toolpack dataclass"


def test_toolpack_loader_methods_align_with_spec() -> None:
    component = _toolpacks_component()
    classes = _classes_by_name(component)
    loader_spec = classes.get("ToolpackLoader")
    assert loader_spec is not None, (
        "ToolpackLoader class missing in toolpacks_runtime.interfaces.classes"
    )

    spec_methods = set(map(str, loader_spec.get("methods", [])))
    assert spec_methods, "ToolpackLoader methods missing from spec"

    actual_methods = {
        name for name, member in inspect.getmembers(ToolpackLoader, predicate=inspect.isfunction)
    }
    missing = spec_methods - actual_methods
    assert not missing, f"ToolpackLoader missing methods defined in spec: {sorted(missing)}"
