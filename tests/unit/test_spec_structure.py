from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

SPEC_PATH = Path("codex/specs/ragx_master_spec.yaml")


def _load_spec() -> dict[str, Any]:
    with SPEC_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError("Master spec must load to a mapping")
    return data


def test_spec_has_components_and_tool_registry() -> None:
    spec = _load_spec()
    assert "components" in spec and isinstance(spec["components"], list)
    assert "tool_registry" in spec and isinstance(spec["tool_registry"], dict)
    ids = {c.get("id") for c in spec["components"] if isinstance(c, dict)}
    for required in ["dsl", "mcp_server", "vector_db_core"]:
        assert required in ids, f"Missing component '{required}' in spec"


def test_spec_defines_toolpack_class_location() -> None:
    spec = _load_spec()
    components = spec.get("components")
    assert isinstance(components, list), "spec.components must be a list"

    toolpack_entry = None
    component_id = None

    for component in components:
        if not isinstance(component, dict) or component.get("id") != "mcp_server":
            continue
        classes = component.get("interfaces", {}).get("classes") or []
        for entry in classes:
            if isinstance(entry, dict) and entry.get("name") == "Toolpack":
                toolpack_entry = entry
                component_id = "mcp_server"
                break
        if toolpack_entry:
            break

    if toolpack_entry is None:
        for component in components:
            if not isinstance(component, dict) or component.get("id") != "toolpacks_runtime":
                continue
            classes = component.get("interfaces", {}).get("classes") or []
            for entry in classes:
                if isinstance(entry, dict) and entry.get("name") == "Toolpack":
                    toolpack_entry = entry
                    component_id = "toolpacks_runtime"
                    break
            if toolpack_entry:
                break

    if toolpack_entry is None:
        pytest.fail(
            "Toolpack class not found in spec components; expected under mcp_server or toolpacks_runtime"
        )

    assert toolpack_entry.get("name") == "Toolpack"
    assert toolpack_entry.get("fields"), f"Toolpack spec under {component_id} missing fields"


def test_spec_tests_section_is_mapping() -> None:
    spec = _load_spec()
    tests_section = spec.get("tests")
    assert isinstance(tests_section, dict), "spec.tests must be a mapping"
    for key in ["unit", "e2e", "ci"]:
        assert key in tests_section, f"tests section missing '{key}'"
    assert isinstance(tests_section["unit"], list)
    assert isinstance(tests_section["e2e"], list)
    assert isinstance(tests_section["ci"], dict)
    assert all(isinstance(name, str) for name in tests_section["unit"])
    assert all(isinstance(name, str) for name in tests_section["e2e"])

    open_decisions = spec.get("open_decisions")
    assert isinstance(open_decisions, list), "open_decisions must be a list"
    for decision in open_decisions:
        assert isinstance(decision, dict), "each open decision must be a mapping"
        assert {"id", "question", "options", "default"} <= set(decision)


def test_minimal_yaml_loads(tmp_path: Path) -> None:
    content = (
        "tests:\n"
        "  unit:\n"
        "    - demo_unit\n"
        "  e2e:\n"
        "    - demo_e2e\n"
        "  ci:\n"
        "    coverage_minimum: 85\n"
        "open_decisions:\n"
        "  - id: sample_decision\n"
        "    question: Example?\n"
        "    options: [a, b]\n"
        "    default: a\n"
    )
    sample = tmp_path / "sample.yaml"
    sample.write_text(content, encoding="utf-8")
    loaded = yaml.safe_load(sample.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    assert "tests" in loaded and "open_decisions" in loaded
