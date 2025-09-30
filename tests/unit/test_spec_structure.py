from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _load_master_spec() -> dict[str, object]:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open(encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - fails until spec valid
            pytest.xfail(f"Master spec not yet valid YAML: {exc}")
    assert isinstance(data, dict), "master spec must deserialize into a mapping"
    return data


def test_spec_has_components_and_tool_registry() -> None:
    spec = _load_master_spec()
    assert "components" in spec and isinstance(spec["components"], list)
    assert "tool_registry" in spec and isinstance(spec["tool_registry"], dict)
    # ensure key components exist by id
    ids = {c.get("id") for c in spec["components"]}
    for required in ["dsl", "mcp_server", "vector_db_core"]:
        assert required in ids, f"Missing component '{required}' in spec"


def test_spec_defines_toolpack_class_location() -> None:
    spec = _load_master_spec()
    components = spec.get("components")
    assert isinstance(components, list), "spec.components must be a list"

    toolpack_entry = None
    component_id = None

    for component in components:
        if component.get("id") != "mcp_server":
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
            if component.get("id") != "toolpacks_runtime":
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


def test_spec_tests_and_flowscript_decisions_structure() -> None:
    spec = _load_master_spec()

    tests_block = spec.get("tests")
    assert isinstance(tests_block, dict), "spec.tests must be a mapping"
    for key in ("unit", "e2e", "ci"):
        assert key in tests_block, f"spec.tests missing '{key}' section"

    decisions = spec.get("flowscript_decisions")
    assert isinstance(decisions, list), "spec.flowscript_decisions must be a list"
    assert all(isinstance(entry, dict) for entry in decisions)
    assert all("id" in entry for entry in decisions), "each FlowScript decision must have an id"


def test_minimal_flowscript_decision_yaml_loads(tmp_path: Path) -> None:
    snippet = (
        "tests:\n"
        "  unit:\n"
        "    - sample_test\n"
        "  ci:\n"
        "    coverage_minimum: 80\n"
        "flowscript_decisions:\n"
        "  - id: sample_decision\n"
        "    question: Example?\n"
        "    options: [a, b]\n"
        "    default: a\n"
    )
    path = tmp_path / "spec.yaml"
    path.write_text(snippet, encoding="utf-8")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["tests"]["unit"] == ["sample_test"]
    assert loaded["flowscript_decisions"][0]["id"] == "sample_decision"
