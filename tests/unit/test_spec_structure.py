from pathlib import Path

import pytest
import yaml


def test_spec_has_components_and_tool_registry() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.fail(f"Master spec must remain valid YAML: {exc}")
    assert "components" in spec and isinstance(spec["components"], list)
    assert "tool_registry" in spec and isinstance(spec["tool_registry"], dict)
    # ensure key components exist by id
    ids = {c.get("id") for c in spec["components"]}
    for required in ["dsl", "mcp_server", "vector_db_core"]:
        assert required in ids, f"Missing component '{required}' in spec"


def test_spec_defines_toolpack_class_location() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.fail(f"Master spec must remain valid YAML: {exc}")

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


def test_spec_tests_block_and_flowscript_decisions() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.fail(f"Master spec must remain valid YAML: {exc}")

    tests_block = spec.get("tests")
    assert isinstance(tests_block, dict), "spec.tests must be a mapping"
    assert "unit" in tests_block and "e2e" in tests_block

    decisions = spec.get("open_decisions")
    assert isinstance(decisions, list), "spec.open_decisions must be a list"

    decision_ids = {entry.get("id") for entry in decisions if isinstance(entry, dict)}
    for required in [
        "flowscript_parser_engine",
        "flowscript_error_surface",
        "flowscript_expr_interp",
    ]:
        assert required in decision_ids
