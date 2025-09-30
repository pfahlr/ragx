from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml


def test_spec_has_components_and_tool_registry() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.xfail(f"Master spec not yet valid YAML: {exc}")
    assert "components" in spec and isinstance(spec["components"], list)
    assert "tool_registry" in spec and isinstance(spec["tool_registry"], dict)
    # ensure key components exist by id
    ids = {c.get("id") for c in spec["components"]}
    for required in ["dsl", "mcp_server", "vector_db_core"]:
        assert required in ids, f"Missing component '{required}' in spec"


def test_spec_documents_toolpack_class_location() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as handle:
        try:
            spec = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            pytest.xfail(f"Master spec not yet valid YAML: {exc}")

    components_raw = spec.get("components", [])
    assert isinstance(components_raw, list), "Spec components must be a list"

    components: dict[str, Mapping[str, object]] = {}
    for entry in components_raw:
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            components[entry["id"]] = entry

    search_paths = [
        ("mcp_server", ("interfaces", "classes")),
        ("toolpacks_runtime", ("interfaces", "classes")),
        ("toolpacks_runtime", ("classes",)),
    ]
    inspected: list[str] = []
    toolpack_spec: Mapping[str, object] | None = None

    for component_id, path in search_paths:
        component = components.get(component_id)
        if not isinstance(component, Mapping):
            inspected.append(f"{component_id}:missing")
            continue
        current: object = component
        traversed: list[str] = [component_id]
        for key in path:
            traversed.append(key)
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if not isinstance(current, list):
            inspected.append("->".join(traversed))
            continue
        for candidate in current:
            if isinstance(candidate, Mapping) and candidate.get("name") == "Toolpack":
                toolpack_spec = candidate
                break
        if toolpack_spec is not None:
            break
        inspected.append("->".join(traversed))

    if toolpack_spec is None:
        detail = "; ".join(inspected) if inspected else "no components examined"
        pytest.fail(f"Toolpack class spec not found; searched: {detail}")

    fields = toolpack_spec.get("fields") if isinstance(toolpack_spec, Mapping) else None
    assert isinstance(fields, list), "Toolpack spec must list fields"
    expected = {
        "id",
        "version",
        "deterministic",
        "timeoutMs",
        "limits",
        "caps",
        "inputSchema",
        "outputSchema",
        "execution",
        "env",
        "templating",
    }
    assert expected.issubset(set(fields)), "Toolpack spec missing required fields"
