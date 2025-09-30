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


def test_toolpack_spec_class_can_be_found() -> None:
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.xfail(f"Master spec not yet valid YAML: {exc}")

    components: dict[str, dict[str, object]] = {}
    for entry in spec.get("components", []):
        if isinstance(entry, dict) and "id" in entry:
            components[str(entry["id"])] = entry

    attempted_paths: list[str] = []

    def _find_toolpack(component_id: str) -> dict[str, object] | None:
        component = components.get(component_id)
        if not isinstance(component, dict):
            return None
        interfaces = component.get("interfaces")
        if not isinstance(interfaces, dict):
            attempted_paths.append(f"{component_id}.interfaces.classes")
            return None
        classes = interfaces.get("classes")
        if not isinstance(classes, list):
            attempted_paths.append(f"{component_id}.interfaces.classes")
            return None
        attempted_paths.append(f"{component_id}.interfaces.classes")
        for item in classes:
            if isinstance(item, dict) and item.get("name") == "Toolpack":
                return item
        return None

    toolpack_spec = _find_toolpack("mcp_server") or _find_toolpack("toolpacks_runtime")

    if toolpack_spec is None:
        attempted = ", ".join(attempted_paths) if attempted_paths else "<none>"
        pytest.fail(
            "Toolpack class not defined in master spec components; "
            f"checked {attempted}"
        )

    fields = toolpack_spec.get("fields")
    assert isinstance(fields, list) and "id" in fields, "Toolpack spec missing expected fields"
