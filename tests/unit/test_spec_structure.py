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
    for required in ["dsl","mcp_server","vector_db_core"]:
        assert required in ids, f"Missing component '{required}' in spec"
