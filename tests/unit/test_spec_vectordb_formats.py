from pathlib import Path

import yaml


def test_vectordb_accept_format_flag_in_spec():
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    assert spec_path.exists(), "Master spec missing"
    with spec_path.open() as f:
        spec = yaml.safe_load(f)
    vb = spec["arg_spec"]["vectordb_builder"]
    flags = {entry["flag"]: entry for entry in vb}
    assert "--accept-format" in flags, "Missing --accept-format in vectordb_builder"
    entry = flags["--accept-format"]
    assert entry.get("repeatable") is True
    assert set(entry.get("choices", [])) == {"pdf", "md"}
    assert sorted(entry.get("default", [])) == ["md", "pdf"]


def test_spec_mentions_markdown_and_front_matter_contracts():
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    with spec_path.open() as f:
        spec = yaml.safe_load(f)
    comps = {c["id"]: c for c in spec["components"]}
    assert "vector_db_core" in comps, "vector_db_core component missing"
    vcore = comps["vector_db_core"]
    dc = vcore.get("data_contracts", {})
    assert "corpus_input" in dc, "vector_db_core.data_contracts.corpus_input missing"
    assert "research_collector" in comps, "research_collector component missing"
    rc = comps["research_collector"]
    rcdc = rc.get("data_contracts", {})
    assert (
        "markdown_front_matter" in rcdc
    ), "research_collector.data_contracts.markdown_front_matter missing"
