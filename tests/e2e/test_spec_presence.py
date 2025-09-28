from pathlib import Path


def test_master_spec_exists():
    assert Path("codex/specs/ragx_master_spec.yaml").exists(), "Master spec is missing."
