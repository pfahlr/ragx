from pathlib import Path

def test_master_spec_exists():
    spec_path = Path("codex/specs/ragx_master_spec.yaml")
    assert spec_path.exists(), "Master spec is missing."
