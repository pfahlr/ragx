import os

def test_master_spec_exists():
    assert os.path.exists("codex/specs/ragx_master_spec.yaml"), "Master spec is missing."
