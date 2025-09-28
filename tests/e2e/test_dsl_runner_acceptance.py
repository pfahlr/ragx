from pathlib import Path

import pytest

SPEC_PATH = Path("codex/specs/ragx_master_spec.yaml")

@pytest.mark.skipif(not SPEC_PATH.exists(), reason="Master spec missing")
def test_dsl_cli_exposes_expected_subcommands():
    # Placeholder: once implemented, `task-runner --help` should include 'run' and 'lint'
    pytest.skip("CLI not implemented yet")

@pytest.mark.xfail(reason="Flow validation and dry-run not implemented yet", strict=False)
def test_dsl_flow_dry_run_plan_prints_nodes(tmp_path):
    # When implemented: `task-runner run --spec flows/example.react_self_refine.yaml --dry-run`
    # should print a JSON plan with 'nodes' and 'control' keys.
    raise AssertionError("Implement FlowRunner.plan and CLI glue")
