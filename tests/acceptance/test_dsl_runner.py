import json
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.xfail(reason="DSL Runner not implemented yet")


def test_task_runner_executes_example_flow(repo_root: Path) -> None:
    # Arrange: example flow (minimal)
    flows_dir = repo_root / "flows"
    flows_dir.mkdir(exist_ok=True)
    flow_path = flows_dir / "example.react_self_refine.yaml"
    if not flow_path.exists():
        flow_path.write_text(textwrap.dedent("""
        version: 0.1
        globals:
          tools: {}
        graph:
          nodes:
            - id: seed
              kind: unit
              spec:
                type: llm
                tool_ref: gpt
                system: "Test system"
                prompt: "Hello {{inputs.name}}"
              inputs:
                name: ${root.name}
              outputs: [draft]
          control: []
        """))
    # Act: run minimal dry-run to get plan
    cmd = [
        "python",
        "-m",
        "pkgs.dsl.cli",
        "run",
        "--spec",
        str(flow_path),
        "--dry-run",
    ]
    # The module path is a placeholder; implement pkgs/dsl/cli.py to satisfy this.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Assert: exits 0 and prints a plan JSON
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    assert out.startswith("{") and out.endswith("}")
    plan = json.loads(out)
    assert "nodes" in plan and "control" in plan
