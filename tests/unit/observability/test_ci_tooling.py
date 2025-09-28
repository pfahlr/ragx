import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

PYPROJECT_PATH = Path("pyproject.toml")
CI_WORKFLOW_PATH = Path(".github/workflows/ci.yml")
MAKEFILE_PATH = Path("Makefile")


def test_pyproject_ci_tooling_config():
    assert PYPROJECT_PATH.exists(), "pyproject.toml must exist per spec"

    config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    tool_config = config.get("tool", {})

    ruff_config = tool_config.get("ruff")
    assert ruff_config is not None, "tool.ruff configuration missing"
    assert ruff_config.get("line-length") == 100
    assert ruff_config.get("target-version") == "py311"
    lint_config = ruff_config.get("lint")
    assert lint_config is not None, "tool.ruff.lint configuration missing"
    assert lint_config.get("select") == ["E", "F", "I", "UP", "B"]

    mypy_config = tool_config.get("mypy")
    assert mypy_config is not None, "tool.mypy configuration missing"
    assert mypy_config.get("python_version") == "3.11"
    assert mypy_config.get("ignore_missing_imports") is True
    assert mypy_config.get("strict") is False

    pytest_ini_options = tool_config.get("pytest", {}).get("ini_options")
    assert pytest_ini_options is not None, "tool.pytest.ini_options missing"
    assert pytest_ini_options.get("testpaths") == ["tests"]
    assert pytest_ini_options.get("addopts") == "-q"


def test_ci_workflow_runs_linters_typechecks_tests():
    assert CI_WORKFLOW_PATH.exists(), "CI workflow file must be present"

    workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    build_job = jobs.get("build")
    assert build_job is not None, "build job missing from CI workflow"

    steps = build_job.get("steps", [])
    uses_steps = {step.get("uses"): step for step in steps if step.get("uses")}
    assert "actions/checkout@v4" in uses_steps

    setup_python_step = uses_steps.get("actions/setup-python@v5")
    assert setup_python_step is not None, "Python setup step missing"
    assert setup_python_step.get("with", {}).get("python-version") == "3.11"

    run_commands = [step.get("run") for step in steps if step.get("run")]
    assert "pip install -r requirements.txt || true" in run_commands
    assert "pip install ruff mypy pytest coverage yamllint || true" in run_commands
    assert "ruff check . || true" in run_commands
    assert "mypy . || true" in run_commands
    assert "pytest --maxfail=1 --disable-warnings || true" in run_commands


def test_makefile_has_ci_targets():
    assert MAKEFILE_PATH.exists(), "Makefile must exist"
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert ".PHONY: lint typecheck test codex-bootstrap" in content
    assert "lint:" in content
    assert "typecheck:" in content
    assert "test:" in content
    assert "codex-bootstrap:" in content

    assert "ruff check ." in content
    assert "mypy ." in content
    assert "pytest -q || true" in content
    assert "python scripts/codex_next_tasks.py" in content


def test_codex_next_tasks_lists_sorted_tasks():
    script_path = Path("scripts/codex_next_tasks.py")
    assert script_path.exists(), "codex_next_tasks script missing"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    out_lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    assert out_lines, "Script output is empty"
    assert out_lines[0] == "Next tasks:", "Script must print header"

    listed_lines = out_lines[1:]
    expected_tasks = sorted(
        str(path)
        for path in Path("codex/agents/TASKS").glob("*.yaml")
    )
    assert listed_lines == [
        f"- {task}" for task in expected_tasks
    ], "Tasks must be sorted and prefixed"
