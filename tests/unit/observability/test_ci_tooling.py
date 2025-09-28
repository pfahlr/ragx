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

    assert ".PHONY: lint typecheck test codex-bootstrap check unit integration e2e acceptance" in content
    assert "lint:" in content
    assert "typecheck:" in content
    assert "test:" in content
    assert "check: lint typecheck test" in content
    assert "codex-bootstrap:" in content

    assert "PYTHON ?= python3" in content
    assert "ruff check ." in content and "ruff check . || true" not in content
    assert "yamllint -s codex/ flows/" in content
    assert "mypy ." in content and "mypy . || true" not in content
    assert "pytest --maxfail=1 --disable-warnings" in content and "|| true" not in content
    assert "-m scripts.codex_next_tasks" in content
