import tomllib
from pathlib import Path

import yaml


def test_pyproject_ci_tooling_configuration():
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml must exist at project root"

    data = tomllib.loads(pyproject_path.read_text())
    tool_config = data.get("tool")
    assert tool_config is not None, "pyproject.toml must define [tool] table"

    ruff_config = tool_config.get("ruff")
    assert ruff_config is not None, "pyproject.toml must define [tool.ruff] configuration"
    assert ruff_config.get("line-length") == 100
    assert ruff_config.get("target-version") == "py311"
    lint_config = ruff_config.get("lint")
    assert lint_config is not None, "pyproject.toml must define [tool.ruff.lint] configuration"
    assert lint_config.get("select") == ["E", "F", "I", "UP", "B"]

    mypy_config = tool_config.get("mypy")
    assert mypy_config is not None, "pyproject.toml must define [tool.mypy] configuration"
    assert mypy_config.get("python_version") == "3.11"
    assert mypy_config.get("ignore_missing_imports") is True
    assert mypy_config.get("strict") is True

    pytest_config = tool_config.get("pytest")
    assert (
        pytest_config is not None
    ), "pyproject.toml must define [tool.pytest.ini_options] configuration"
    ini_options = pytest_config.get("ini_options")
    assert ini_options is not None
    assert ini_options.get("testpaths") == ["tests"]
    assert ini_options.get("addopts") == "-q"


def test_ci_workflow_scaffold():
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow configuration must exist"

    workflow = yaml.safe_load(workflow_path.read_text())
    assert workflow.get("name") == "ci"
    assert set(workflow.get("on", [])) == {"push", "pull_request"}

    jobs = workflow.get("jobs")
    assert jobs is not None and "build" in jobs

    build_job = jobs["build"]
    assert build_job.get("runs-on") == "ubuntu-latest"

    steps = build_job.get("steps")
    assert isinstance(steps, list) and steps, "CI workflow must define build steps"

    expected_steps = [
        {"uses": "actions/checkout@v4"},
        {"uses": "actions/setup-python@v5", "with": {"python-version": "3.11"}},
        {"run": "pip install -r requirements.txt || true"},
        {"run": "pip install ruff mypy pytest coverage yamllint"},
        {"run": "ruff check ."},
        {"run": "mypy ."},
        {"run": "pytest --maxfail=1 --disable-warnings"},
    ]

    for expected, actual in zip(expected_steps, steps, strict=True):
        for key, value in expected.items():
            assert actual.get(key) == value


def test_makefile_includes_ci_targets():
    makefile_path = Path("Makefile")
    assert makefile_path.exists(), "Makefile must exist"

    makefile_text = makefile_path.read_text()
    for target in ("lint:", "typecheck:", "test:", "codex-bootstrap:"):
        assert target in makefile_text, f"Makefile must define {target} target"
