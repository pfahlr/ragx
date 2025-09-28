import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{context} must be a mapping"
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, context: str) -> Sequence[object]:
    assert isinstance(value, Sequence), f"{context} must be a sequence"
    return cast(Sequence[object], value)


def _as_list(value: object, context: str) -> list[object]:
    assert isinstance(value, list), f"{context} must be a list"
    return cast(list[object], value)


def _as_dict(value: object, context: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{context} must be a mapping"
    return cast(dict[str, object], value)


def test_pyproject_ci_tooling_configuration() -> None:
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml must exist at project root"

    data_raw = tomllib.loads(pyproject_path.read_text())
    tool_config = _as_mapping(data_raw, "[tool]").get("tool")
    tool_mapping = _as_mapping(tool_config, "[tool]")

    ruff_mapping = _as_mapping(tool_mapping.get("ruff"), "[tool.ruff]")
    line_length = ruff_mapping.get("line-length")
    target_version = ruff_mapping.get("target-version")
    lint_mapping = _as_mapping(ruff_mapping.get("lint"), "[tool.ruff.lint]")
    select = _as_sequence(lint_mapping.get("select"), "[tool.ruff.lint].select")

    assert line_length == 100
    assert target_version == "py311"
    assert list(select) == ["E", "F", "I", "UP", "B"]

    mypy_mapping = _as_mapping(tool_mapping.get("mypy"), "[tool.mypy]")
    assert mypy_mapping.get("python_version") == "3.11"
    assert mypy_mapping.get("ignore_missing_imports") is True
    assert mypy_mapping.get("strict") is False

    pytest_mapping = _as_mapping(tool_mapping.get("pytest"), "[tool.pytest]")
    ini_options = _as_mapping(
        pytest_mapping.get("ini_options"), "[tool.pytest.ini_options]"
    )
    assert ini_options.get("testpaths") == ["tests"]
    assert ini_options.get("addopts") == "-q"


def test_ci_workflow_scaffold() -> None:
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow configuration must exist"

    workflow_raw = yaml.safe_load(workflow_path.read_text())
    workflow = _as_dict(workflow_raw, "workflow root")
    assert workflow.get("name") == "ci"

    triggers = workflow.get("on")
    if triggers is None:
        triggers = workflow.get(True)  # type: ignore[arg-type, call-overload]
    triggers_mapping = _as_dict(triggers, "workflow.on")
    assert set(triggers_mapping) == {"push", "pull_request"}

    jobs = _as_dict(workflow.get("jobs"), "workflow.jobs")
    build_job = _as_dict(jobs.get("build"), "workflow.jobs.build")
    assert build_job.get("runs-on") == "ubuntu-latest"

    steps = _as_list(build_job.get("steps"), "workflow.jobs.build.steps")

    step_dicts = [
        _as_dict(step, f"workflow.jobs.build.steps[{index}]")
        for index, step in enumerate(steps)
    ]

    uses_values = {step.get("uses") for step in step_dicts if "uses" in step}
    assert "actions/checkout@v4" in uses_values
    assert "actions/setup-python@v5" in uses_values

    python_step = next(
        step for step in step_dicts if step.get("uses") == "actions/setup-python@v5"
    )
    python_with = _as_dict(
        python_step.get("with"), "workflow.jobs.build.steps[setup-python].with"
    )
    assert python_with.get("python-version") == "3.11"

def test_makefile_includes_ci_targets() -> None:
    makefile_path = Path("Makefile")
    assert makefile_path.exists(), "Makefile must exist"

    makefile_text = makefile_path.read_text()
    for target in ("lint:", "typecheck:", "test:", "codex-bootstrap:"):
        assert target in makefile_text, f"Makefile must define {target} target"
