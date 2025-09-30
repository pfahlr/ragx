import pathlib
import yaml

TASK_FILE = pathlib.Path("codex/agents/TASKS/06a_core_tools_minimal_subset.yaml")


def load_task():
    text = TASK_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise AssertionError("Task YAML must parse to a mapping")
    return data


def test_task_includes_required_metadata_keys():
    task = load_task()
    for key in ("id", "title", "owner", "priority", "phase", "depends_on", "spec_refs"):
        assert key in task, f"missing required key: {key}"
    assert task["owner"] == "codex", "owner must be codex per spec guidelines"
    assert task["priority"] == "P0", "priority must be P0 for core tools bootstrap"
    assert isinstance(task["depends_on"], list) and task["depends_on"], "depends_on must be non-empty list"


def test_spec_refs_cover_core_components():
    task = load_task()
    spec_refs = task.get("spec_refs", {})
    components = set(spec_refs.get("components", []))
    expected = {"core_tools", "toolpacks_runtime", "mcp_server"}
    missing = expected - components
    assert not missing, f"spec_refs.components missing: {sorted(missing)}"
    arg_refs = set(spec_refs.get("arg_spec", []))
    assert {"mcp_server", "task_runner"}.issubset(arg_refs), "arg_spec refs must include MCP server + task runner flags"


def test_acceptance_includes_logging_and_diff_tests():
    task = load_task()
    acceptance = task.get("acceptance", [])
    required = {
        "tests/unit/test_core_tools_minimal_subset_schemas.py",
        "tests/unit/test_core_tools_structured_logging.py",
        "tests/integration/test_core_tools_log_diff.py",
        "tests/e2e/test_mcp_minimal_core_tools.py",
        "./scripts/ensure_green.sh",
    }
    missing = required - set(acceptance)
    assert not missing, f"acceptance criteria missing entries: {sorted(missing)}"


def test_observability_requirements_call_out_json_logs():
    task = load_task()
    requirements = "\n".join(task.get("observability_requirements", []))
    for token in ("json", "timestamp", "agent_id", "task_id", "step_id", "persist"):
        assert token in requirements, f"observability requirements must mention '{token}'"


def test_log_diff_strategy_uses_deepdiff_and_has_whitelist():
    task = load_task()
    strategy = task.get("log_diff_strategy", {})
    assert strategy.get("tool") == "python.deepdiff", "log diff strategy must use python.deepdiff"
    whitelist = strategy.get("whitelist_fields")
    assert isinstance(whitelist, list) and whitelist, "log diff whitelist_fields must be present"
    for field in ("timestamp", "run_id", "trace_id"):
        assert field in whitelist, f"log diff whitelist must include {field}"


def test_test_plan_includes_schema_logging_and_retry_checks():
    task = load_task()
    plan = "\n".join(task.get("test_plan", []))
    for token in ("schema", "logging", "retry", "fallback", "golden log"):
        assert token in plan, f"test plan must cover {token}"
