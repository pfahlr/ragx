from __future__ import annotations

from dataclasses import asdict

import pytest

from pkgs.transforms.sandbox import (
    Sandbox,
    SandboxExecutionError,
    SandboxMemoryError,
    SandboxResult,
    SandboxTimeoutError,
)


@pytest.fixture()
def sandbox() -> Sandbox:
    return Sandbox(cpu_ms=500, mem_mb=32, net_enabled=False)


def _result_dict(result: SandboxResult) -> dict[str, object]:
    data = asdict(result)
    # duration is non-deterministic; ensure present but avoid flakiness
    assert isinstance(data["duration_ms"], float) and data["duration_ms"] >= 0
    data.pop("duration_ms")
    return data


def test_run_python_executes_transform_and_returns_result(sandbox: Sandbox) -> None:
    code = """
from math import sqrt

print(f"starting:{inputs['a']}+{inputs['b']}")
value = sqrt(inputs['a'] ** 2 + inputs['b'] ** 2)

def transform(payload):
    return {
        "hypotenuse": round(value, 3),
        "sum": payload['a'] + payload['b'],
    }
"""

    result = sandbox.run_python(code, inputs={"a": 3, "b": 4})

    snapshot = _result_dict(result)
    assert snapshot == {
        "result": {"hypotenuse": 5.0, "sum": 7},
        "stdout": "starting:3+4\n",
        "stderr": "",
        "stdout_snippet": "starting:3+4\n",
        "stderr_snippet": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "exit_code": 0,
    }


def test_run_python_exposes_inputs_without_mutating_original(sandbox: Sandbox) -> None:
    payload = {"value": 21}
    code = """
inputs['value'] += 1
inputs['double'] = inputs['value'] * 2
result = dict(inputs)
"""

    result = sandbox.run_python(code, inputs=payload)

    snapshot = _result_dict(result)
    assert snapshot["result"] == {"value": 22, "double": 44}
    assert payload == {"value": 21}, "Sandbox must not mutate caller inputs"


def test_run_python_raises_with_stderr_on_exception(sandbox: Sandbox) -> None:
    code = """
import sys

print("about to fail")
sys.stderr.write("boom\\n")
raise ValueError("bad payload")
"""

    with pytest.raises(SandboxExecutionError) as excinfo:
        sandbox.run_python(code, inputs={})

    err = excinfo.value
    assert "ValueError" in str(err)
    assert err.result.exit_code != 0
    assert err.result.stdout == "about to fail\n"
    assert err.result.stderr == "boom\n"
    assert err.result.stderr_snippet.endswith("boom\n")


def test_run_python_times_out_when_process_exceeds_budget(sandbox: Sandbox) -> None:
    code = """
while True:
    pass
"""

    with pytest.raises(SandboxTimeoutError):
        sandbox.run_python(code, inputs={}, timeout_ms=50)


def test_run_python_enforces_memory_limits() -> None:
    sandbox = Sandbox(cpu_ms=500, mem_mb=64, net_enabled=False)
    code = """
chunks = []
for _ in range(256):
    chunks.append("x" * (1024 * 1024))
result = len(chunks)
"""

    with pytest.raises(SandboxMemoryError) as excinfo:
        sandbox.run_python(code, inputs={})

    assert "memory" in str(excinfo.value).lower()


def test_run_python_truncates_logs_above_cap(sandbox: Sandbox) -> None:
    code = "print('x' * 1024)"
    result = sandbox.run_python(code, inputs={}, log_max_bytes=32)

    assert result.stdout.startswith("x")
    assert result.stdout_truncated is True
    assert result.stdout_snippet.endswith("[truncated]")
    suffix_bytes = "… [truncated]".encode()
    assert len(result.stdout_snippet.encode()) <= 32 + len(suffix_bytes)
    assert result.stderr_truncated is False
