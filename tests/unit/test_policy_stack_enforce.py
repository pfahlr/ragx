from __future__ import annotations

import pytest

from pkgs.dsl.policy import PolicyStack, PolicyTraceRecorder, PolicyViolationError

TOOLS = {
    "llm": {"tags": ["analysis"]},
    "retriever": {"tags": ["retrieval", "internal"]},
    "browser": {"tags": ["external", "search"]},
}


def _stack(trace: PolicyTraceRecorder | None = None) -> PolicyStack:
    stack = PolicyStack(tools=TOOLS, tool_sets={"retrieval": ["retriever"]}, trace=trace)
    stack.push({"allow_tools": ["retrieval", "browser"]}, scope="graph")
    return stack


def test_enforce_returns_snapshot_on_success() -> None:
    trace = PolicyTraceRecorder()
    stack = _stack(trace)
    snapshot = stack.enforce("retriever")
    assert snapshot.allowed == frozenset({"retriever"})
    assert snapshot.decisions["retriever"].allowed is True
    assert all(event.event != "policy_violation" for event in trace.events)


def test_enforce_raises_and_traces_on_violation() -> None:
    trace = PolicyTraceRecorder()
    stack = _stack(trace)
    stack.push({"deny_tags": ["external"]}, scope="node")
    with pytest.raises(PolicyViolationError) as exc:
        stack.enforce("browser")
    assert exc.value.denial.tool == "browser"
    assert "deny:tag:external" in exc.value.denial.reasons
    assert trace.events[-1].event == "policy_violation"
    payload = trace.events[-1].data
    assert payload["tool"] == "browser"
    assert payload["stack_depth"] == 2


def test_enforce_optional_violation_return() -> None:
    stack = _stack()
    stack.push({"deny_tools": ["retriever"]}, scope="node")
    snapshot = stack.enforce("retriever", raise_on_violation=False)
    assert "retriever" in snapshot.denied
    assert snapshot.decisions["retriever"].allowed is False
