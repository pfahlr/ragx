from __future__ import annotations

import pytest

from pkgs.dsl.models import PolicyViolationError
from pkgs.dsl.policy import PolicyStack

TOOL_REGISTRY = {
    "gpt": {"tags": ["llm", "analysis"]},
    "web_search": {"tags": ["search", "external"]},
}


@pytest.fixture()
def stack() -> PolicyStack:
    policy_stack = PolicyStack(tool_registry=TOOL_REGISTRY, tool_sets={})
    policy_stack.push({"deny_tags": ["external"]}, scope="globals")
    return policy_stack


def test_enforce_blocks_disallowed_tool(stack: PolicyStack) -> None:
    with pytest.raises(PolicyViolationError) as excinfo:
        stack.enforce("web_search")

    assert excinfo.value.denial.tool == "web_search"
    assert excinfo.value.denial.reason == "denied:tag"


def test_enforce_emits_violation_event(stack: PolicyStack) -> None:
    recorder = stack.recorder
    with pytest.raises(PolicyViolationError):
        stack.enforce("web_search")

    events = recorder.events
    assert events[-1].event == "policy_violation"
    assert events[-1].data["tool"] == "web_search"
    assert events[-1].data["reason"] == "denied:tag"


def test_enforce_can_return_snapshot_without_raise() -> None:
    stack = PolicyStack(tool_registry=TOOL_REGISTRY, tool_sets={})
    stack.push({"deny_tags": ["external"]}, scope="globals")

    snapshot = stack.enforce("web_search", raise_on_violation=False)
    assert snapshot.denied == {"web_search"}
    assert snapshot.denials[0].reason == "denied:tag"

