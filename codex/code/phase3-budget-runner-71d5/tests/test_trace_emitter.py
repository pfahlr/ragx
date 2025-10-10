from types import MappingProxyType

from phase3_budget_runner_71d5.trace import TraceEventEmitter, TraceRecorder


def test_emitter_records_and_forwards_payload():
    recorder = TraceRecorder()
    forwarded = []

    def sink(event):
        forwarded.append(event)

    emitter = TraceEventEmitter(recorder=recorder, sink=sink)
    payload = {"cost": {"tokens": 10}, "mutable": []}
    emitter.emit("budget_charge", scope="node-1", payload=payload)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event.payload, MappingProxyType)
    assert event.payload["cost"] == {"tokens": 10}
    assert forwarded[0] is event

    try:
        event.payload["new"] = 1  # type: ignore[misc]
    except TypeError:
        pass
    else:
        raise AssertionError("payload must be immutable")


def test_policy_sink_adapter_wraps_policy_events():
    recorder = TraceRecorder()
    emitter = TraceEventEmitter(recorder=recorder)
    policy_sink = emitter.policy_sink()

    class DummyPolicyEvent:
        def __init__(self):
            self.event = "policy_push"
            self.scope = "run"
            self.data = {"policy": {"allow_tools": ("echo",)}}

    policy_sink(DummyPolicyEvent())

    assert recorder.events[0].event == "policy_push"
    assert recorder.events[0].payload["policy"] == {"allow_tools": ("echo",)}
