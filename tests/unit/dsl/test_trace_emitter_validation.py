import pytest

from pkgs.dsl.trace import TraceEventEmitter


def _budget_validator(event):
    if event.event == "budget_charge" and "spec_name" not in event.payload:
        raise ValueError("budget charge payload missing spec_name")


def test_validator_accepts_valid_budget_event() -> None:
    emitter = TraceEventEmitter()
    captured: list = []
    emitter.attach_validator(_budget_validator)
    emitter.attach_sink(captured.append)

    event = emitter.emit(
        "budget_charge",
        scope_type="run",
        scope_id="run-1",
        payload={"spec_name": "run", "remaining": {}, "overage": {}},
    )

    assert emitter.events == (event,)
    assert captured == [event]


def test_validator_blocks_invalid_event() -> None:
    emitter = TraceEventEmitter()
    captured: list = []
    emitter.attach_validator(_budget_validator)
    emitter.attach_sink(captured.append)

    with pytest.raises(ValueError):
        emitter.emit(
            "budget_charge",
            scope_type="run",
            scope_id="run-1",
            payload={"remaining": {}},
        )

    assert emitter.events == ()
    assert captured == []
