import importlib
import os

import pytest

from tests.helpers.canonical import hash_events

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FLOW = os.path.join(ROOT, "flows", "examples", "multishot_smoke.yaml")

CANDIDATES = [
    "ragx.flows.runner",
    "flows.runner",
    "apps.mcp_server.runner",
    "apps.flows.run",
]

def _find_runner():
    for modname in CANDIDATES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, "run_flow", None)
        if callable(fn):
            return fn
    return None

@pytest.mark.skipif(_find_runner() is None, reason="No flow runner found in known locations")
@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_parity_smoke_shape(transport):
    run_flow = _find_runner()
    os.environ.setdefault("RAGX_SEED", "42")
    os.environ.setdefault("RAGX_MCP_ENVELOPE_VALIDATION", "shadow")

    events = run_flow(FLOW, transport=transport)
    assert isinstance(events, list) and events, "Runner must return a non-empty list of event dicts"
    assert all(isinstance(e, dict) for e in events)

@pytest.mark.skipif(_find_runner() is None, reason="No flow runner found in known locations")
def test_parity_http_vs_stdio_identical_canonical():
    run_flow = _find_runner()
    os.environ["RAGX_SEED"] = "42"
    os.environ.setdefault("RAGX_MCP_ENVELOPE_VALIDATION", "shadow")

    events_stdio = run_flow(FLOW, transport="stdio")
    events_http = run_flow(FLOW, transport="http")

    assert hash_events(events_stdio) == hash_events(events_http), (
        "Canonicalized logs must match across transports"
    )
