import importlib
import os

import pytest

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
def test_budget_exhaustion_behaviour():
    run_flow = _find_runner()
    os.environ.setdefault("RAGX_MCP_ENVELOPE_VALIDATION", "shadow")
    os.environ["RAGX_SEED"] = "42"
    os.environ["RAGX_MAX_CALLS"] = "1"  # if supported, runner caps calls

    events = run_flow(FLOW, transport="stdio")
    assert any(
        (e.get("event") == "error" and "budget" in (e.get("code") or ""))
        or (e.get("meta", {}).get("budget_exhausted") is True)
        for e in events
    ), "Runner should indicate budget exhaustion deterministically under tight limits"
