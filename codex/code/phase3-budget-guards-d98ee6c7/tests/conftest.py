from __future__ import annotations

import sys
from pathlib import Path

import pytest

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))


@pytest.fixture()
def trace_collector():
    events = []

    class Collector:
        def write(self, event: str, payload):
            events.append((event, payload))

    return Collector(), events
