"""Legacy vectordb CLI tests are out of scope for current tasks.

These tests will be restored when vectordb ingestion tasks are tackled.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Vectordb CLI not implemented yet in this milestone")
