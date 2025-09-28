import pytest

@pytest.mark.xfail(reason="MCP server not implemented yet", strict=False)
def test_mcp_envelope_schema_contract():
    # Minimal envelope contract example the server must uphold
    sample = {
        "ok": True,
        "data": {"echo":"ok"},
        "meta": {
            "tool": "web.search.query",
            "version": "1.0.0",
            "durationMs": 1,
            "traceId": "test-trace",
            "warnings": []
        },
        "errors": []
    }
    # In real test: validate against apps/mcp_server/schemas/envelope.schema.json
    assert set(sample) == {"ok", "data", "meta", "errors"}
    assert isinstance(sample["meta"]["durationMs"], int)
