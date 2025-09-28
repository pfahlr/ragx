import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import cast


import pytest

pytestmark = pytest.mark.xfail(reason="MCP Server not implemented yet")

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        address = cast(tuple[str, int], sock.getsockname())
        port = address[1]
    return port


def test_mcp_http_envelope_and_discover(tmp_path: Path) -> None:
    port = _free_port()
    cmd = [
        "python",
        "-m",
        "apps.mcp_server.main",
        "--http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for server
        time.sleep(1.0)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/mcp/discover") as response:
            body = json.loads(response.read().decode("utf-8"))
        assert isinstance(body, dict)
        assert "ok" in body and "meta" in body
        assert body["ok"] is True
        assert "tools" in body.get("data", {})
    finally:
        proc.terminate()
        proc.wait(timeout=5)
