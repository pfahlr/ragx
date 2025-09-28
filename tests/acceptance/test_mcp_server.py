import json, os, socket, time, subprocess, sys, pytest
from pathlib import Path

pytestmark = pytest.mark.xfail(reason="MCP Server not implemented yet")

def _free_port():
    s = socket.socket()
    s.bind(('',0))
    port = s.getsockname()[1]
    s.close()
    return port

def test_mcp_http_envelope_and_discover(tmp_path):
    port = _free_port()
    cmd = ["python","-m","apps.mcp_server.main",
           "--http","--host","127.0.0.1","--port",str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
      # Wait for server
      time.sleep(1.0)
      import urllib.request, json
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/mcp/discover") as r:
          body = json.loads(r.read().decode("utf-8"))
      assert isinstance(body, dict)
      assert "ok" in body and "meta" in body
      assert body["ok"] is True
      assert "tools" in body.get("data",{})
    finally:
      proc.terminate()
      proc.wait(timeout=5)
