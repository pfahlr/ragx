# MCP Server Bootstrap

The MCP server exposes the discovery, prompt, and tool APIs defined in
`codex/specs/ragx_master_spec.yaml`. The bootstrap stage ships both HTTP and
STDIO transports that delegate to a shared `McpService` for schema validation,
prompt loading, and tool execution.

## CLI

Launch the server with the `apps.mcp_server.cli` module (entrypoint
`python -m apps.mcp_server.cli`) or its console script alias `mcp-server`.

```
usage: mcp-server [--http] [--stdio] [--host HOST] [--port PORT]
                  [--max-connections MAX_CONNECTIONS]
                  [--shutdown-grace SECONDS]
                  [--log-level {DEBUG,INFO,WARN,ERROR}]
                  [--once] [--deterministic-ids]
                  [--log-dir PATH]
```

Key switches:

- `--http` / `--stdio` enable the FastAPI or JSON-RPC STDIO transports. If no
  transport flag is provided the HTTP server is enabled by default.
- `--host` (`127.0.0.1` by default) and `--port` (`3333`) configure the HTTP
  bind address.
- `--max-connections` limits concurrent HTTP requests; the value is passed to
  Uvicorn's `limit_concurrency` option.
- `--shutdown-grace` configures the graceful-shutdown timeout (seconds).
- `--once` performs a deterministic handshake (discover, prompt, tool fetch)
  and exits. Combine with `--deterministic-ids` to regenerate golden logs.
- `--log-dir` selects the root directory for structured logs. By default logs
  are written under `runs/mcp_server/`.

The CLI loads toolpacks from `apps/mcp_server/toolpacks`, prompts from
`apps/mcp_server/prompts`, and response schemas from
`apps/mcp_server/schemas/mcp`.

## HTTP transport

`apps.mcp_server.http.main:create_app` returns a FastAPI application with the
following routes:

| Method | Path                      | Handler                 |
| ------ | ------------------------- | ----------------------- |
| GET    | `/mcp/discover`           | `McpService.discover`   |
| GET    | `/mcp/prompt/{promptId}`  | `McpService.get_prompt` |
| POST   | `/mcp/tool/{toolId}`      | `McpService.invoke_tool`|
| GET    | `/healthz`                | `McpService.health`     |

Responses are wrapped in the `Envelope` model and validated against Draft
2020-12 JSON Schemas before returning to the client.

## STDIO transport

`apps.mcp_server.stdio.JsonRpcStdioServer` implements a newline-delimited
JSON-RPC 2.0 loop. Each request is validated and dispatched to the shared
service. The server emits one JSON response per request and flushes stdout
after every write to remain deterministic for tests.

## Structured logging

All requests emit JSONL records through `JsonLogWriter`. Records include the
fields specified in the bootstrap contract (`ts`, `agentId`, `taskId`,
`stepId`, `transport`, `route`, `traceId`, `spanId`, `requestId`, `status`,
`durationMs`, `attempt`, `inputBytes`, `outputBytes`, `metadata`, `error`).
Metadata is enriched with `runId`, `attemptId`, `schemaVersion`,
`deterministic`, and `logPath`. Logs rotate with keep-last-5 retention and a
stable `runs/mcp_server/bootstrap.latest.jsonl` symlink.

The repository includes a golden fixture at
`tests/fixtures/mcp/server/bootstrap_golden.jsonl`. Use the diff tool
`scripts/diff_mcp_server_logs.py` to compare new runs while ignoring the
volatile whitelist (`ts`, `durationMs`, `traceId`, `spanId`, `runId`,
`attemptId`, `requestId`, `logPath`).

## Regenerating the golden log

Run the CLI in deterministic once mode to regenerate the golden log:

```
python -m apps.mcp_server.cli --once --deterministic-ids --log-dir tmp/runs
python scripts/diff_mcp_server_logs.py --new tmp/runs/mcp_server/bootstrap.latest.jsonl \
    --golden tests/fixtures/mcp/server/bootstrap_golden.jsonl
```

After verification, update the fixture and symlink in `tests/fixtures/mcp/logs`.
