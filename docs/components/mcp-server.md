# MCP Tools — Project Summary

Here’s the plain-English version you can drop into your docs.

---

# What this whole thing does (in human terms)

## The problem we’re fixing

We use lots of little “tools” to do research and writing: search the web, fetch documents, check citations, plan outlines, etc. Each one used to behave a bit differently—different inputs, different error messages, sometimes different results for the same request. That made the system harder to trust and harder to extend.

## The simple idea

We built a single “power strip” that all tools plug into. No matter which tool you call, you:

* ask in the same way,
* get answers in the same format,
* and have the same safety rules (time limits, size limits, clear errors).

Apps can talk to this power strip over the web (HTTP) or directly through the command line (STDIO). Under the hood it’s called the **Model Context Protocol (MCP)**, but you can think of it as a universal adapter for tools.

## How it works (without the jargon)

1. **One front door**
   Everything goes through a small server that speaks two “languages”: web requests and command-line messages. Both paths do the exact same thing.

2. **Consistent answers**
   Every tool answers with the same envelope: “Did it work?”, “What’s the data?”, “How long did it take?”, “Any warnings or errors?”. That consistency is what lets us automate checks and build reliable pipelines.

3. **Click-to-add tools (no code changes)**
   New tools are described in a tiny **YAML file**—basically a form that says:

   * the tool’s name and version,
   * what it expects and returns,
   * how to run it (a Python function, a Node/PHP script, a CLI command, or an HTTP call).

   Drop that file in the tools folder and it shows up automatically. If it misbehaves (wrong shape, too slow, output too big), the server blocks it.

4. **Prompts are versioned “templates”**
   The instructions we give models (“prompts”) are treated like files with versions (v1, v2, v3). That means we can pin a specific version in a workflow and know we’ll get the same behavior tomorrow.

5. **Guardrails by default**

   * **Timeouts:** a tool can’t run forever.
   * **Size limits:** it can’t return a book when we asked for a paragraph.
   * **Idempotency:** asking the exact same thing within 10 minutes gives the exact same answer.
   * **Schema checks:** the server verifies each answer is shaped correctly before handing it to you.

6. **Proof it works (tests & CI)**
   We have automated tests that run the tools end-to-end with canned inputs. If a change breaks a rule, CI fails before anything ships.

## What it can do today

* **Search the web** (via connectors) and return clean, comparable result lists.
* **Load documents** by link and hand back text chunks.
* **Query a vector index** (semantic search).
* **Audit citations** (spot missing or malformed references).
* **Render Markdown** (turn structured content into .md text).
* **Plan research** (generate multi-perspective questions and a simple “mind map”).
* **Human-in-the-loop checks** (optionally pause to approve an outline or add questions).
* **Run full research jobs** (plan → retrieve/summarize in parallel → assemble report).

## Why this matters

* **Fewer surprises:** same inputs → same outputs; clear errors when something’s off.
* **Easy to extend:** add a new tool with a YAML file; no core code edits.
* **Auditable:** everything is logged with a trace ID and timing—easy to debug.
* **Safe by design:** time/size limits and strict validation catch most issues early.
* **Future-proof:** prompts and tools are versioned, so upgrades don’t break existing flows.

## A quick example

You ask the system to write a short report on “urban heat islands.”

1. The **planner** generates questions from different angles (history, methods, equity).
2. The **retriever** searches the web and the vector index.
3. An optional **re-ranker** reorders results to push the best sources up.
4. The **citation checker** ensures references are valid.
5. The **publisher** assembles the outline, report, and bibliography.
6. If you want, you hit an **approval step** to tweak the outline before writing.

Every step is just another tool plugged into the same adapter, so the whole thing feels cohesive and predictable.

## What’s next (nice-to-haves)

* **Freshness & reliability scores** for sources.
* **Smarter routing/parallelism** for speed.
* **Visual diffs** of mind-maps between runs.
* **Streaming updates** over the web API for long jobs.

---

**Bottom line:**
This turns a messy pile of scripts into a **clean, reliable toolbox** you can use from any app. It’s easier to add new capabilities, easier to test, and much harder to break.


## Scope & Goal

Expose **all tools and prompts** via **Model Context Protocol (MCP)** with deterministic, versioned behavior suitable for CI gating and automated tests.

---

## Transports

* **HTTP (FastAPI)**

  * Routes:

    * `GET /mcp/discover`
    * `GET /mcp/prompt/:domain/:name/:major` → `{ body, spec }`
    * `POST /mcp/tool/:toolName` → **uniform envelope**
* **STDIO (JSON-RPC 2.0, LSP framing)**

  * Methods:

    * `mcp.discover`
    * `mcp.prompt.get` `{ domain, name, major }`
    * `mcp.tool.invoke` `{ tool, input }`

Both transports call the same service layer; identical contracts.

---

## Uniform Envelope (all tool calls)

```json
{
  "ok": true,
  "data": { /* tool-specific */ },
  "meta": {
    "tool": "web.search.query",
    "version": "1.2.0",
    "durationMs": 123,
    "traceId": "uuid-v4",
    "warnings": []
  },
  "errors": []
}
```

* Canonical error codes: `INVALID_INPUT`, `NOT_FOUND`, `RATE_LIMIT`, `TIMEOUT`, `UNAVAILABLE`, `INTERNAL`, `NONDETERMINISTIC`, `UNSUPPORTED`.

---

## Contracts & Guardrails

* **Idempotency:** identical inputs → identical `{ok,data}` for **10 minutes** (unless tool marked non-deterministic).
* **Timeouts:** default **30s**; on exceed → `TIMEOUT`.
* **Size limits:** inputs ≤ **64KB**, outputs ≤ **256KB** (exceed → `INVALID_INPUT`/`INTERNAL` with `details.size`).
* **Output validation:** every tool’s response is validated against a **JSON Schema**.

---

## Discovery Payload

`GET /mcp/discover` (and `mcp.discover`) returns:

* `server { name, version }`
* `tools[] { name, version, inputSchemaRef, outputSchemaRef, caps { timeoutMs, maxInputBytes, maxOutputBytes, network[] } }`
* `prompts[] { id, versions[] { major, description, specRef } }`
* `health { status, uptimeSec }`

---

## Prompts as First-Class Assets

* Files:
  `prompts/packs/<domain>/<name>.vN.md` (+ `<name>.spec.yaml`)
  Registry: `prompts/REGISTRY.yaml`
* MCP URIs: `mcp.prompt:<domain>/<name>@<major>`
* `mcp.prompt.get`/`GET /mcp/prompt/...` returns `{ body, spec }` (spec includes input schema/constraints).

---

## Core Tool Set (canonical names)

* `mcp.tool:web.search`  → `web.search.query`
* `mcp.tool:docs.load`   → `docs.load.fetch`
* `mcp.tool:vector.query`→ `vector.query.search`
* `mcp.tool:citations.audit` → `citations.audit.check`
* `mcp.tool:exports.render`  → `exports.render.markdown`

(We also defined additional MCP tools for specific workflows: `planner.multi_perspective`, `hitl.outline_review`, `hitl.questions_merge`, and `research.execute`.)

Each tool has versioned **input/output JSON Schemas** under `schemas/tools/`.

---

## Declarative **Toolpacks** (drop-in YAML tools)

Add new tools without code changes by dropping a `.tool.yaml`:

```yaml
id: vector.query.search        # canonical name
version: 1.1.0
deterministic: true
timeoutMs: 30000
limits: { maxInputBytes: 65536, maxOutputBytes: 262144 }
caps: { network: [https] }     # optional

inputSchema:  { $ref: "../../schemas/tools/vector_query_search.input.schema.json" }
outputSchema: { $ref: "../../schemas/tools/vector_query_search.output.schema.json" }

execution:
  kind: cli | python | node | http | php
  # examples:
  # python: module: "pkg.mod:func"  | script: "./tool.py"
  # cli:    cmd: ["curl","-s","https://...","--data-urlencode","q={{input.q}}"]
  # node:   node: "./tool.js"
  # http:   url: "https://svc/run" ; headers: {"Authorization":"Bearer {{env.TOKEN}}"}
  # php:    php: "./tool.php" ; phpBinary: "/usr/bin/php"

env: { passthrough: ["TOKEN"] }
templating: { engine: jinja2, cacheKey: "{{id}}|{{version}}|{{ input | tojson }}" }
onSuccess: { path: "$" }       # map raw → data
```

**Subprocess convention (python/node/php/cli):**

* stdin: **one JSON line** `{"input": <object>}`
* stdout: **one JSON line** for the **data** (or a full envelope; server will extract `.data`).

The server auto-loads all `**/*.tool.yaml`, advertises them in discovery, enforces schemas/time/size, and runs them via the specified `execution.kind`.

---

## REST Facade (thin, optional)

For parity/convenience we mirror some MCP tools via REST (FastAPI), e.g.:

* `POST /v1/search` → normalized search results
* `POST /v1/plan` → planner output (questions, perspectives, graph)
* `POST /v1/research` → planner→executor→publisher results (MVP sync)

(REST uses the same business logic as MCP.)

---

## Testing & CI (Conformance)

* **Pytest** suites (offline, deterministic) covering:

  * Envelope shape, schema validation, determinism, size/time errors
  * Prompt registry loading and `mcp.prompt.get`
  * Toolpacks execution across all `execution.kind` (with fixtures/mocks)
* **CI gate** (GitHub Actions): start HTTP server, run tests, fail on any contract break.
* Optional NDCG/quality checks for retrieval flow and reranking on canned fixtures.

---

## Versioning & Stability

* **SemVer** for tools & prompts (`@major` pin for prompts).
* Keep **N** and **N-1** major versions discoverable simultaneously; deprecate with warnings before removal.

---

### TL;DR

* Two transports (HTTP, STDIO) → same deterministic contracts.
* Uniform envelope + JSON Schema validation for every tool.
* Discovery lists **tools + prompts** with schema refs and caps.
* Drop-in **YAML Toolpacks** (cli/python/node/http/**php**) to add tools without code changes.
* Tests/CI enforce idempotency, size/time limits, and schema conformance.

