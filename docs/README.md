# RAGX Docs

## Phase 3/6 Budget Guard Deliverable

The Phase 3 consolidation (hardened during Phase 6) brings the budget guard and runner integration into the canonical `pkgs/dsl` package. Key entry points:

| Module | Purpose |
| ------ | ------- |
| `pkgs/dsl/budget_models.py` | Immutable cost snapshots, specs, and decisions with mapping-proxy trace exports. |
| `pkgs/dsl/budget_manager.py` | Scope lifecycle, preview/commit, breach recording, and inspection helpers. |
| `pkgs/dsl/flow_runner.py` | Adapter orchestration that sequences policy allowlists, budget enforcement, and loop control. |
| `pkgs/dsl/trace.py` | Trace event emitter with sink/validator hooks. |

### Getting started

```bash
# Run the dedicated regression suite shipped with the task
pytest codex/code/07b_budget_guards_and_runner_integration.yaml/tests -q

# Execute targeted unit coverage from earlier phases
pytest tests/unit/dsl/test_flow_runner_loop_policy.py -q
```

### Extension hooks

* **Trace sinks** — pass a callable to `TraceEventEmitter.attach_sink` for centralized logging.
* **Trace validators** — use `TraceEventEmitter.attach_validator` to enforce schema invariants (see `test_trace_auto_phase6.py`).
* **Policy telemetry** — instantiate `PolicyStack` with a `PolicyTraceRecorder` or sink to capture push/pop/resolved/violation events.

### Documentation map

* [`docs/07b_budget_guards_and_runner_integration.yaml.md`](07b_budget_guards_and_runner_integration.yaml.md) — detailed design, invariants, and customization guidance.
* [`README.md`](../README.md#4-budget-guards--flowrunner-integration-phase-3) — high-level architecture and acceptance plan.
