# Phase 3 Post-Execution – Budget guards and runner integration

## Test Results
- `pytest codex/code/work/tests/test_flow_runner_budget_integration.py -q`
- `pytest codex/code/work/tests/test_flow_runner_loops.py -q`
- `pytest codex/code/work/tests -q`

All suites passed after implementing loop orchestration and policy trace bridging.

## Coverage & Behaviour Notes
- Loop summaries provide `iterations`, `stop_reason`, and optional breach payloads for budget halts.
- Policy denials no longer trigger redundant manual trace events; they flow through the shared emitter via the bridge.
- Scope charging order now includes active loop scopes, ensuring accumulated spend respects layered budgets.
