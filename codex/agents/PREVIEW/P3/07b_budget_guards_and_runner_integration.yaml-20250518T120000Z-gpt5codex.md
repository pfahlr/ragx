# Phase 3 Preview — Task 07b Budget Guards & Runner Integration

## Scope
- Implement immutable budget domain models (specs, snapshots, breaches) and centralize decision logic.
- Deliver a reusable `BudgetManager` orchestrating preview/commit flows with shared trace emission.
- Provide FlowRunner integration with ToolAdapters, PolicyStack, and trace ordering suitable for policy+budget observability.

## Key Modules
- `budget_models.py` – enums/dataclasses for budgets, costs, breaches, and summaries.
- `trace_emitter.py` – ordered trace collector shared by budgets and policies.
- `budget_manager.py` – scope registration, preview/commit decisions, and warning/stop propagation.
- `policy_stack.py` – minimal policy façade with deterministic trace recording.
- `flow_runner.py` – orchestrates execution loop, hooking policies and budgets to adapters.

## Test Coverage Targets
- `test_budget_manager.py` validates allow/warn/stop decisions, immutability, and error handling.
- `test_flow_runner.py` exercises stop semantics, policy denials, warning accumulation, and trace ordering.

## Integration Notes
- Budget warnings emit on both preview and commit to satisfy observability requirements.
- FlowRunner stops cleanly on budget or policy triggers and emits `run_stop` metadata.
- Result objects expose executed node IDs, collected warnings, and stop reason for downstream use.
