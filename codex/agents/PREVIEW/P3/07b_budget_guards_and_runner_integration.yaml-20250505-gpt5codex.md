# Phase 3 Preview — Budget guards integrated in FlowRunner

## Overview
- Introduced immutable cost and budget dataclasses plus a shared `TraceEventEmitter` for schema-aligned payloads.
- Added `BudgetMeter` and `BudgetManager` to handle preflight/commit cycles and emit `budget_preflight`/`budget_charge`/`budget_breach` events.
- Implemented a deterministic `FlowRunner` prototype that wires adapters, policies, and budgets across run/node/loop scopes while capturing stop reasons and loop halt traces.

## Key Decisions
- `TraceEventEmitter` never relies on truthiness (len==0) to avoid inadvertently discarding injected sinks.
- Loop preflight warnings do not halt execution; enforcement happens on commit to allow deterministic stop-trace emission after real spend is known.
- `FlowRunner` records loop outputs as iteration lists and emits `run_halt`/`loop_halt` signals for downstream observability.

## Test Plan
- `codex/code/work/tests/test_budget_models.py` validates model immutability and meter preview semantics.
- `codex/code/work/tests/test_budget_manager.py` exercises trace emission and hard vs soft budgets.
- `codex/code/work/tests/test_flow_runner_budget_integration.py` covers run-level preflight stops and loop budget halts with policy traces.
