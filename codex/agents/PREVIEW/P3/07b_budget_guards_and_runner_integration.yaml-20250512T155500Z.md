# Phase 3 Preview – 07b_budget_guards_and_runner_integration

## Overview
- Established immutable budgeting data classes (`CostSnapshot`, `BudgetSpec`, `BudgetChargeOutcome`) with normalization helpers that coalesce seconds→milliseconds and ignore unset token/call limits.
- Built a `BudgetManager` facade that surfaces `preflight` and `commit` APIs, emits structured trace events (`budget_charge`, `budget_remaining`, `budget_warning`, `budget_breach`), and returns immutable outcomes for inspection.
- Implemented an adapter-backed `FlowRunner` integrating `PolicyStack`, `BudgetManager`, loop handling, and shared `TraceEventEmitter` so hard-stop budgets halt runs/loops while soft budgets warn.

## Key Decisions
- Treat token/call limits as disabled unless explicitly >0 to avoid false-positive breaches when specs omit those keys.
- Emit trace payloads through a shared `TraceEventEmitter`, ensuring policy and budget events share immutable mapping-proxy payloads.
- Use deterministic FakeAdapter fixtures in tests to simulate estimate vs. execute phases and validate stop semantics.

## Testing
- Added `codex/code/work/tests/test_budget_manager.py` covering normalization, preflight hard-stop detection, warnings, and trace ordering.
- Added `codex/code/work/tests/test_flow_runner_budgets.py` covering run/loop/node budgets, policy trace emissions, and stop vs warn behaviour.
