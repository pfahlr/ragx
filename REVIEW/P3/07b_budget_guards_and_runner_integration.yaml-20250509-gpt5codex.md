# Review Checklist – Phase 3

## Budget Models & Normalisation
- [x] `normalize_cost` converts seconds to integer milliseconds and rejects unsupported metrics.
- [x] BudgetSpec/BudgetMeter enforce warn/error/stop actions with immutable outcomes.
- [x] `BudgetManager` emits `budget_charge`/`budget_breach` and aggregates warnings.

## Runner Integration
- [x] FlowRunner registers run/node/loop scopes and propagates costs to the BudgetManager.
- [x] Loop execution halts on `BudgetStopSignal` while preserving completed iterations.
- [x] Policy resolution emits allow/deny traces and blocks unknown tools.

## Tests & Determinism
- [x] Pytest modules cover cost utilities, manager semantics, and runner integration.
- [x] Fake adapters provide deterministic cost/output pairs; no network or timing dependencies.
- [x] Trace assertions validate presence/order of `run_start`, `budget_breach`, and `run_end` events.

## Outstanding Considerations
- [ ] Integration with full DSL schema (decision nodes, fallback chains) – deferred.
- [ ] Trace schema validation against shared spec – candidate for future augmentation.

