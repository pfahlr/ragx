# Phase 3 Review Notes – Budget guards and runner integration

## Checklist
- [x] TraceEventEmitter emits mapping-proxy payloads and forwards to optional sink.
- [x] BudgetManager enforces scope lifecycle and raises on duplicate `enter_scope` / missing scope exit.
- [x] BudgetManager emits `budget_breach` before `BudgetBreachError` surfaces and preserves history for inspection.
- [x] FlowRunner applies run + node budgets, emits policy violation traces, and cleans up scopes in `finally` blocks.
- [x] Tests cover warn vs stop semantics, immutable payloads, adapter-driven breach flow, and policy denial handling.

## Notes
- Integration tests rely on concrete `PolicyStack`; no mocks hide enforcement order.
- Warning scenario ensures trace ordering `[budget_breach, budget_charge]` for soft budgets.
- Node breach test confirms run scope never emits `run_complete` when execution halts.
