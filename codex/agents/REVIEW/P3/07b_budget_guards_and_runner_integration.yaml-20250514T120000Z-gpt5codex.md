# Phase 3 Review — Budget Guards & FlowRunner Integration

## Checklist
- [x] Budget models convert numeric inputs to decimals and freeze payloads.
- [x] `BudgetManager` enforces hard stop, loop stop, and soft warn semantics with preview immutability.
- [x] FlowRunner emits `run_start`, `budget_charge`, `budget_breach`, and `run_end` events with scope metadata.
- [x] Stop reason payloads include scope level, action, remaining, and overages.
- [x] Tests cover hard breach exception, soft warning accumulation, and loop stop halting adapters.

## Notes
- Added `BudgetBreachError.outcomes` to recover partial outcomes for trace emission after hard stops.
- Tests append repo root to `sys.path` to ensure importability when executed in isolation; revisit once repository packaging is standardised.
