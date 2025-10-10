# Review Checklist — Phase 3 Budget Guards Integration

## Code Quality
- [x] `BudgetSpec.from_mapping` normalizes floats/ints and converts seconds → ms.
- [x] `BudgetMeter` marks loop stop scopes when limits are reached or exceeded.
- [x] `BudgetManager.commit` emits `budget_commit`, `budget_warning`, and `budget_breach` events with immutable payloads.
- [x] FlowRunner handles policy push/pop symmetry and reports policy violations with scope-aware reasons.
- [x] Loop execution halts on budget stop reasons and honours max_iterations.

## Tests
- [x] Unit tests cover budget models, manager semantics, and runner loop/node policies.
- [x] Tests inject deterministic fake adapters to avoid nondeterminism.
- [x] `conftest.py` adds repo root to `sys.path` for test discovery.

## Follow-ups
- [ ] Extend FlowRunner to support transform/decision nodes.
- [ ] Add schema validation for emitted trace payloads.
- [ ] Provide integration tests combining policy violations and soft budget warnings.
