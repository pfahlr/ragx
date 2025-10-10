# Phase 3 Preview – Budget guards and runner integration

## Overview
- Branch workspace `codex/code/integrate_budget_guards_runner_p3` stages the canonical budget models, manager, and FlowRunner abstractions from the synthesis plan.
- Trace emission uses the shared `TraceEventEmitter` to capture immutable payloads for budget charges, breaches, run/node lifecycle, and policy violations.
- Tests target cost normalisation, decision blocking, manager lifecycle (including history), and FlowRunner integration with policy enforcement and adapter-backed execution.

## Planned Validation
- Unit suites under `codex/code/integrate_budget_guards_runner_p3/tests/` cover trace emitter, budget model math, manager orchestration, and FlowRunner end-to-end behaviours.
- pytest command: `pytest codex/code/integrate_budget_guards_runner_p3/tests -q`
