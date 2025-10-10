# Phase 3 Preview — 07b_budget_guards_and_runner_integration

## Overview
- Established canonical staging package `codex.code.phase3_budget_runner_r7h3.dsl` with shared cost normalization, budget orchestration, trace emission, and runner wiring.
- Budget model aligns with P2 synthesis: immutable `BudgetSpec`, `BudgetCheck`, `BudgetChargeOutcome`, and severity-aware `BudgetAction`.
- Trace emitter wraps `TraceWriter` to produce deterministic `budget_charge`, `budget_breach`, and `budget_commit` payloads that include immutable mappings.

## Test Strategy
- Unit suite validates normalization semantics, warnings vs. stops/errors, multi-scope precedence, and immutability guarantees.
- Integration tests execute FlowRunner through loop and single-shot paths with fake adapters, asserting trace sequencing and stop reasons for warn/stop/error paths.

## Open Risks
- Policy stack interactions limited to resolve/validate hooks; deeper policy trace coverage deferred.
- Cost normalization currently handles time/tokens/requests; extending to spec-only metrics will require additional fixtures.
