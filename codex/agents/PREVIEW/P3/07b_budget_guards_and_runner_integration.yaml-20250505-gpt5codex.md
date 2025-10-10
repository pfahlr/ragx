# Phase 3 Preview – Budget guards and runner integration

## Overview
- Normalised cost math through `CostSnapshot` with seconds-to-milliseconds conversion and immutable helpers.
- Structured budget models (`BudgetSpec`, `BudgetChargeOutcome`, `BudgetDecision`) driving deterministic trace payloads.
- Scope-aware `BudgetManager` orchestrating preview/commit flows with breach telemetry and history snapshots.
- Adapter-backed `FlowRunner` enforcing policies, applying budgets per scope, and emitting run/node lifecycle traces.

## Planned Validation
- Unit tests for model math, breach stop/warn semantics, and immutable trace payloads.
- Manager tests covering preview/commit, hard-stop exceptions, warn-and-commit behaviour, and scope lifecycle guards.
- Runner integration tests verifying adapter orchestration, policy enforcement, hard-stop interruption, and soft-run warnings.
