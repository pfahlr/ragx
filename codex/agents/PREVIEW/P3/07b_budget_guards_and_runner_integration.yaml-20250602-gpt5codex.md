# Phase 3 Preview — Budget Guards & Runner Integration

## Overview
- Implements immutable budget domain models, BudgetManager orchestration, and a TraceWriter-backed FlowRunner slice inside `budget_integration.py`.
- Focuses on deterministic trace payloads with schema enforcement and scope-aware budget enforcement supporting warn vs stop actions.
- Tests target model arithmetic, manager lifecycle, trace schema, and runner loop behaviors using adapter doubles.

## Scope
- Applies only to the codex phase-3 sandbox (`codex/code/07b_budget_guards_and_runner_integration.yaml`).
- No changes to production runner yet; provides validated building blocks and integration slice for subsequent upstream merges.

## Purpose
- Validate the synthesized architecture from Phase 2 across models, manager, trace bridge, and FlowRunner orchestration using TDD.
- Establish confidence in breach diagnostics, stop propagation, and trace chronology before porting into main runner package.
