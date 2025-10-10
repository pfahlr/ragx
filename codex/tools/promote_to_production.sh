#!/usr/bin/env bash

# promote_to_production.sh
# -------------------------
# CI-safe wrapper around the Python promotion script.
# Automatically locates and promotes the top-ranked Phase 4 branch code to production.

set -euo pipefail

## 🔧 Config
PLAN_PATH="${1:-}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-true}"  # Overwrite by default

## 📁 Default fallback plan locator (if not passed in)
if [[ -z "$PLAN_PATH" ]]; then
  PLAN_PATH=$(find codex/agents/P4/ -type f -name "*production_copy_plan.yaml" | sort | tail -n 1)
fi

## ✅ Validate
if [[ ! -f "$PLAN_PATH" ]]; then
  echo "❌ Could not find a production_copy_plan.yaml"
  exit 1
fi

echo "🔍 Using copy plan: $PLAN_PATH"
echo "🧪 Dry run: $DRY_RUN"
echo "⚠️ Force overwrite: $FORCE"
echo

## 🧠 Run the promotion
python3 codex/tools/promote_to_production.py "$PLAN_PATH" \
  ${DRY_RUN:+--dry-run} \
  ${FORCE:+--force}

echo "✅ Done."

## ✅ Optional: Git auto-commit
if [[ "${AUTO_COMMIT:-true}" == "true" ]]; then
  echo "📦 Staging promoted files..."
  git add production/*

  COMMIT_MSG="codex: promote {{CODEX_TASK}} top-ranked implementation to production"
  echo "📝 Committing with message: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG" || echo "ℹ️ No changes to commit"

  if [[ "${AUTO_PUSH:-false}" == "true" ]]; then
    echo "🚀 Pushing changes to remote..."
    git push
  else
    echo "🛑 Skipping push (AUTO_PUSH=false)"
  fi
fi

