#!/usr/bin/env python3
"""
Codex Utility Script: promote_to_production.py
----------------------------------------------
Reads a production_copy_plan.yaml generated in Phase 4 and safely copies
the finalized implementation and test files from `codex/code/<branch>/`
into their target production locations in the project.

Usage:
    python codex/tools/promote_to_production.py \
        codex/agents/P4/production_copy_plan.yaml [--dry-run] [--force]

Example:
    python codex/tools/promote_to_production.py \
        codex/agents/P4/production_copy_plan.yaml --dry-run
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def load_plan(plan_path: str):
    """Load and validate the production copy plan YAML file."""
    if not os.path.exists(plan_path):
        print(f"❌ Plan file not found: {plan_path}")
        sys.exit(1)
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    if not plan or "actions" not in plan:
        print(f"❌ Invalid or empty copy plan: {plan_path}")
        sys.exit(1)
    return plan


def ensure_dir_exists(path: str):
    """Ensure the destination directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_copy(src: str, dst: str, dry_run=False, force=False):
    """Safely copy files, respecting dry-run and force flags."""
    if not os.path.exists(src):
        print(f"⚠️ Source not found: {src}")
        return False

    if os.path.exists(dst) and not force:
        print(f"⏩ Skipping existing file (use --force to overwrite): {dst}")
        return False

    ensure_dir_exists(dst)

    if dry_run:
        print(f"🧪 [DRY-RUN] Would copy: {src} → {dst}")
        return True

    shutil.copy2(src, dst)
    print(f"✅ Copied: {src} → {dst}")
    return True


def log_summary(log_path: str, summary_data: dict):
    """Append summary of operations to a YAML log file."""
    ensure_dir_exists(log_path)
    with open(log_path, "a") as log:
        yaml.dump(summary_data, log, sort_keys=False)
    print(f"🗂️  Log written to: {log_path}")


# ------------------------------------------------------------
# Main Promotion Process
# ------------------------------------------------------------

def promote(plan_path, dry_run=False, force=False):
    """Execute the promotion based on the YAML plan."""
    plan = load_plan(plan_path)
    actions = plan.get("actions", [])
    task = plan.get("task")
    branch = plan.get("winning_branch", plan.get("branch", "unknown"))

    print("\n🚀 Starting Promotion to Production")
    print(f"📄 Task: {task}")
    print(f"🌿 Branch: {branch}")
    print(f"📦 Actions: {len(actions)}\n")

    results = []
    for action in actions:
        src = action.get("from")
        dst = action.get("to")
        success = safe_copy(src, dst, dry_run=dry_run, force=force)
        results.append({"from": src, "to": dst, "status": "ok" if success else "skipped"})

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "task": task,
        "branch": branch,
        "dry_run": dry_run,
        "force_overwrite": force,
        "files_processed": len(results),
        "results": results,
    }

    log_path = f"codex/POSTEXECUTION/P4/{task}-production_promotion_log.yaml"
    log_summary(log_path, summary)

    print("\n✅ Promotion complete.")
    if dry_run:
        print("💡 (No files were actually moved; use --force to overwrite.)")


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote Codex branch code into production.")
    parser.add_argument("plan", help="Path to production_copy_plan.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without copying files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    promote(args.plan, dry_run=args.dry_run, force=args.force)

