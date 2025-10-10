import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
BRANCH_ROOT = TEST_DIR.parent
REPO_ROOT = BRANCH_ROOT.parent.parent.parent

for path in (str(REPO_ROOT), str(BRANCH_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
