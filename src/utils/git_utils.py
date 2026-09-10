"""Git metadata used by checkpoints and experiment logs.

See docs/components.md#shared-utilities.
"""

import subprocess

from utils.constants import REPO_PATH


def get_current_commit_hash(strict=False) -> str:
    """Return the full commit hash of the repository checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_PATH, text=True
        ).strip()
    except Exception:
        if strict:
            raise
        return "unknown"
