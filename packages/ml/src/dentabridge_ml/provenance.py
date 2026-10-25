"""Which code produced a result."""

import os
import subprocess


def git_commit() -> str:
    """Short hash of HEAD; DENTABRIDGE_COMMIT overrides it for trees copied without .git."""
    if given := os.environ.get("DENTABRIDGE_COMMIT"):
        return given
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"
