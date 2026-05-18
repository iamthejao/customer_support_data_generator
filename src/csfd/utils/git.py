"""Git provenance helpers — capture the working tree's HEAD SHA for run records."""

from __future__ import annotations

import subprocess


def current_git_sha() -> str | None:
    """Return the SHA-1 of HEAD in the current working tree, or ``None``.

    Returns ``None`` if git is unavailable, the directory is not a git
    repository, or the subprocess fails for any reason. Intended for run-level
    provenance stamping; never raises.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None
