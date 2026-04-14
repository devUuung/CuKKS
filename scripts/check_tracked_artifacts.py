#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import subprocess

TRACKED_ARTIFACT_PATTERNS = (
    "bindings/**/build*/**",
    "build*/**",
    "dist/**",
    "*.egg-info/**",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _matches_artifact(path: str) -> bool:
    normalized = path.rstrip("/")
    for pattern in TRACKED_ARTIFACT_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def main() -> int:
    offenders = sorted(path for path in _tracked_files() if _matches_artifact(path))
    if not offenders:
        print("No tracked build artifacts detected.")
        return 0

    print("Tracked build artifacts detected:")
    for path in offenders:
        print(f" - {path}")
    print("Move local outputs under .artifacts/ or remove them from git.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
