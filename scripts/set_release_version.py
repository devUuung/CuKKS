#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYPROJECT_FILES = [
    ROOT / "pyproject.toml",
    ROOT / "cukks-cu118/pyproject.toml",
    ROOT / "cukks-cu121/pyproject.toml",
    ROOT / "cukks-cu124/pyproject.toml",
    ROOT / "cukks-cu128/pyproject.toml",
    ROOT / "bindings/openfhe_backend/pyproject.toml",
]

PYTHON_FILES = [
    ROOT / "cukks/__init__.py",
]

VERSION_RE = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")$')
INIT_VERSION_RE = re.compile(r'(?m)^(__version__\s*=\s*")([^"]+)(")$')
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9.-]+)?$")


def replace_version(path: Path, pattern: re.Pattern[str], version: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, count = pattern.subn(rf"\1{version}\3", text, count=1)
    if count != 1:
        raise ValueError(f"Expected exactly one version field in {path}")
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def read_version(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Could not find version field in {path}")
    return match.group(2)


def verify(version: str) -> list[str]:
    mismatches: list[str] = []
    for path in PYPROJECT_FILES:
        actual = read_version(path, VERSION_RE)
        if actual != version:
            mismatches.append(f"{path.relative_to(ROOT)}: expected {version}, found {actual}")
    for path in PYTHON_FILES:
        actual = read_version(path, INIT_VERSION_RE)
        if actual != version:
            mismatches.append(f"{path.relative_to(ROOT)}: expected {version}, found {actual}")
    return mismatches


def write(version: str) -> list[str]:
    changed: list[str] = []
    for path in PYPROJECT_FILES:
        if replace_version(path, VERSION_RE, version):
            changed.append(str(path.relative_to(ROOT)))
    for path in PYTHON_FILES:
        if replace_version(path, INIT_VERSION_RE, version):
            changed.append(str(path.relative_to(ROOT)))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync CuKKS release version across package files.")
    parser.add_argument("version", help="Release version without leading v")
    parser.add_argument("--write", action="store_true", help="Write the requested version into tracked files")
    args = parser.parse_args()

    if not SEMVER_RE.match(args.version):
        print(f"Invalid version format: {args.version}", file=sys.stderr)
        return 1

    if args.write:
        changed = write(args.version)
        if changed:
            print("Updated version in:")
            for path in changed:
                print(f"- {path}")
        else:
            print(f"All tracked version files already set to {args.version}")

    mismatches = verify(args.version)
    if mismatches:
        print("Version mismatch detected:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1

    print(f"All tracked version files match {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
