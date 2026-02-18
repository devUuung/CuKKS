"""
CLI tool for managing the CuKKS GPU backend package.

Usage:
    python -m cukks.install_backend              # auto-detect & install
    python -m cukks.install_backend cu128         # install specific backend
    python -m cukks.install_backend 12.8          # install by CUDA version
    python -m cukks.install_backend --status      # show current status

Also available as:
    cukks-install-backend [OPTIONS] [BACKEND]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional

from ._cuda_compat import (
    CUDA_TO_PACKAGE,
    PACKAGE_TO_CUDA,
    SUPPORTED_CUDA_VERSIONS,
    detect_cuda_version,
    get_installed_backend,
    get_installed_backend_cuda_version,
    resolve_backend_package,
    validate_backend_cuda,
)


def _parse_backend_arg(value: str) -> Optional[str]:
    """Normalise a user-provided backend specifier to a package name.

    Accepts:
        "cu128"        → "cukks-cu128"
        "cukks-cu128"  → "cukks-cu128"
        "12.8"         → "cukks-cu128"  (via registry)
    """
    v = value.strip()

    # Already a full package name
    if v.startswith("cukks-cu") and v[8:].isdigit():
        return v

    # Short form: "cu128"
    if v.startswith("cu") and v[2:].isdigit():
        return f"cukks-{v}"

    # Version string: "12.8"
    if "." in v:
        pkg = resolve_backend_package(v)
        if pkg:
            return pkg
        # Exact version not in registry — try CUDA_TO_PACKAGE directly
        return CUDA_TO_PACKAGE.get(v)

    return None


def _print_status() -> None:
    """Print current backend and CUDA compatibility status."""
    torch_cuda = detect_cuda_version()
    backend = get_installed_backend()
    backend_cuda = get_installed_backend_cuda_version()
    status, message = validate_backend_cuda()

    print("CuKKS Backend Status")
    print("=" * 40)
    print(f"  PyTorch CUDA:      {torch_cuda or 'N/A (CPU-only)'}")
    print(f"  Installed backend: {backend or 'None'}")
    print(f"  Backend CUDA:      {backend_cuda or 'N/A'}")
    print(f"  Status:            {status}")
    if message:
        print(f"  Detail:            {message}")
    print()

    if status == "no_backend" and torch_cuda:
        recommended = resolve_backend_package(torch_cuda)
        if recommended:
            print(f"  Recommended: pip install {recommended}")
    elif status == "mismatch":
        recommended = resolve_backend_package(torch_cuda) if torch_cuda else None
        if recommended:
            print(f"  Fix: pip install {recommended}")

    print()
    print("Available backends:")
    for cuda_ver in SUPPORTED_CUDA_VERSIONS:
        pkg = CUDA_TO_PACKAGE[cuda_ver]
        marker = ""
        if backend and pkg == backend:
            marker = " ← installed"
        elif torch_cuda and resolve_backend_package(torch_cuda) == pkg:
            marker = " ← recommended"
        print(f"  {pkg:20s}  (CUDA {cuda_ver}){marker}")


def _install_package(package: str) -> int:
    cmd = [sys.executable, "-m", "pip", "install", package]
    print(f"Running: {' '.join(cmd)}")
    print()
    return subprocess.call(cmd)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for ``cukks-install-backend`` CLI."""
    parser = argparse.ArgumentParser(
        prog="cukks-install-backend",
        description="Install or manage the CuKKS GPU backend package.",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        default=None,
        help=(
            'Backend to install. Accepts: "cu128", "cukks-cu128", "12.8". '
            "If omitted, auto-detects from PyTorch."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current backend status and exit.",
    )
    args = parser.parse_args(argv)

    # ── Status mode ──────────────────────────────────────────────────────
    if args.status:
        _print_status()
        return

    # ── Determine target package ─────────────────────────────────────────
    if args.backend:
        package = _parse_backend_arg(args.backend)
        if package is None:
            print(f"ERROR: Cannot resolve backend from {args.backend!r}.", file=sys.stderr)
            print(f"Valid options: {', '.join(CUDA_TO_PACKAGE.values())}", file=sys.stderr)
            print(f"Or CUDA versions: {', '.join(SUPPORTED_CUDA_VERSIONS)}", file=sys.stderr)
            sys.exit(1)
    else:
        # Auto-detect from PyTorch
        cuda_ver = detect_cuda_version()
        if cuda_ver is None:
            print(
                "ERROR: Cannot auto-detect CUDA version.\n"
                "Either PyTorch is not installed, or it's a CPU-only build.\n"
                "Specify explicitly: cukks-install-backend cu121",
                file=sys.stderr,
            )
            sys.exit(1)

        package = resolve_backend_package(cuda_ver)
        if package is None:
            print(
                f"ERROR: PyTorch uses CUDA {cuda_ver}, but no compatible backend found.\n"
                f"Available: {', '.join(f'{v} → {p}' for v, p in CUDA_TO_PACKAGE.items())}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Detected PyTorch CUDA {cuda_ver} → {package}")

    # ── Check if already installed ───────────────────────────────────────
    current = get_installed_backend()
    if current == package:
        print(f"{package} is already installed.")
        cuda_ver_pkg = PACKAGE_TO_CUDA.get(package, "?")
        print(f"CUDA version: {cuda_ver_pkg}")
        return
    elif current and current != package:
        print(f"Replacing {current} with {package}...")

    # ── Install ──────────────────────────────────────────────────────────
    exit_code = _install_package(package)
    if exit_code != 0:
        print(f"\nInstallation failed (exit code {exit_code}).", file=sys.stderr)
        sys.exit(exit_code)

    print(f"\n✓ {package} installed successfully.")


if __name__ == "__main__":
    main()
