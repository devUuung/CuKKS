"""
Dynamic setup.py that detects PyTorch's CUDA version and adds the matching
cukks-cuXXX backend package as an install dependency.

Environment variable overrides:
    CUKKS_BACKEND=cukks-cu128    Force a specific backend package
    CUKKS_NO_BACKEND=1           Skip backend entirely (CPU-only)

This file works alongside pyproject.toml (PEP 517).  setuptools reads
static metadata from pyproject.toml and merges the dynamic install_requires
produced here.
"""

import os
import sys
import warnings

from setuptools import setup


def _get_cuda_dependency() -> list[str]:
    """Return ``["cukks-cuXXX"]`` matching torch's CUDA, or ``[]``."""

    # ── Override: skip backend entirely ──────────────────────────────────
    if os.environ.get("CUKKS_NO_BACKEND", "").strip() in ("1", "true", "yes"):
        return []

    # ── Override: force specific backend ─────────────────────────────────
    forced = os.environ.get("CUKKS_BACKEND", "").strip()
    if forced:
        # Normalise: accept "cu128", "cukks-cu128", "12.8"
        if forced.startswith("cukks-cu"):
            return [forced]
        if forced.startswith("cu") and forced[2:].isdigit():
            return [f"cukks-{forced}"]
        # Looks like a version string (e.g. "12.8")
        if "." in forced:
            try:
                from cukks._cuda_compat import resolve_backend_package

                pkg = resolve_backend_package(forced)
                if pkg:
                    return [pkg]
            except Exception:
                pass
        warnings.warn(
            f"CUKKS_BACKEND={forced!r} not recognised — ignoring.",
            stacklevel=2,
        )
        return []

    # ── Auto-detect from torch ───────────────────────────────────────────
    try:
        from cukks._cuda_compat import detect_cuda_version, resolve_backend_package
    except ImportError:
        # _cuda_compat itself might not be importable during an sdist build
        # where the package tree hasn't been laid out yet.  Fall back to
        # inline detection.
        try:
            import torch
        except ImportError:
            return []  # torch not installed yet — nothing we can do
        cuda_str = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_str is None:
            return []

        # Inline round-down resolution (mirrors _cuda_compat logic)
        registry = {
            "11.8": "cukks-cu118",
            "12.1": "cukks-cu121",
            "12.4": "cukks-cu124",
            "12.8": "cukks-cu128",
        }
        parts = cuda_str.strip().split(".")
        if len(parts) < 2:
            return []
        try:
            target_major, target_minor = int(parts[0]), int(parts[1])
        except ValueError:
            return []

        if cuda_str in registry:
            return [registry[cuda_str]]

        best = None
        for ver, pkg in registry.items():
            vparts = ver.split(".")
            cmaj, cmin = int(vparts[0]), int(vparts[1])
            if cmaj != target_major or cmin > target_minor:
                continue
            if best is None or cmin > best[0]:
                best = (cmin, pkg)
        return [best[1]] if best else []

    cuda_ver = detect_cuda_version()
    if cuda_ver is None:
        return []  # CPU-only torch or torch not installed

    pkg = resolve_backend_package(cuda_ver)
    if pkg is None:
        print(
            f"WARNING: PyTorch uses CUDA {cuda_ver} but no compatible CuKKS "
            f"backend found. Install manually: pip install cukks-cuXXX",
            file=sys.stderr,
        )
        return []

    return [pkg]


setup(
    install_requires=_get_cuda_dependency(),
)
