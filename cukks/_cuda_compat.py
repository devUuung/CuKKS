"""
CUDA version detection and backend package resolution for CuKKS.

Used at three stages:
  1. Install-time  (setup.py)              — detect torch CUDA, add matching dependency
  2. Runtime       (cukks/__init__.py)     — warn on CUDA mismatch
  3. CLI           (python -m cukks.install_backend) — post-install auto-detection

Design constraints:
  - No external dependencies beyond stdlib (torch is imported lazily)
  - Safe to import when torch is not yet installed
  - Round-down only: never select a backend built for a *higher* CUDA than torch
"""

from __future__ import annotations

import importlib.metadata
import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "CUDA_TO_PACKAGE",
    "PACKAGE_TO_CUDA",
    "SUPPORTED_CUDA_VERSIONS",
    "detect_cuda_version",
    "resolve_backend_package",
    "get_installed_backend",
    "get_installed_backend_cuda_version",
    "validate_backend_cuda",
]

# ── Package registry ─────────────────────────────────────────────────────────

CUDA_TO_PACKAGE: Dict[str, str] = {
    "11.8": "cukks-cu118",
    "12.1": "cukks-cu121",
    "12.4": "cukks-cu124",
    "12.8": "cukks-cu128",
}

PACKAGE_TO_CUDA: Dict[str, str] = {v: k for k, v in CUDA_TO_PACKAGE.items()}

SUPPORTED_CUDA_VERSIONS: List[str] = sorted(
    CUDA_TO_PACKAGE.keys(),
    key=lambda v: tuple(int(x) for x in v.split(".")),
)

_BACKEND_NAME_RE = re.compile(r"^cukks-cu\d+$")


# ── Version parsing ──────────────────────────────────────────────────────────

def _parse_version(v: str) -> Tuple[int, int]:
    """Parse ``"12.1"`` → ``(12, 1)``.

    Raises :class:`ValueError` on malformed input.
    """
    parts = v.strip().split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected major.minor, got {v!r}")
    return int(parts[0]), int(parts[1])


# ── CUDA detection ───────────────────────────────────────────────────────────

def detect_cuda_version() -> Optional[str]:
    """Return the CUDA version string from PyTorch (e.g. ``"12.1"``), or *None*.

    Returns *None* when:
      - ``torch`` is not installed
      - ``torch`` was built without CUDA (CPU-only)
      - ``torch.version.cuda`` is malformed
    """
    try:
        import torch  # noqa: F811 — intentional lazy import
    except ImportError:
        return None

    cuda_str: Optional[str] = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_str is None:
        return None

    # Normalise to ``"major.minor"`` (torch sometimes returns e.g. ``"12.1.1"``)
    try:
        major, minor = _parse_version(cuda_str)
        return f"{major}.{minor}"
    except (ValueError, IndexError):
        return None


# ── Package resolution ───────────────────────────────────────────────────────

def resolve_backend_package(cuda_version: Optional[str]) -> Optional[str]:
    """Given a CUDA version string, return the best matching ``cukks-cuXXX`` package.

    Resolution strategy (**round-down only**):

    1. Exact match in :data:`CUDA_TO_PACKAGE` → return it.
    2. Same major version: pick the **highest** available minor that is
       **≤** *cuda_version*.
    3. No qualifying candidate → return *None*.

    Examples::

        "12.1"  → "cukks-cu121"   (exact)
        "12.3"  → "cukks-cu121"   (round down: 12.1 is highest ≤ 12.3)
        "12.5"  → "cukks-cu124"   (round down)
        "12.0"  → None            (no 12.x ≤ 12.0 available)
        "11.7"  → None            (no 11.x ≤ 11.7 available)
        None    → None            (CPU-only)
    """
    if cuda_version is None:
        return None

    # Fast path: exact match
    if cuda_version in CUDA_TO_PACKAGE:
        return CUDA_TO_PACKAGE[cuda_version]

    try:
        target_major, target_minor = _parse_version(cuda_version)
    except ValueError:
        return None

    # Collect candidates: same major AND minor ≤ target
    best_minor: Optional[int] = None
    best_key: Optional[str] = None
    for ver_str in CUDA_TO_PACKAGE:
        try:
            cand_major, cand_minor = _parse_version(ver_str)
        except ValueError:
            continue
        if cand_major != target_major:
            continue
        if cand_minor > target_minor:
            continue  # never round up
        if best_minor is None or cand_minor > best_minor:
            best_minor = cand_minor
            best_key = ver_str

    return CUDA_TO_PACKAGE[best_key] if best_key is not None else None


# ── Installed backend detection ──────────────────────────────────────────────

def get_installed_backend() -> Optional[str]:
    """Return the name of the installed ``cukks-cu*`` package, or *None*.

    Scans :mod:`importlib.metadata` for packages whose name matches
    ``cukks-cu\\d+``.
    """
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            if name and _BACKEND_NAME_RE.match(name):
                return name
    except Exception:
        pass
    return None


def get_installed_backend_cuda_version() -> Optional[str]:
    """Return the CUDA version that the installed backend was built for.

    Returns *None* if no backend is installed or its name is not in the
    registry.
    """
    backend = get_installed_backend()
    if backend is None:
        return None
    return PACKAGE_TO_CUDA.get(backend)


# ── Runtime validation ───────────────────────────────────────────────────────

def validate_backend_cuda() -> Tuple[str, Optional[str]]:
    """Validate that the installed backend matches PyTorch's CUDA version.

    Returns:
        ``(status, message)`` where *status* is one of:

        ``"ok"``
            Backend matches torch CUDA version exactly.
        ``"approximate"``
            Backend CUDA is lower than torch's within the same major
            (safe — forward-compatible).
        ``"mismatch"``
            Backend CUDA is **higher** than torch's, or different major
            (**dangerous** — may cause segfaults).
        ``"no_cuda"``
            Torch has no CUDA (CPU-only).  No backend needed.
        ``"no_backend"``
            No ``cukks-cu*`` package installed.
    """
    torch_cuda = detect_cuda_version()
    if torch_cuda is None:
        return "no_cuda", "PyTorch has no CUDA support (CPU-only build)."

    backend_cuda = get_installed_backend_cuda_version()
    if backend_cuda is None:
        recommended = resolve_backend_package(torch_cuda)
        hint = f"  pip install {recommended}" if recommended else ""
        return "no_backend", (
            f"No CuKKS GPU backend installed.  PyTorch uses CUDA {torch_cuda}.\n"
            f"Install with:\n{hint}\n"
            "Or run: python -m cukks.install_backend"
        )

    # Both known — compare
    if backend_cuda == torch_cuda:
        return "ok", f"Backend CUDA {backend_cuda} matches PyTorch CUDA {torch_cuda}."

    try:
        torch_maj, torch_min = _parse_version(torch_cuda)
        back_maj, back_min = _parse_version(backend_cuda)
    except ValueError:
        return "mismatch", (
            f"Cannot parse CUDA versions: torch={torch_cuda}, backend={backend_cuda}."
        )

    backend_pkg = get_installed_backend()
    recommended = resolve_backend_package(torch_cuda)

    if back_maj != torch_maj:
        return "mismatch", (
            f"CUDA major version mismatch: PyTorch uses CUDA {torch_cuda} "
            f"but {backend_pkg} was built for CUDA {backend_cuda}.\n"
            f"Fix: pip install {recommended}" if recommended else ""
        )

    if back_min > torch_min:
        # Backend is HIGHER than torch → dangerous (dual libcudart)
        return "mismatch", (
            f"Backend {backend_pkg} was built for CUDA {backend_cuda}, "
            f"but PyTorch uses CUDA {torch_cuda} (lower).\n"
            "This causes two incompatible libcudart to load → segfaults.\n"
            + (f"Fix: pip install {recommended}" if recommended else "")
        )

    # back_min < torch_min, same major → safe (forward-compatible)
    return "approximate", (
        f"Backend {backend_pkg} (CUDA {backend_cuda}) is compatible with "
        f"PyTorch CUDA {torch_cuda} (same major, backend ≤ torch)."
    )
