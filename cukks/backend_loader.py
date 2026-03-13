"""
Backend Loader — Centralised backend discovery and loading.

Consolidates the backend loading logic that was previously split across
``cukks.context._load_backend()`` and ``cukks._cuda_compat``.

Usage::

    from cukks.backend_loader import load_backend

    ConfigClass, ContextClass = load_backend()  # may raise RuntimeError
    cfg = ConfigClass(...)
    ctx = ContextClass(cfg, ...)
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Optional, Tuple, Type

# Cache: (ConfigClass, ContextClass) or None if not yet loaded.
_cache: Optional[Tuple[Type[Any], Type[Any]]] = None
_cache_lock = threading.Lock()


def load_backend() -> Tuple[Type[Any], Type[Any]]:
    """Return ``(CKKSConfig, CKKSContext)`` from the best available backend.

    Resolution order:

    1. **Direct import** — ``from ckks import CKKSConfig, CKKSContext``
       (works when any ``cukks-cu*`` wheel or a dev-editable install of
       ``bindings/openfhe_backend`` is present).
    2. **Auto-install** — calls ``_cuda_compat.auto_install_backend()`` to
       pip-install the correct wheel, then retries the import.

    Raises
    ------
    RuntimeError
        If neither attempt succeeds.  The error message includes
        actionable install instructions.
    """
    global _cache

    # Fast path: already loaded.
    if _cache is not None:
        return _cache

    with _cache_lock:
        # Double-check after acquiring the lock.
        if _cache is not None:
            return _cache

        # ── Attempt 1: direct import ────────────────────────────────
        cfg, ctx = _try_import()
        if cfg is not None and ctx is not None:
            _cache = (cfg, ctx)
            return _cache

        # ── Attempt 2: auto-install + retry ─────────────────────────
        from ._cuda_compat import auto_install_backend

        success, _package, _err = auto_install_backend(quiet=False)
        if success:
            importlib.invalidate_caches()
            cfg, ctx = _try_import()
            if cfg is not None and ctx is not None:
                _cache = (cfg, ctx)
                return _cache

        # ── All attempts failed — build a helpful error ─────────────
        raise RuntimeError(_build_error_message())


def _try_import() -> Tuple[Optional[Type[Any]], Optional[Type[Any]]]:
    """Attempt to import the backend classes.  Returns (None, None) on failure."""
    try:
        from ckks import CKKSConfig, CKKSContext  # type: ignore[import-untyped]
        return CKKSConfig, CKKSContext
    except ImportError:
        return None, None


def _build_error_message() -> str:
    """Produce a user-facing error when no backend is available."""
    from ._cuda_compat import detect_cuda_version, resolve_backend_package

    cuda_ver = detect_cuda_version()
    recommended = resolve_backend_package(cuda_ver) if cuda_ver else None

    if recommended:
        return (
            f"CKKS backend not available. PyTorch uses CUDA {cuda_ver}.\n"
            f"Install the matching backend:\n"
            f"  pip install {recommended}\n"
            f"Or run: python -m cukks.install_backend\n"
            f"\n"
            f"To enable auto-installation, ensure CUKKS_AUTO_INSTALL is not set to '0'."
        )
    if cuda_ver:
        return (
            f"CKKS backend not available. PyTorch uses CUDA {cuda_ver}, "
            f"but no compatible backend was found.\n"
            f"Available backends: pip install cukks-cu118|cu121|cu124|cu128\n"
            f"Or run: python -m cukks.install_backend --status"
        )
    return (
        "CKKS backend not available. No CUDA-enabled PyTorch detected.\n"
        "Install PyTorch with CUDA support first, then:\n"
        "  pip install cukks-cu121  # (match your CUDA version)\n"
        "Or run: python -m cukks.install_backend --status"
    )
