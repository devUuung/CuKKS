"""
CuKKS: PyTorch-compatible Encrypted Deep Learning Inference.

This library provides a seamless interface for running trained PyTorch models
on encrypted data using CKKS homomorphic encryption with GPU acceleration.

Quick Start:
    >>> import torch.nn as nn
    >>> import cukks
    >>> 
    >>> # 1. Train your model normally in PyTorch
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 128),
    ...     nn.ReLU(),
    ...     nn.Linear(128, 10)
    ... )
    >>> train(model, data)
    >>> 
    >>> # 2. Convert to encrypted model
    >>> enc_model, ctx = cukks.convert(model)
    >>> 
    >>> # 3. Encrypt input and run inference
    >>> enc_input = ctx.encrypt(test_input)
    >>> enc_output = enc_model(enc_input)
    >>> 
    >>> # 4. Decrypt output
    >>> output = ctx.decrypt(enc_output)

The library handles:
    - Automatic model conversion with optimizations (BatchNorm folding, etc.)
    - Polynomial approximations for non-linear activations
    - GPU-accelerated CKKS operations via OpenFHE backend
    - Convenient PyTorch-like API

For more control, you can use the lower-level APIs:
    - CKKSInferenceContext: Manage encryption parameters
    - EncryptedTensor: Operate on encrypted data
    - cukks.nn: Individual encrypted layers
"""

__version__ = "0.1.2"

# Core classes
from .context import CKKSInferenceContext, InferenceConfig
from .tensor import EncryptedTensor
from .converter import convert, estimate_depth, warm_cache, ModelConverter, ConversionOptions

# Submodules
from . import nn
from . import batching
from . import stats
from .batching import SlotPacker

# CUDA backend compatibility check (runs once at import time)
from ._cuda_compat import validate_backend_cuda as _validate_backend_cuda


def _check_cuda_compat() -> None:
    """Warn if installed backend doesn't match PyTorch's CUDA version."""
    import warnings

    try:
        status, message = _validate_backend_cuda()
    except Exception:
        return  # Never crash user code on import

    if status == "mismatch":
        warnings.warn(
            f"CuKKS CUDA mismatch: {message}\n"
            "This may cause segfaults. Fix with: python -m cukks.install_backend",
            RuntimeWarning,
            stacklevel=2,
        )


_check_cuda_compat()

__all__ = [
    # Version
    "__version__",
    # Core
    "CKKSInferenceContext",
    "InferenceConfig",
    "EncryptedTensor",
    # Conversion
    "convert",
    "estimate_depth",
    "warm_cache",
    "ModelConverter",
    "ConversionOptions",
    # Batching
    "SlotPacker",
    # Submodules
    "nn",
    "batching",
    "stats",
    # Utility
    "get_backend_info",
    "is_available",
]


def get_backend_info() -> dict:
    """Get information about the CKKS backend.

    Returns:
        Dictionary with keys:
            backend (str | None):          "openfhe-gpu" or None
            available (bool):              Whether the backend is importable
            cuda (bool):                   Whether the backend uses CUDA
            torch_cuda_version (str|None): PyTorch's CUDA version (e.g. "12.8")
            installed_backend (str|None):  Installed cukks-cuXXX package name
            compat_status (str):           "ok" / "approximate" / "mismatch" / "no_cuda" / "no_backend"
    """
    from ._cuda_compat import (
        detect_cuda_version,
        get_installed_backend,
        validate_backend_cuda,
    )

    torch_cuda = detect_cuda_version()
    installed = get_installed_backend()
    status, _msg = validate_backend_cuda()

    try:
        from ckks import CKKSConfig as _CKKSConfig  # noqa: F401
        from ckks import CKKSContext as _CKKSContext  # noqa: F401

        _ = _CKKSConfig, _CKKSContext
        return {
            "backend": "openfhe-gpu",
            "available": True,
            "cuda": True,
            "torch_cuda_version": torch_cuda,
            "installed_backend": installed,
            "compat_status": status,
        }
    except ImportError:
        return {
            "backend": None,
            "available": False,
            "cuda": False,
            "torch_cuda_version": torch_cuda,
            "installed_backend": installed,
            "compat_status": status,
        }


def is_available() -> bool:
    """Check if the CKKS backend is available.
    
    Returns:
        True if the backend is installed and ready.
    """
    return get_backend_info()["available"]
