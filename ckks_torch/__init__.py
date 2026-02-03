"""
CuKKS: PyTorch-compatible Encrypted Deep Learning Inference.

This library provides a seamless interface for running trained PyTorch models
on encrypted data using CKKS homomorphic encryption with GPU acceleration.

Quick Start:
    >>> import torch.nn as nn
    >>> import ckks_torch
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
    >>> enc_model, ctx = ckks_torch.convert(model)
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
    - ckks_torch.nn: Individual encrypted layers
"""

__version__ = "0.1.0"

# Core classes
from .context import CKKSInferenceContext, InferenceConfig
from .tensor import EncryptedTensor
from .converter import convert, estimate_depth, ModelConverter, ConversionOptions

# Submodules
from . import nn
from . import batching
from . import stats
from .batching import SlotPacker

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
        Dictionary with backend information.
    """
    try:
        from ckks import CKKSConfig as _CKKSConfig  # noqa: F401
        from ckks import CKKSContext as _CKKSContext  # noqa: F401
        
        # Use the imports to verify availability
        _ = _CKKSConfig, _CKKSContext
        return {
            "backend": "openfhe-gpu",
            "available": True,
            "cuda": True,  # OpenFHE-GPU always uses CUDA
        }
    except ImportError:
        return {
            "backend": None,
            "available": False,
            "cuda": False,
        }


def is_available() -> bool:
    """Check if the CKKS backend is available.
    
    Returns:
        True if the backend is installed and ready.
    """
    return get_backend_info()["available"]
