"""
Graph analysis utilities for CuKKS model conversion.

This module provides torch.fx-based analysis passes that inspect
PyTorch model graphs to determine which layers qualify for
optimized encrypted conversion (e.g., inverse-free LayerNorm).
"""

from .inverse_free_detect import detect_inverse_free_layernorms

__all__ = ["detect_inverse_free_layernorms"]
