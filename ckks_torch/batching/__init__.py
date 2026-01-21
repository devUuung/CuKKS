"""
Batching utilities for CKKS encrypted inference.

This module provides tools for efficient batch processing of multiple samples
within a single CKKS ciphertext, leveraging SIMD slot parallelism.

Classes:
    SlotPacker: Pack/unpack multiple samples into CKKS slots.
"""

from .packing import SlotPacker

__all__ = ["SlotPacker"]
