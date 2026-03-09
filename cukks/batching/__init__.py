"""
Batching utilities for CKKS encrypted inference.

This module provides tools for efficient batch processing of multiple samples
within a single CKKS ciphertext, leveraging SIMD slot parallelism.

Classes:
    SlotPacker: Pack/unpack multiple samples into CKKS slots.
"""

from .layout import PackingLayout, TokenBlockPacker
from .packing import SlotPacker
from .amcp import AMCPacker
from .dhp import DHPacker
from .rihp import RIHPacker

__all__ = [
    "SlotPacker",
    "PackingLayout",
    "TokenBlockPacker",
    "RIHPacker",
    "AMCPacker",
    "DHPacker",
]
