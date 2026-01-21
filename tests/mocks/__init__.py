"""Mock CKKS backend for testing without OpenFHE."""

from .mock_backend import (
    MockCKKSConfig,
    MockCKKSContext,
    MockCKKSTensor,
)

__all__ = [
    "MockCKKSConfig",
    "MockCKKSContext", 
    "MockCKKSTensor",
]
