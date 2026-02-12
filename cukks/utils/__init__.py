"""
Utility functions for CuKKS.
"""

from .approximations import (
    chebyshev_coefficients,
    minimax_coefficients,
    fit_polynomial,
)

__all__ = [
    "chebyshev_coefficients",
    "minimax_coefficients",
    "fit_polynomial",
]
