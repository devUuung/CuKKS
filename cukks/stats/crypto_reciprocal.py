"""CryptoReciprocal: Encrypted reciprocal using Chebyshev polynomials.

Reference: Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
Statistical Analysis Framework. CIKM'25.

This module implements 1/x approximation for encrypted values using:
1. Chebyshev polynomial approximation (degree 15)

Supports both Mock and OpenFHE backends without bootstrapping.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly, chebfit

if TYPE_CHECKING:
    from cukks.tensor import EncryptedTensor


@functools.lru_cache(maxsize=4)
def _compute_reciprocal_coeffs(
    domain: tuple[float, float], degree: int = 15
) -> list[float]:
    """Compute Chebyshev polynomial coefficients for 1/x.

    Args:
        domain: (a, b) where x in [a, b]
        degree: Polynomial degree (higher = more accurate, default 15)

    Returns:
        Power basis coefficients [c0, c1, ..., c_degree]

    Note:
        Results are cached. First call computes, subsequent calls return cached.
    """
    a, b = domain

    def g(t: float) -> float:
        x = (b - a) * (t + 1) / 2 + a
        return 1.0 / x

    n_nodes = 100
    nodes = np.cos((2 * np.arange(n_nodes) + 1) * np.pi / (2 * n_nodes))
    values = np.array([g(t) for t in nodes])

    cheb_coeffs = chebfit(nodes, values, degree)
    power_coeffs = cheb2poly(cheb_coeffs)

    return power_coeffs.tolist()


def crypto_reciprocal_shallow(
    enc_tensor: EncryptedTensor,
    domain: tuple[float, float] = (0.5, 10.0),
    degree: int = 15,
) -> EncryptedTensor:
    """Compute 1/x using Chebyshev polynomial only (no bootstrap).

    This is a shallow-depth version that works without bootstrapping.
    Uses only polynomial approximation without refinement.

    Args:
        enc_tensor: Encrypted input tensor.
        domain: Input domain (a, b). Narrower domain = better accuracy.
        degree: Polynomial degree (default 15).

    Returns:
        EncryptedTensor containing 1/x approximation.

    Note:
        - No bootstrap required
        - Lower accuracy for wide domains
        - Depth consumption: ~log2(degree) for polynomial evaluation
    """
    x_orig = enc_tensor

    a, b = domain
    alpha = 2.0 / (b - a)
    beta = -(a + b) / (b - a)
    t = x_orig.mul(alpha).rescale().add(beta)

    coeffs = _compute_reciprocal_coeffs(domain, degree)

    y = t.poly_eval(coeffs).rescale()

    return y
