"""CryptoInvSqrt: Encrypted inverse square root using Chebyshev + Newton.

Reference: Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
Statistical Analysis Framework. CIKM'25.

This module implements 1/sqrt(x) approximation for encrypted values using:
1. Chebyshev polynomial approximation (degree 15)
2. Newton-Raphson refinement (2 iterations)
3. Bootstrap operations for depth management

Supports both Mock and OpenFHE backends.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly, chebfit

if TYPE_CHECKING:
    from ckks_torch.tensor import EncryptedTensor


@functools.lru_cache(maxsize=4)
def _compute_inv_sqrt_coeffs(
    domain: tuple[float, float], degree: int = 15
) -> list[float]:
    """Compute Chebyshev polynomial coefficients for 1/sqrt(x).

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
        return 1.0 / np.sqrt(x)

    n_nodes = 100
    nodes = np.cos((2 * np.arange(n_nodes) + 1) * np.pi / (2 * n_nodes))
    values = np.array([g(t) for t in nodes])

    cheb_coeffs = chebfit(nodes, values, degree)
    power_coeffs = cheb2poly(cheb_coeffs)

    return power_coeffs.tolist()


def _get_coeffs(domain: tuple[float, float]) -> list[float]:
    """Get cached coefficients for domain."""
    return _compute_inv_sqrt_coeffs(domain, degree=15)


def crypto_inv_sqrt(
    enc_tensor: EncryptedTensor,
    domain: tuple[float, float] = (0.1, 100.0),
) -> EncryptedTensor:
    """Compute 1/sqrt(x) using Chebyshev + Newton approximation.

    This function approximates the inverse square root for encrypted values
    using a combination of Chebyshev polynomial approximation followed by
    Newton-Raphson refinement iterations.

    Args:
        enc_tensor: Encrypted input tensor. Values are ASSUMED to be within
            the specified domain. Runtime validation is not possible due to
            encryption.
        domain: Input domain (a, b). Must be (0.1, 100.0) in v1.
            The domain (0.001, 100.0) from the plan was too wide for polynomial
            approximation - the function varies by 316x. Domain (0.1, 100.0)
            allows accurate approximation with MRE < 1e-3.

    Returns:
        EncryptedTensor containing 1/sqrt(x) approximation.

    Raises:
        NotImplementedError: If domain != (0.1, 100.0)
        RuntimeError: If context does not have enable_bootstrap=True

    Domain Assumption:
        This function ASSUMES input values are within [0.1, 100.0].
        Values outside this range will produce incorrect results.
        Since the input is encrypted, runtime validation is not possible.

    Note:
        Uses 2 bootstrap operations internally:
        - Bootstrap #1 after Chebyshev polynomial evaluation
        - Bootstrap #2 after Newton refinement iterations

    Reference:
        Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
        Statistical Analysis Framework. CIKM'25.

    Example:
        >>> enc_x = ctx.encrypt(torch.tensor([1.0, 4.0, 9.0]))
        >>> enc_inv_sqrt = crypto_inv_sqrt(enc_x)
        >>> # Result approximates [1.0, 0.5, 0.333...]
    """
    if domain != (0.1, 100.0):
        raise NotImplementedError(
            f"v1: domain must be (0.1, 100.0), got {domain}"
        )

    ctx = enc_tensor.context
    config = getattr(ctx, "config", None)
    if config is None or not getattr(config, "enable_bootstrap", False):
        raise RuntimeError(
            "crypto_inv_sqrt requires bootstrapping. "
            "Create context with enable_bootstrap=True"
        )

    x_orig = enc_tensor

    a, b = 0.1, 100.0
    alpha = 2.0 / (b - a)
    beta = -(a + b) / (b - a)
    t = x_orig.mul(alpha).add(beta)

    coeffs = _get_coeffs(domain)

    y = t.poly_eval(coeffs)

    y = y.bootstrap()

    for _ in range(2):
        y_sq = y.mul(y)
        xy_sq = x_orig.mul(y_sq)
        three_minus = xy_sq.mul(-1.0).add(3.0)
        y = y.mul(three_minus).mul(0.5)

    y = y.bootstrap()

    return y


def crypto_inv_sqrt_shallow(
    enc_tensor: EncryptedTensor,
    domain: tuple[float, float] = (1.0, 10.0),
    degree: int = 15,
) -> EncryptedTensor:
    """Compute 1/sqrt(x) using Chebyshev polynomial only (no bootstrap).

    This is a shallow-depth version that works without bootstrapping.
    Uses only polynomial approximation without Newton refinement.
    Suitable for OpenFHE GPU where bootstrap may crash.

    Args:
        enc_tensor: Encrypted input tensor.
        domain: Input domain (a, b). Narrower domain = better accuracy.
            Default (1.0, 10.0) gives good accuracy without bootstrap.
        degree: Polynomial degree (default 15).

    Returns:
        EncryptedTensor containing 1/sqrt(x) approximation.

    Note:
        - No bootstrap required
        - Lower accuracy than full crypto_inv_sqrt (~2% MRE for narrow domain)
        - Depth consumption: ~log2(degree) for polynomial evaluation

    Reference:
        Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
        Statistical Analysis Framework. CIKM'25.
    """
    x_orig = enc_tensor

    a, b = domain
    alpha = 2.0 / (b - a)
    beta = -(a + b) / (b - a)
    t = x_orig.mul(alpha).add(beta)

    coeffs = _compute_inv_sqrt_coeffs(domain, degree)

    y = t.poly_eval(coeffs)

    return y
