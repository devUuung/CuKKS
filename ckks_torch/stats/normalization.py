"""Encrypted normalization functions: mean, variance, std.

Reference: Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
Statistical Analysis Framework. CIKM'25.

This module provides encrypted statistical functions that operate on
EncryptedTensor objects without revealing the underlying data.

Supports both Mock and OpenFHE backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .crypto_inv_sqrt import crypto_inv_sqrt

if TYPE_CHECKING:
    from ckks_torch.tensor import EncryptedTensor


def _check_size_limit(enc_tensor: EncryptedTensor) -> None:
    """Check v1 size limit (1024).

    Args:
        enc_tensor: Tensor to check

    Raises:
        ValueError: If size > 1024
    """
    size = enc_tensor.shape[0]
    if size > 1024:
        raise ValueError(f"v1: size must be <= 1024, got {size}")


def encrypted_mean(enc_tensor: EncryptedTensor) -> EncryptedTensor:
    """Compute encrypted mean.

    Args:
        enc_tensor: Encrypted input tensor

    Returns:
        EncryptedTensor of shape (1,) containing mean in slot[0]

    Raises:
        ValueError: If size > 1024

    Reference:
        Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
        Statistical Analysis Framework. CIKM'25.
    """
    _check_size_limit(enc_tensor)
    n = enc_tensor.shape[0]
    sum_x = enc_tensor.sum_slots()
    return sum_x.mul(1.0 / n).rescale()


def encrypted_variance(enc_tensor: EncryptedTensor) -> EncryptedTensor:
    """Compute encrypted population variance: Var(X) = E[X^2] - E[X]^2.

    Uses population variance (divisor n, not n-1).

    Args:
        enc_tensor: Encrypted input tensor

    Returns:
        EncryptedTensor of shape (1,) containing variance in slot[0]

    Raises:
        ValueError: If size > 1024

    Note:
        Consumes depth 2: one for squaring, one for scaling.

    Reference:
        Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
        Statistical Analysis Framework. CIKM'25.
    """
    _check_size_limit(enc_tensor)
    n = enc_tensor.shape[0]

    x_sq = enc_tensor.mul(enc_tensor).rescale()
    sum_sq = x_sq.sum_slots()
    e_x_sq = sum_sq.mul(1.0 / n).rescale()

    sum_x = enc_tensor.sum_slots()
    mean = sum_x.mul(1.0 / n).rescale()
    mean_sq = mean.mul(mean).rescale()

    return e_x_sq.sub(mean_sq)


def encrypted_std(
    enc_tensor: EncryptedTensor,
    epsilon: float = 0.1,
) -> EncryptedTensor:
    """Compute encrypted standard deviation: std = sqrt(var + epsilon).

    Args:
        enc_tensor: Encrypted input tensor
        epsilon: Numerical stability constant. Must be >= 0.1 to ensure
            (var + epsilon) falls within crypto_inv_sqrt domain [0.1, 100.0].
            Default 0.1.

    Returns:
        EncryptedTensor of shape (1,) containing std in slot[0]

    Raises:
        ValueError: If size > 1024 or epsilon < 0.1

    Note:
        Uses crypto_inv_sqrt internally, which requires bootstrapping.
        Total depth: variance (2) + inv_sqrt (with 2 bootstraps)

    Reference:
        Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
        Statistical Analysis Framework. CIKM'25.
    """
    _check_size_limit(enc_tensor)

    if epsilon < 0.1:
        raise ValueError(f"epsilon must be >= 0.1, got {epsilon}")

    var = encrypted_variance(enc_tensor)
    var_eps = var.add(epsilon)
    inv_sqrt_var = crypto_inv_sqrt(var_eps)
    return var_eps.mul(inv_sqrt_var).rescale()
