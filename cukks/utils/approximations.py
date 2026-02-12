"""
Polynomial Approximation Utilities.

Tools for computing polynomial approximations of activation functions
for use in CKKS encrypted inference.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple

import torch

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def chebyshev_coefficients(
    func: Callable[[torch.Tensor], torch.Tensor],
    degree: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    num_samples: int = 256,
) -> List[float]:
    """Compute Chebyshev polynomial approximation coefficients.
    
    Approximates a function using Chebyshev interpolation, which
    often provides good uniform approximation error.
    
    Args:
        func: The function to approximate.
        degree: Degree of the polynomial.
        domain: The interval on which to approximate.
        num_samples: Number of sample points for fitting.
        
    Returns:
        List of polynomial coefficients [a0, a1, ..., an].
        
    Example:
        >>> def relu(x):
        ...     return torch.maximum(x, torch.zeros_like(x))
        >>> coeffs = chebyshev_coefficients(relu, degree=7)
    """
    if np is not None:
        a, b = domain
        
        # Chebyshev nodes
        k = np.arange(1, degree + 2)
        nodes = np.cos(np.pi * (2 * k - 1) / (2 * (degree + 1)))
        
        # Map to domain [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        
        # Evaluate function
        x_tensor = torch.tensor(x, dtype=torch.float64)
        y_tensor = func(x_tensor)
        y = y_tensor.numpy()
        
        # Fit Chebyshev polynomial
        coeffs = np.polynomial.chebyshev.chebfit(nodes, y, degree)
        
        # Convert Chebyshev coefficients to power series
        power_coeffs = np.polynomial.chebyshev.cheb2poly(coeffs)
        
        # Adjust for domain transformation
        if domain != (-1.0, 1.0):
            power_coeffs = _transform_coefficients(power_coeffs, domain)
        
        return power_coeffs.tolist()
    else:
        # Fallback without numpy
        return _chebyshev_fallback(func, degree, domain, num_samples)


def minimax_coefficients(
    func: Callable[[torch.Tensor], torch.Tensor],
    degree: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    num_samples: int = 256,
) -> List[float]:
    """Compute minimax polynomial approximation coefficients.
    
    Minimax approximation minimizes the maximum error (L-infinity norm)
    over the domain. This is often preferred for cryptographic applications.
    
    Args:
        func: The function to approximate.
        degree: Degree of the polynomial.
        domain: The interval on which to approximate.
        num_samples: Number of sample points.
        
    Returns:
        List of polynomial coefficients.
        
    Note:
        Currently uses Chebyshev as an approximation to minimax.
        For true minimax, use the Remez algorithm.
    """
    # For simplicity, we use Chebyshev which is a good approximation to minimax
    # True minimax would use the Remez algorithm
    return chebyshev_coefficients(func, degree, domain, num_samples)


def fit_polynomial(
    func: Callable[[torch.Tensor], torch.Tensor],
    degree: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    num_samples: int = 256,
    method: str = "chebyshev",
) -> List[float]:
    """Fit a polynomial to a function.
    
    Args:
        func: The function to approximate.
        degree: Degree of the polynomial.
        domain: The interval on which to approximate.
        num_samples: Number of sample points.
        method: Approximation method ("chebyshev" or "least_squares").
        
    Returns:
        List of polynomial coefficients [a0, a1, ..., an].
    """
    if method == "chebyshev":
        return chebyshev_coefficients(func, degree, domain, num_samples)
    elif method == "least_squares":
        return _least_squares_fit(func, degree, domain, num_samples)
    else:
        raise ValueError(f"Unknown method: {method}")


def _chebyshev_fallback(
    func: Callable[[torch.Tensor], torch.Tensor],
    degree: int,
    domain: Tuple[float, float],
    num_samples: int,
) -> List[float]:
    """Simple fallback polynomial fitting without numpy."""
    a, b = domain
    
    # Sample points
    x = torch.linspace(a, b, num_samples, dtype=torch.float64)
    y = func(x)
    
    # Simple least squares fit using torch
    X = torch.stack([x ** i for i in range(degree + 1)], dim=1)
    
    # Solve normal equations: (X^T X) c = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    coeffs = torch.linalg.solve(XtX, Xty)
    
    return coeffs.tolist()


def _least_squares_fit(
    func: Callable[[torch.Tensor], torch.Tensor],
    degree: int,
    domain: Tuple[float, float],
    num_samples: int,
) -> List[float]:
    """Least squares polynomial fit."""
    a, b = domain
    
    x = torch.linspace(a, b, num_samples, dtype=torch.float64)
    y = func(x)
    
    X = torch.stack([x ** i for i in range(degree + 1)], dim=1)
    
    # Use QR decomposition for stability
    Q, R = torch.linalg.qr(X)
    coeffs = torch.linalg.solve(R, Q.T @ y)
    
    return coeffs.tolist()


def _transform_coefficients(
    coeffs: Any,
    domain: Tuple[float, float],
) -> Any:
    """Transform polynomial coefficients from [-1,1] to custom domain."""
    if np is None:
        raise ImportError("Numpy is required for _transform_coefficients")
    
    a, b = domain
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    
    n = len(coeffs)
    result = np.zeros(n)
    
    # Transform: p(x) on [-1,1] -> p(scale*x + shift) on [a,b]
    for i, c in enumerate(coeffs):
        for j in range(i + 1):
            binom = math.comb(i, j)
            result[j] += c * binom * (scale ** j) * (shift ** (i - j))
    
    return result


# =============================================================================
# Pre-computed Coefficients for Common Activations
# =============================================================================

RELU_COEFFS = {
    # ReLU approximations on [-1, 1]
    2: [0.375, 0.5, 0.125],
    3: [0.3125, 0.5, 0.25, -0.0625],
    4: [0.2734375, 0.5, 0.3125, -0.125, 0.0234375],
    7: [0.25, 0.5, 0.3125, -0.15625, 0.078125, -0.0390625, 0.01953125, -0.009765625],
}

GELU_COEFFS = {
    # GELU approximations on [-3, 3]
    4: [0.0, 0.5, 0.0, 0.0398942, 0.0],
    6: [0.0, 0.5, 0.0, 0.044715, 0.0, -0.00578, 0.0],
}

SIGMOID_COEFFS = {
    # Sigmoid approximations on [-4, 4]
    3: [0.5, 0.25, 0.0, -0.0104167],
    5: [0.5, 0.25, 0.0, -0.0208333, 0.0, 0.00130208],
}

SWISH_COEFFS = {
    # SiLU/Swish approximations on [-4, 4]
    4: [0.0, 0.5, 0.25, 0.0, -0.0104167],
}


def get_precomputed_coeffs(
    activation: str,
    degree: int,
) -> Optional[List[float]]:
    """Get pre-computed polynomial coefficients.
    
    Args:
        activation: Name of the activation ("relu", "gelu", "sigmoid", "swish").
        degree: Polynomial degree.
        
    Returns:
        List of coefficients or None if not available.
    """
    coeffs_map = {
        "relu": RELU_COEFFS,
        "gelu": GELU_COEFFS,
        "sigmoid": SIGMOID_COEFFS,
        "swish": SWISH_COEFFS,
        "silu": SWISH_COEFFS,
    }
    
    if activation.lower() not in coeffs_map:
        return None
    
    return coeffs_map[activation.lower()].get(degree)
