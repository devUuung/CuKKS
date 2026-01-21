"""
Encrypted activation functions using polynomial approximations.

Since CKKS only supports addition and multiplication, non-polynomial
activations must be approximated using polynomials. This module provides
various approximation methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


# =============================================================================
# Polynomial Approximation Coefficients
# =============================================================================

def _chebyshev_relu_coeffs(degree: int = 7, domain: tuple = (-1, 1)) -> List[float]:
    """Compute power-basis polynomial coefficients for ReLU approximation.
    
    Uses least-squares fitting on uniform samples, then converts to power basis.
    """
    try:
        import numpy as np
        a, b = domain
        xs = np.linspace(a, b, 512)
        ys = np.maximum(xs, 0.0)
        cheb_coeffs = np.polynomial.chebyshev.chebfit(xs, ys, degree)
        power_coeffs = np.polynomial.chebyshev.cheb2poly(cheb_coeffs)
        while len(power_coeffs) > 1 and abs(power_coeffs[-1]) < 1e-12:
            power_coeffs = power_coeffs[:-1]
        return power_coeffs.tolist()
    except ImportError:
        return [0.0, 0.5, 0.25]


def _minimax_relu_coeffs(degree: int = 4) -> List[float]:
    """Minimax polynomial approximation for ReLU on [-1, 1].
    
    These are pre-computed optimal coefficients.
    """
    # Pre-computed minimax coefficients for common degrees
    coefficients = {
        2: [0.375, 0.5, 0.125],
        3: [0.3125, 0.5, 0.25, -0.0625],
        4: [0.2734375, 0.5, 0.3125, -0.125, 0.0234375],
    }
    return coefficients.get(degree, _chebyshev_relu_coeffs(degree))


def _gelu_poly_coeffs(degree: int = 4) -> List[float]:
    """Polynomial approximation for GELU: x * Phi(x).
    
    Uses Chebyshev approximation for GELU on [-1, 1].
    """
    try:
        import numpy as np
        from numpy.polynomial import chebyshev
        
        def gelu(x: Any) -> Any:
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
        # Chebyshev approximation on [-1, 1]
        x = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        y = gelu(x)
        coeffs = chebyshev.chebfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        # Fallback: hardcoded degree-4 coefficients
        # GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3)))
        return [0.0, 0.5, 0.0, 0.0398942, 0.0]


def _sigmoid_poly_coeffs(degree: int = 4) -> List[float]:
    """Polynomial approximation for sigmoid on [-4, 4].
    
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    try:
        import numpy as np
        from numpy.polynomial import chebyshev
        
        def sigmoid(x: Any) -> Any:
            return 1.0 / (1.0 + np.exp(-x))
        
        x = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        y = sigmoid(x)
        coeffs = chebyshev.chebfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        return [0.5, 0.25, 0.0, -0.0208333, 0.0]


def _tanh_poly_coeffs(degree: int = 5) -> List[float]:
    """Polynomial approximation for tanh."""
    try:
        import numpy as np
        from numpy.polynomial import chebyshev
        
        x = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        y = np.tanh(x)
        coeffs = chebyshev.chebfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        return [0.0, 1.0, 0.0, -0.333333, 0.0, 0.133333]


def _silu_poly_coeffs(degree: int = 4) -> List[float]:
    """Polynomial approximation for SiLU (x * sigmoid(x))."""
    try:
        import numpy as np
        from numpy.polynomial import chebyshev
        
        def silu(x: Any) -> Any:
            return x / (1.0 + np.exp(-x))
        
        x = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
        y = silu(x)
        coeffs = chebyshev.chebfit(x, y, degree)
        return coeffs.tolist()
    except ImportError:
        return [0.0, 0.5, 0.25, 0.0, -0.0104167]


# =============================================================================
# Activation Modules
# =============================================================================

class EncryptedSquare(EncryptedModule):
    """Square activation: f(x) = x^2.
    
    This is the simplest polynomial activation and is exact in CKKS.
    Often used when training models specifically for encrypted inference.
    """
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        return x.square().rescale()
    
    def mult_depth(self) -> int:
        return 1


class EncryptedReLU(EncryptedModule):
    """Approximate ReLU using polynomial approximation.
    
    Args:
        degree: Degree of the polynomial approximation. Higher = more accurate
            but deeper circuit.
        domain: The interval on which to approximate. Default is [-1, 1].
            Input should be normalized to this range.
        method: Approximation method: "chebyshev" or "minimax".
    
    Note:
        Inputs should be normalized to the specified domain for best accuracy.
    """
    
    def __init__(
        self,
        degree: int = 4,
        domain: tuple = (-4, 4),
        method: str = "chebyshev",
    ) -> None:
        super().__init__()
        self.degree = degree
        self.domain = domain
        self.method = method
        
        if method == "minimax":
            self.coeffs = _minimax_relu_coeffs(degree)
        else:
            self.coeffs = _chebyshev_relu_coeffs(degree, domain)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        # Polynomial evaluation depth is ceil(log2(degree))
        import math
        return max(1, int(math.ceil(math.log2(self.degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={self.degree}, domain={self.domain}, method='{self.method}'"


class EncryptedGELU(EncryptedModule):
    """Approximate GELU using polynomial approximation.
    
    GELU(x) = x * Phi(x) where Phi is the CDF of the standard normal.
    
    Args:
        degree: Degree of the polynomial approximation.
    """
    
    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        self.degree = degree
        self.coeffs = _gelu_poly_coeffs(degree)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        import math
        return max(1, int(math.ceil(math.log2(self.degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class EncryptedSiLU(EncryptedModule):
    """Approximate SiLU (Swish) using polynomial approximation.
    
    SiLU(x) = x * sigmoid(x)
    
    Args:
        degree: Degree of the polynomial approximation.
    """
    
    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        self.degree = degree
        self.coeffs = _silu_poly_coeffs(degree)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        import math
        return max(1, int(math.ceil(math.log2(self.degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class EncryptedSigmoid(EncryptedModule):
    """Approximate sigmoid using polynomial approximation.
    
    Args:
        degree: Degree of the polynomial approximation.
    """
    
    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        self.degree = degree
        self.coeffs = _sigmoid_poly_coeffs(degree)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        import math
        return max(1, int(math.ceil(math.log2(self.degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class EncryptedTanh(EncryptedModule):
    """Approximate tanh using polynomial approximation.
    
    Args:
        degree: Degree of the polynomial approximation.
    """
    
    def __init__(self, degree: int = 5) -> None:
        super().__init__()
        self.degree = degree
        self.coeffs = _tanh_poly_coeffs(degree)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        import math
        return max(1, int(math.ceil(math.log2(self.degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class EncryptedPolynomial(EncryptedModule):
    """Apply a custom polynomial activation.
    
    Useful for models trained with polynomial activations directly.
    
    Args:
        coeffs: Polynomial coefficients [a0, a1, a2, ...] for
                a0 + a1*x + a2*x^2 + ...
    """
    
    def __init__(self, coeffs: Sequence[float]) -> None:
        super().__init__()
        self.coeffs = list(coeffs)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        result = x.poly_eval(self.coeffs)
        return result.rescale()
    
    def mult_depth(self) -> int:
        import math
        degree = len(self.coeffs) - 1
        return max(1, int(math.ceil(math.log2(degree + 1))))
    
    def extra_repr(self) -> str:
        return f"degree={len(self.coeffs) - 1}"
