"""
Encrypted LayerNorm layer.

LayerNorm is implemented via pure HE polynomial approximation for the
inverse square root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from .module import EncryptedModule
from ..stats.crypto_inv_sqrt import _compute_inv_sqrt_coeffs

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedLayerNorm(EncryptedModule):
    """Encrypted LayerNorm layer.
    
    LayerNorm normalizes the input across the feature dimension:
    
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Uses Chebyshev polynomial approximation for the inverse square root,
    avoiding any decryption during inference.
    
    Args:
        normalized_shape: Input shape from an expected input of size
            [*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]].
            If a single integer is provided, it is treated as a single dimension.
        weight: Optional learnable scale parameter (gamma). If None, defaults to ones.
        bias: Optional learnable shift parameter (beta). If None, defaults to zeros.
        eps: A value added to the denominator for numerical stability. Default: 1e-5
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...], torch.Size],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        
        # Convert normalized_shape to list
        if isinstance(normalized_shape, int):
            self.normalized_shape: List[int] = [normalized_shape]
        elif isinstance(normalized_shape, (torch.Size, tuple)):
            self.normalized_shape = list(normalized_shape)  # pyright: ignore[reportAssignmentType]
        else:
            self.normalized_shape = normalized_shape  # pyright: ignore[reportAssignmentType]
        self.eps = eps
        
        # Store weight and bias as float64 on CPU
        if weight is not None:
            self.weight = weight.detach().to(dtype=torch.float64, device="cpu")
        else:
            self.weight = torch.ones(self.normalized_shape, dtype=torch.float64)
        
        if bias is not None:
            self.bias = bias.detach().to(dtype=torch.float64, device="cpu")
        else:
            self.bias = torch.zeros(self.normalized_shape, dtype=torch.float64)
        
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply layer normalization using pure HE polynomial approximation.
        
        Computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
        using only homomorphic operations with Chebyshev polynomial
        approximation for the inverse square root.
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            Encrypted output after layer normalization.
        """
        n = self.normalized_shape[0]
        
        # Step 1: Compute mean via sum_and_broadcast
        # sum_and_broadcast(n) sums first n slots and replicates to all n positions
        mean_broadcast = x.sum_and_broadcast(n).mul(1.0 / n).rescale()
        
        # Step 2: Center the input
        centered = x.sub(mean_broadcast)
        
        # Step 3: Compute variance
        sq = centered.square().rescale()
        var_broadcast = sq.sum_and_broadcast(n).mul(1.0 / n).rescale()
        
        # Step 4: Add epsilon for numerical stability
        var_eps = var_broadcast.add(self.eps)
        
        # Step 5: Compute 1/sqrt(var+eps) using Chebyshev polynomial
        # Map var_eps to Chebyshev domain [-1, 1]
        domain = (0.01, 10.0)
        a, b = domain
        alpha = 2.0 / (b - a)
        beta_map = -(a + b) / (b - a)
        t = var_eps.mul(alpha).add(beta_map)
        
        # Evaluate Chebyshev polynomial for 1/sqrt
        coeffs = _compute_inv_sqrt_coeffs(domain, degree=15)
        inv_std = t.poly_eval(coeffs)
        
        # Step 6: Normalize
        normalized = centered.mul(inv_std).rescale()
        
        # Step 7: Apply scale (gamma) and shift (beta)
        # self.weight is a torch.Tensor (plaintext), self.bias is a torch.Tensor (plaintext)
        output = normalized.mul(self.weight).rescale()
        output = output.add(self.bias)
        
        return output
    
    def mult_depth(self) -> int:
        """Estimated multiplicative depth for pure HE LayerNorm."""
        return 18
    
    @classmethod
    def from_torch(cls, module: torch.nn.LayerNorm) -> "EncryptedLayerNorm":
        """Create from a PyTorch LayerNorm layer.
        
        Args:
            module: The PyTorch LayerNorm layer to convert.
            
        Returns:
            EncryptedLayerNorm.
        """
        # Convert tuple to list to satisfy type checker
        normalized_shape_list: List[int] = list(module.normalized_shape)  # pyright: ignore[reportAssignmentType,reportArgumentType]
        return cls(
            normalized_shape=normalized_shape_list,
            weight=module.weight.data if module.weight is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            eps=module.eps,
        )
    
    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"
