"""
EncryptedApproxAttention - Approximate attention using polynomial softmax.

For CKKS encrypted inference, exact softmax is not possible.
We approximate softmax using Taylor expansion of exp(x).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _taylor_exp_coeffs(degree: int) -> List[float]:
    """Compute Taylor series coefficients for exp(x).
    
    exp(x) ≈ 1 + x + x²/2! + x³/3! + ... + x^n/n!
    
    Args:
        degree: Maximum degree of polynomial.
        
    Returns:
        List of coefficients [1, 1, 1/2!, 1/3!, ..., 1/n!].
    """
    coeffs = []
    factorial = 1.0
    for i in range(degree + 1):
        if i > 0:
            factorial *= i
        coeffs.append(1.0 / factorial)
    return coeffs


class EncryptedApproxAttention(EncryptedModule):
    """Approximate multi-head attention for encrypted inference.
    
    Uses polynomial approximation for softmax since CKKS only supports
    polynomial operations. The softmax is approximated using Taylor
    expansion of exp(x).
    
    Note:
        This is an approximation - accuracy depends on input range
        and polynomial degree. Best results when attention scores
        are normalized to a small range (e.g., [-2, 2]).
    
    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        softmax_degree: Degree of polynomial for exp approximation.
        
    Example:
        >>> attn = EncryptedApproxAttention(embed_dim=64, num_heads=4)
        >>> output = attn(query, key, value)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        softmax_degree: int = 4,
    ) -> None:
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax_degree = softmax_degree
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Taylor coefficients for exp(x) approximation
        self._exp_coeffs = _taylor_exp_coeffs(softmax_degree)
        
        # Projection weights (initialized as identity for now)
        # These can be set via from_torch or manually
        self.q_weight: Optional[torch.Tensor] = None
        self.k_weight: Optional[torch.Tensor] = None
        self.v_weight: Optional[torch.Tensor] = None
        self.out_weight: Optional[torch.Tensor] = None
        
        self.q_bias: Optional[torch.Tensor] = None
        self.k_bias: Optional[torch.Tensor] = None
        self.v_bias: Optional[torch.Tensor] = None
        self.out_bias: Optional[torch.Tensor] = None
    
    def _apply_projection(
        self,
        x: "EncryptedTensor",
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> "EncryptedTensor":
        """Apply linear projection if weights are set."""
        if weight is None:
            return x
        return x.matmul(weight, bias)
    
    def _approx_exp(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Approximate exp(x) using Taylor series.
        
        exp(x) ≈ 1 + x + x²/2! + x³/3! + ...
        
        Args:
            x: Input encrypted tensor.
            
        Returns:
            Approximation of exp(x).
        """
        return x.poly_eval(self._exp_coeffs)
    
    def _approx_softmax_row(
        self,
        scores: "EncryptedTensor",
        seq_len: int,
    ) -> "EncryptedTensor":
        """Approximate softmax over the last dimension.
        
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
        
        For encrypted computation, we:
        1. Apply polynomial exp approximation
        2. Sum all values
        3. Divide (multiply by approximate inverse)
        
        Note: This is a simplified version that works on flattened scores.
        For production, more sophisticated normalization is needed.
        
        Args:
            scores: Attention scores (flattened).
            seq_len: Sequence length for normalization.
            
        Returns:
            Approximate softmax probabilities.
        """
        # Apply approximate exp
        exp_scores = self._approx_exp(scores)
        
        # For simplicity, we use a fixed normalization factor
        # In practice, we'd compute sum and divide
        # But division in CKKS requires multiplicative inverse approximation
        # Here we use 1/seq_len as a simple approximation
        # This works reasonably for uniform-ish attention
        norm_factor = 1.0 / seq_len
        
        return exp_scores.mul(norm_factor)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        raise NotImplementedError(
            "Use attention(query, key, value) instead of attention(x). "
            "See forward_attention() for the actual implementation."
        )
    
    def forward_attention(
        self,
        query: "EncryptedTensor",
        key: "EncryptedTensor",
        value: "EncryptedTensor",
    ) -> "EncryptedTensor":
        """Forward pass of approximate attention.
        
        Computes: Attention(Q, K, V) = approx_softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Args:
            query: Query tensor, shape (seq_len, embed_dim) or (embed_dim,).
            key: Key tensor, same shape as query.
            value: Value tensor, same shape as query.
            
        Returns:
            Attention output with same shape as input.
        """
        # Apply input projections if weights are set
        q = self._apply_projection(query, self.q_weight, self.q_bias)
        k = self._apply_projection(key, self.k_weight, self.k_bias)
        v = self._apply_projection(value, self.v_weight, self.v_bias)
        
        # Infer sequence length from shape (used for normalization later)
        _ = 1 if len(query.shape) == 1 else query.shape[0]
        
        # For single-head simplified attention:
        # scores = Q @ K^T * scale
        # We need K transposed - for encrypted, we precompute K^T as plaintext
        # or use the encrypted tensor's matmul with transposed weight
        
        # Decrypt K temporarily for matmul (in real impl, this would be encrypted matmul)
        # For now, we simulate attention computation
        k_plain = k._context.decrypt(k)
        k_t = k_plain.T.to(dtype=torch.float64)
        
        # Q @ K^T
        scores = q.matmul(k_t)
        
        # Scale by 1/sqrt(d_k)
        scores = scores.mul(self.scale)
        
        # Apply approximate softmax
        # Get effective seq_len for normalization
        if len(k_plain.shape) == 1:
            effective_seq_len = 1
        else:
            effective_seq_len = k_plain.shape[0]
        
        attn_weights = self._approx_softmax_row(scores, max(effective_seq_len, 1))
        
        # Apply attention weights to values
        v_plain = v._context.decrypt(v)
        v_plain = v_plain.to(dtype=torch.float64)
        if v_plain.ndim == 1:
            v_plain = v_plain.unsqueeze(0)
        
        output = attn_weights.matmul(v_plain)
        
        # Apply output projection if set
        output = self._apply_projection(output, self.out_weight, self.out_bias)
        
        return output
    
    def mult_depth(self) -> int:
        """Estimate multiplicative depth of attention.
        
        Includes:
        - Q/K/V projections: 1 each (if used)
        - Q @ K^T: 1
        - Softmax polynomial: degree
        - attn @ V: 1
        - Output projection: 1 (if used)
        """
        depth = 0
        
        # Input projections
        if self.q_weight is not None:
            depth += 3  # Q, K, V projections
        
        # Q @ K^T
        depth += 1
        
        # Polynomial softmax approximation
        depth += self.softmax_degree
        
        # Attention @ V
        depth += 1
        
        # Output projection
        if self.out_weight is not None:
            depth += 1
        
        return depth
    
    @classmethod
    def from_torch(
        cls,
        attention: torch.nn.MultiheadAttention,
        softmax_degree: int = 4,
    ) -> "EncryptedApproxAttention":
        """Create from PyTorch MultiheadAttention.
        
        Args:
            attention: PyTorch MultiheadAttention module.
            softmax_degree: Degree for softmax polynomial approximation.
            
        Returns:
            EncryptedApproxAttention with copied weights.
        """
        embed_dim = attention.embed_dim
        num_heads = attention.num_heads
        
        enc_attn = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            softmax_degree=softmax_degree,
        )
        
        # Extract weights from PyTorch attention
        # MultiheadAttention stores in_proj_weight as (3*embed_dim, embed_dim)
        # containing [W_q, W_k, W_v] stacked
        if attention.in_proj_weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
            in_proj = attention.in_proj_weight.data.detach().to(dtype=torch.float64)
            enc_attn.q_weight = in_proj[:embed_dim, :]
            enc_attn.k_weight = in_proj[embed_dim:2*embed_dim, :]
            enc_attn.v_weight = in_proj[2*embed_dim:, :]
        elif attention.q_proj_weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
            enc_attn.q_weight = attention.q_proj_weight.data.detach().to(dtype=torch.float64)
            enc_attn.k_weight = attention.k_proj_weight.data.detach().to(dtype=torch.float64)  # pyright: ignore[reportOptionalMemberAccess]
            enc_attn.v_weight = attention.v_proj_weight.data.detach().to(dtype=torch.float64)  # pyright: ignore[reportOptionalMemberAccess]
        
        if attention.in_proj_bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            in_bias = attention.in_proj_bias.data.detach().to(dtype=torch.float64)
            enc_attn.q_bias = in_bias[:embed_dim]
            enc_attn.k_bias = in_bias[embed_dim:2*embed_dim]
            enc_attn.v_bias = in_bias[2*embed_dim:]
        
        if attention.out_proj is not None:  # pyright: ignore[reportUnnecessaryComparison]
            enc_attn.out_weight = attention.out_proj.weight.data.detach().to(dtype=torch.float64)
            if attention.out_proj.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                enc_attn.out_bias = attention.out_proj.bias.data.detach().to(dtype=torch.float64)
        
        return enc_attn
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"softmax_degree={self.softmax_degree}"
        )
