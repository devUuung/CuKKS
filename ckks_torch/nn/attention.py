"""
EncryptedApproxAttention - Approximate attention using polynomial softmax.

For CKKS encrypted inference, exact softmax is not possible.
We approximate softmax using Taylor expansion of exp(x).
All attention computations run in pure HE using cipher-cipher operations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Union, cast

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
    
    Supports both single-token (seq_len=1) and multi-token (seq_len>1) attention:
    - seq_len=1: Uses Taylor polynomial softmax approximation
    - seq_len>1: Uses Power-Softmax (p=2) with crypto_reciprocal for normalization
      Maximum seq_len is 8 due to multiplicative depth constraints.
    
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

    def _ct_dot(
        self, a: "EncryptedTensor", b: "EncryptedTensor", dim: int
    ) -> "EncryptedTensor":
        """Compute cipher-cipher dot product: sum(a * b)."""
        product = a.mul(b).rescale()
        return product.sum_and_broadcast(dim)

    def _power_softmax(
        self,
        scores: List["EncryptedTensor"],
        shift: float = 2.0,
        eps: float = 0.01,
    ) -> List["EncryptedTensor"]:
        """Compute Power-Softmax (p=2) for attention weights.

        t_j = s_j + shift
        u_j = t_j^2
        Z = sum(u_j)
        w_j = u_j / Z
        """
        from ckks_torch.stats.crypto_reciprocal import crypto_reciprocal_shallow

        shifted = [s.add(shift) for s in scores]
        squared = [t.mul(t).rescale() for t in shifted]

        z_sum = squared[0]
        for u in squared[1:]:
            z_sum = z_sum.add(u)

        z_sum_safe = z_sum.add(eps)
        inv_z_sum = crypto_reciprocal_shallow(z_sum_safe, domain=(0.5, 150.0))

        weights = [u.mul(inv_z_sum).rescale() for u in squared]

        weight_sum = weights[0]
        for weight in weights[1:]:
            weight_sum = weight_sum.add(weight)

        inv_weight_sum = crypto_reciprocal_shallow(weight_sum, domain=(0.5, 10.0))
        weights = [weight.mul(inv_weight_sum).rescale() for weight in weights]
        return weights
    
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
        """Self-attention forward pass.
        
        For self-attention, uses x as query, key, and value.
        Raises NotImplementedError when seq_len > 1.
        """
        return cast("EncryptedTensor", self.forward_attention(x, x, x))
    
    def _forward_attention_multi(
        self,
        query: List["EncryptedTensor"],
        key: List["EncryptedTensor"],
        value: List["EncryptedTensor"],
    ) -> List["EncryptedTensor"]:
        seq_len = len(query)
        
        if seq_len > 8:
            raise NotImplementedError(
                f"seq_len={seq_len} not supported. Maximum is 8 for Power-Softmax attention."
            )
        
        if seq_len != len(key) or seq_len != len(value):
            raise ValueError("query, key, value must have same length.")
        
        q_list = [self._apply_projection(q, self.q_weight, self.q_bias) for q in query]
        k_list = [self._apply_projection(k, self.k_weight, self.k_bias) for k in key]
        v_list = [self._apply_projection(v, self.v_weight, self.v_bias) for v in value]
        
        outputs = []
        for i in range(seq_len):
            scores = []
            for j in range(seq_len):
                s_ij = self._ct_dot(q_list[i], k_list[j], self.embed_dim)
                s_ij = s_ij.mul(self.scale).rescale()
                scores.append(s_ij)
            
            weights = self._power_softmax(scores)
            
            y_i = weights[0].mul(v_list[0]).rescale()
            for j in range(1, seq_len):
                y_i = y_i.add(weights[j].mul(v_list[j]).rescale())
            
            y_i = self._apply_projection(y_i, self.out_weight, self.out_bias)
            outputs.append(y_i)
        
        return outputs

    def forward_attention(
        self,
        query: Union["EncryptedTensor", List["EncryptedTensor"]],
        key: Union["EncryptedTensor", List["EncryptedTensor"]],
        value: Union["EncryptedTensor", List["EncryptedTensor"]],
    ) -> Union["EncryptedTensor", List["EncryptedTensor"]]:
        """Forward pass of approximate attention using pure HE operations.
        
        Computes single-token attention using cipher-cipher multiplications
        and sum_and_broadcast for dot products. No decryption occurs.
        
        Currently supports seq_len=1 only. For seq_len>1, encrypted
        matrix multiplication would require more complex rotation-based
        algorithms.
        
        Args:
            query: Query tensor, shape (embed_dim,) for seq_len=1.
            key: Key tensor, same shape as query.
            value: Value tensor, same shape as query.
            
        Returns:
            Attention output with same shape as input.
            
        Raises:
            NotImplementedError: If seq_len > 1 (input is 2D with first dim > 1).
        """
        if isinstance(query, list):
            if not isinstance(key, list) or not isinstance(value, list):
                raise ValueError("query, key, value must have same length.")
            return self._forward_attention_multi(query, key, value)
        if isinstance(key, list) or isinstance(value, list):
            raise ValueError("query, key, value must have same length.")
        
        # Guard: only seq_len=1 supported
        if len(query.shape) > 1 and query.shape[0] > 1:
            raise NotImplementedError(
                "EncryptedApproxAttention only supports seq_len=1 in pure HE mode. "
                f"Got query shape {query.shape} with seq_len={query.shape[0]}."
            )
        
        # Apply input projections if weights are set
        q = self._apply_projection(query, self.q_weight, self.q_bias)
        k = self._apply_projection(key, self.k_weight, self.k_bias)
        v = self._apply_projection(value, self.v_weight, self.v_bias)
        
        # Q · K (element-wise cipher×cipher, then sum) = dot product
        # For seq_len=1: Q and K are both shape (embed_dim,)
        # Q @ K^T for 1D = sum(Q * K) which is a scalar score
        qk = q.mul(k).rescale()
        scores = qk.sum_and_broadcast(self.embed_dim).mul(self.scale)
        
        # Apply approximate softmax
        # For seq_len=1, softmax of a single score = 1.0, but we still apply
        # the polynomial approximation for consistency
        attn_weights = self._approx_softmax_row(scores, max(1, 1))
        
        # attn_weights · V (element-wise cipher×cipher)
        # For seq_len=1: attn_weight is effectively a scalar broadcast to embed_dim slots
        # Multiplying with V gives the weighted value
        output = attn_weights.mul(v).rescale()
        
        # Apply output projection if set
        output = self._apply_projection(output, self.out_weight, self.out_bias)
        
        return output
    
    def mult_depth(self) -> int:
        """Estimate multiplicative depth of attention.
        
        For seq_len=1 (Taylor softmax):
        - Q/K/V projections: 3 (if used)
        - Q * K (cipher×cipher): 1
        - Softmax polynomial: softmax_degree
        - attn * V (cipher×cipher): 1
        - Output projection: 1 (if used)
        
        For seq_len>1 (Power-Softmax):
        - Q/K/V projections: 3 (if used)
        - Q * K dot products: 1 per pair
        - Power-Softmax: square(1) + reciprocal(~4) + weight_mul(1) ≈ 6
        - Weighted V sum: 1
        - Output projection: 1 (if used)
        
        Returns depth for Power-Softmax case (~8-12 for typical configs).
        """
        depth = 0
        
        # Input projections
        if self.q_weight is not None:
            depth += 3  # Q, K, V projections
        
        # Q * K cipher×cipher
        depth += 1
        
        # Power-Softmax: square + reciprocal + weight multiplication
        # square: 1, reciprocal: ~4, weight_mul: 1
        depth += 6
        
        # Weighted V sum
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
