"""
EncryptedTTLinear - Tensor Train decomposed encrypted linear layer.

Decomposes weight matrix into TT-cores for reduced parameter count
and fewer rotations in BSGS matmul.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _factorize(n: int) -> List[int]:
    if n < 2:
        return []
    factors: List[int] = []
    remaining = n
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def _next_power_of_two(n: int) -> int:
    return 1 << max(2, (n - 1).bit_length())


def _pad_to_factorizable(n: int) -> int:
    if n < 2:
        return 4
    factors = _factorize(n)
    if len(factors) < 2:
        return _next_power_of_two(n)
    return n


def _balance_factors(
    factors1: List[int],
    factors2: List[int],
    max_cores: int = 4,
) -> Tuple[List[int], List[int]]:
    """Balance two factor lists to have the same length, limited to *max_cores*.

    Having fewer, larger cores keeps the intermediate TT-ranks small relative
    to the matrix dimensions, which avoids rank truncation errors during the
    SVD-based TT decomposition.  A target of 3–4 cores works well in practice
    for dimensions up to ~1024.
    """
    def merge_smallest(values: List[int]) -> List[int]:
        if len(values) < 2:
            return values
        values = sorted(values)
        left = values.pop(0)
        right = values.pop(0)
        values.append(left * right)
        return sorted(values)

    left = sorted(factors1)
    right = sorted(factors2)

    while len(left) > len(right):
        left = merge_smallest(left)
    while len(right) > len(left):
        right = merge_smallest(right)

    while len(left) > max_cores:
        left = merge_smallest(left)
        right = merge_smallest(right)

    return left, right


class EncryptedTTLinear(EncryptedModule):
    """Encrypted linear layer using Tensor Train decomposition.
    
    Decomposes a weight matrix W ∈ ℝ^(out×in) into TT-cores,
    reducing parameter count and enabling faster HE inference
    through smaller sequential matmuls.
    
    Args:
        in_features: Size of input features (possibly padded).
        out_features: Size of output features (possibly padded).
        tt_cores: List of TT-cores, each of shape (r_{k-1}, n_k * m_k, r_k).
        tt_shapes: List of (n_k, m_k) factor pairs for each mode.
        bias: Optional bias vector.
        original_out_features: Original (unpadded) output size.
    
    Example:
        >>> tt_layer = EncryptedTTLinear.from_torch(nn.Linear(784, 128))
        >>> enc_output = tt_layer(enc_input)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_cores: List[torch.Tensor],
        tt_shapes: List[Tuple[int, int]],
        bias: Optional[torch.Tensor] = None,
        original_out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not tt_cores:
            raise ValueError("tt_cores cannot be empty")
        if not tt_shapes:
            raise ValueError("tt_shapes cannot be empty")
        if len(tt_cores) != len(tt_shapes):
            raise ValueError(
                f"tt_cores and tt_shapes must have same length, got {len(tt_cores)} and {len(tt_shapes)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.tt_shapes = tt_shapes
        self._original_out_features = original_out_features or out_features
        
        self.tt_cores: List[torch.Tensor] = []
        for i, core in enumerate(tt_cores):
            core_param = core.detach().to(dtype=torch.float64, device="cpu")
            self.tt_cores.append(core_param)
            self.register_parameter(f"tt_core_{i}", core_param)

        # Pre-compute effective weight matrix
        self._effective_weight = self._reconstruct_weight()
        
        # Store bias
        if bias is not None:
            self.bias = bias.detach().to(dtype=torch.float64, device="cpu")
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix from TT-cores via tensor contraction."""
        # Reshape each core to (r_prev, n_k, m_k, r_next)
        cores_4d = []
        for core, (n_k, m_k) in zip(self.tt_cores, self.tt_shapes):
            r_prev, _, r_next = core.shape
            cores_4d.append(core.reshape(r_prev, n_k, m_k, r_next))

        # Contract sequentially using einsum
        result = cores_4d[0][0]  # (n_1, m_1, r_1) - first core has r_0=1
        for core in cores_4d[1:]:
            # result: (..., r_k), core: (r_k, n_k, m_k, r_{k+1})
            result = torch.einsum("...r,rnmR->...nmR", result, core)

        # Remove trailing rank dimension (should be 1)
        result = result.squeeze(-1)

        # Permute from (n1, m1, n2, m2, ...) to (n1, n2, ..., m1, m2, ...)
        num_cores = len(self.tt_shapes)
        perm = list(range(0, 2 * num_cores, 2)) + list(range(1, 2 * num_cores, 2))
        result = result.permute(*perm).contiguous()

        # Reshape to (out_features, in_features)
        return result.reshape(self.out_features, self.in_features)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass using pre-computed effective weight matrix."""
        # Input validation
        if x.shape and x.shape[0] != self.in_features:
            raise ValueError(
                f"Input size mismatch: expected {self.in_features}, got {x.shape[0]}"
            )
        return x.matmul(self._effective_weight, self.bias)
    
    def mult_depth(self) -> int:
        """Single matmul on pre-computed effective weight."""
        return 1
    
    @classmethod
    def from_torch(
        cls,
        linear: torch.nn.Linear,
        max_rank: Optional[int] = None,
        svd_threshold: float = 1e-6,
    ) -> Optional["EncryptedTTLinear"]:
        """Create from a PyTorch Linear layer using TT decomposition.
        
        Args:
            linear: The PyTorch Linear layer to convert.
            max_rank: Maximum TT-rank (auto-determined if None).
            svd_threshold: Threshold for SVD rank determination (default: 1e-6).
            
        Returns:
            EncryptedTTLinear, or None if layer is too small for TT.
        """
        in_features = linear.in_features
        out_features = linear.out_features

        if in_features * out_features < 1024:
            return None

        padded_in = _pad_to_factorizable(in_features)
        padded_out = _pad_to_factorizable(out_features)

        in_factors = _factorize(padded_in)
        out_factors = _factorize(padded_out)
        in_factors, out_factors = _balance_factors(in_factors, out_factors)

        if not in_factors or not out_factors:
            return None

        padded_in = math.prod(in_factors)
        padded_out = math.prod(out_factors)

        tt_shapes = list(zip(out_factors, in_factors))
        num_cores = len(tt_shapes)

        weight = linear.weight.detach().to(dtype=torch.float64, device="cpu")
        if padded_in != in_features or padded_out != out_features:
            padded_weight = torch.zeros(padded_out, padded_in, dtype=weight.dtype)
            padded_weight[:out_features, :in_features] = weight
            weight = padded_weight

        bias_param = getattr(linear, "bias", None)
        bias = (
            bias_param.detach().to(dtype=torch.float64, device="cpu")
            if bias_param is not None
            else None
        )
        if padded_out != out_features:
            if bias is None:
                bias = torch.zeros(padded_out, dtype=torch.float64)
            else:
                padded_bias = torch.zeros(padded_out, dtype=torch.float64)
                padded_bias[:out_features] = bias
                bias = padded_bias

        weight_tensor = weight.reshape(*out_factors, *in_factors)
        permute_order = []
        for idx in range(num_cores):
            permute_order.append(idx)
            permute_order.append(num_cores + idx)
        weight_tensor = weight_tensor.permute(*permute_order).contiguous()
        weight_tensor = weight_tensor.reshape(*[n * m for n, m in tt_shapes])

        rank_cap = max(1, max_rank) if max_rank is not None else 256

        tt_cores: List[torch.Tensor] = []
        r_prev = 1
        current = weight_tensor
        for k in range(num_cores - 1):
            n_k, m_k = tt_shapes[k]
            left_dim = r_prev * n_k * m_k
            right_dim = current.numel() // left_dim
            matrix = current.reshape(left_dim, right_dim)

            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            if S.numel() == 0 or S[0] == 0:
                rank = 1
            else:
                rel = S / S[0]
                rank = int((rel >= svd_threshold).sum().item())
                if rank < 1:
                    rank = 1

            rank = min(rank, rank_cap, S.numel())

            core = (U[:, :rank] * S[:rank]).reshape(r_prev, n_k * m_k, rank)
            tt_cores.append(core)

            current = Vh[:rank, :]
            r_prev = rank

        n_k, m_k = tt_shapes[-1]
        last_core = current.reshape(r_prev, n_k * m_k, 1)
        tt_cores.append(last_core)

        return cls(
            in_features=padded_in,
            out_features=padded_out,
            tt_cores=tt_cores,
            tt_shapes=tt_shapes,
            bias=bias,
            original_out_features=out_features,
        )
    
    def extra_repr(self) -> str:
        ranks = [self.tt_cores[0].shape[0]] + [c.shape[2] for c in self.tt_cores]
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_cores={len(self.tt_cores)}, "
            f"ranks={ranks}, "
            f"bias={self.bias is not None}"
        )
