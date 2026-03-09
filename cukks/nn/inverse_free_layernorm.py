"""
Inverse-Free LayerNorm for CKKS encrypted inference.

Instead of computing 1/σ (expensive in HE), this module outputs a
σ-scaled result that the *next* standard LayerNorm will cancel via
scale-invariance.  See STIP (ePrint 2026/174), Section 5.2.

Algorithm (per-token vector x of dimension n):
    z_i = x_i - mean(x)                     # centering
    v   = sqrt(λ · Σ z_i²)                  # scaled std (no 1/σ)
    y_i = γ · z_i · sqrt(n·λ) + β · v       # inverse-free output

where λ is chosen so that λ·Σz_i² ∈ (0, 2) for the Taylor sqrt.

Depth cost: 5 levels (mean 1, square+sum 1, sqrt 2, final scale 1)
vs. standard EncryptedLayerNorm: 18 levels.

IMPORTANT: This module is only correct when a subsequent standard
LayerNorm exists to cancel the residual σ factor.  The converter's
torch.fx analysis pass ensures this precondition.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedInverseFreeLayerNorm(EncryptedModule):
    """Inverse-free LayerNorm that avoids 1/σ computation.

    Outputs ``γ · z · sqrt(n·λ) + β · v`` where ``v = sqrt(λ · Σz²)``.
    The σ scaling factor is cancelled by a downstream standard LayerNorm.

    Args:
        normalized_shape: Feature dimension(s) to normalize over.
        weight: Learnable scale γ.  Defaults to ones.
        bias: Learnable shift β.  Defaults to zeros.
        eps: Stability constant (unused in forward — kept for API compat).
        lam: Scaling factor λ that constrains λ·Σz² into (0, 2)
             for the second-order Taylor sqrt.  Default 0.01 works
             for typical transformer hidden sizes (256–4096).
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...], torch.Size],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        lam: float = 0.01,
    ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            self.normalized_shape: List[int] = [normalized_shape]
        elif isinstance(normalized_shape, (torch.Size, tuple)):
            self.normalized_shape = list(normalized_shape)  # pyright: ignore[reportAssignmentType]
        else:
            self.normalized_shape = normalized_shape  # pyright: ignore[reportAssignmentType]

        self.eps = eps
        self.lam = lam

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

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply inverse-free layer normalization.

        Computes:  y_i = γ · z_i · sqrt(n·λ) + β · v
        where z_i = (1/n)·Σ_j(x_i − x_j), v = sqrt(λ · Σ z_i²).

        Returns:
            Encrypted tensor carrying a residual σ factor that must be
            cancelled by a subsequent standard LayerNorm.
        """
        n = math.prod(self.normalized_shape)
        lam = self.lam

        # Pre-compute plaintext constants
        gamma_sqrt_nl = (
            self.weight.reshape(-1) * math.sqrt(n * lam)
        ).tolist()
        beta_flat = self.bias.reshape(-1).tolist()

        if getattr(x, "_packed_batch", False):
            return self._forward_packed(x, n, lam, gamma_sqrt_nl, beta_flat)
        return self._forward_single(x, n, lam, gamma_sqrt_nl, beta_flat)

    # ------------------------------------------------------------------ #

    def _forward_single(
        self,
        x: "EncryptedTensor",
        n: int,
        lam: float,
        gamma_sqrt_nl: list[float],
        beta_flat: list[float],
    ) -> "EncryptedTensor":
        # Step 1: Compute z_i = x_i − mean  (depth: 1 for sum_and_broadcast + mul)
        mean_broadcast = x.sum_and_broadcast(n).mul(1.0 / n).rescale()
        z = x.sub(mean_broadcast)

        # Step 2: Compute v = sqrt(λ · Σ z_i²)
        z_sq = z.square().rescale()
        sum_z_sq = z_sq.sum_and_broadcast(n)  # broadcast so v has same shape
        lam_sum = sum_z_sq.mul(lam).rescale()
        v = self._taylor_sqrt(lam_sum)  # depth: 2

        # Step 3: y_i = γ·z_i·sqrt(n·λ) + β·v   (depth: 1)
        term1 = z.mul(gamma_sqrt_nl).rescale()
        term2 = v.mul(beta_flat).rescale()
        result = term1.add(term2)

        # Store σ = v / √(nλ) for downstream bias scaling.
        # v = √(λ·Σz²) = √(λn)·σ, so σ = v · (1/√(nλ)).
        sigma = v.mul(1.0 / math.sqrt(n * lam)).rescale()
        result._sigma_factor = sigma
        return result

    def _forward_packed(
        self,
        x: "EncryptedTensor",
        n: int,
        lam: float,
        gamma_sqrt_nl: list[float],
        beta_flat: list[float],
    ) -> "EncryptedTensor":
        batch_size = getattr(x, "_batch_size", None)
        sample_shape = getattr(x, "_packed_sample_shape", None) or x.shape[1:]
        normalized_tail = tuple(self.normalized_shape)

        if batch_size is None:
            raise RuntimeError(
                "EncryptedInverseFreeLayerNorm packed path requires batch_size"
            )
        if (
            len(sample_shape) < len(normalized_tail)
            or tuple(sample_shape[-len(normalized_tail):]) != normalized_tail
        ):
            raise RuntimeError(
                "packed sample shape must end with normalized_shape="
                f"{normalized_tail}, got {sample_shape}."
            )

        prefix = (
            int(math.prod(sample_shape[: -len(normalized_tail)]))
            if len(sample_shape) > len(normalized_tail)
            else 1
        )
        reshaped = x.view(batch_size, prefix, n)
        avg_block = torch.full((n, n), 1.0 / n, dtype=torch.float64)

        # Step 1: z = x − mean
        mean_broadcast = reshaped.matmul(avg_block)
        z = reshaped.sub(mean_broadcast)

        # Step 2: v = sqrt(λ · Σ z²)
        z_sq = z.square().rescale()
        var_broadcast = z_sq.matmul(avg_block)  # average → broadcast
        # var_broadcast holds (1/n)·Σz² in each slot; we need λ·Σz² = λ·n·var
        lam_sum = var_broadcast.mul(lam * n).rescale()
        v = self._taylor_sqrt(lam_sum)

        # Step 3: y = γ·z·sqrt(nλ) + β·v
        term1 = z.mul(gamma_sqrt_nl).rescale()
        term2 = v.mul(beta_flat).rescale()
        result = term1.add(term2).view(*x.shape)

        # σ = v / √(nλ)  — see _forward_single for derivation
        sigma = v.mul(1.0 / math.sqrt(n * lam)).rescale()
        result._sigma_factor = sigma
        return result

    # ------------------------------------------------------------------ #
    #  sqrt via 2nd-order Taylor (STIP Algorithm 12)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _taylor_sqrt(x: "EncryptedTensor") -> "EncryptedTensor":
        """Approximate sqrt(x) for x ∈ (0, 2) using Taylor expansion.

        sqrt(x) ≈ 1 + 0.5·(x−1) − 0.125·(x−1)²

        Depth cost: 2 levels (one square, one rescale+combine).
        """
        t = x.sub(1.0)                       # t = x − 1
        t1 = t.mul(0.5).rescale()             # 0.5·t
        t_sq = t.square().rescale()            # t²
        t2 = t_sq.mul(0.125).rescale()         # 0.125·t²
        result = t1.sub(t2).add(1.0)           # 1 + 0.5t − 0.125t²
        return result

    # ------------------------------------------------------------------ #

    def mult_depth(self) -> int:
        """Estimated multiplicative depth: 5 levels."""
        return 5

    @classmethod
    def from_torch(cls, module: torch.nn.LayerNorm, lam: float = 0.01) -> "EncryptedInverseFreeLayerNorm":
        """Create from a PyTorch LayerNorm layer."""
        normalized_shape_list: List[int] = list(module.normalized_shape)  # pyright: ignore[reportAssignmentType,reportArgumentType]
        return cls(
            normalized_shape=normalized_shape_list,
            weight=module.weight.data if module.weight is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            eps=module.eps,
            lam=lam,
        )

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, lam={self.lam}, inverse_free=True"
