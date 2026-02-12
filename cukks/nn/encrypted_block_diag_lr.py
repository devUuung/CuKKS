"""
EncryptedBlockDiagLowRank — encrypted forward for BD + low-rank linear layer.

BD part: standard BSGS matmul with zero-diagonal skip (1 level).
LR part: r inner-product-then-broadcast passes (2 levels).

Total multiplicative depth: 2 (BD at level L-1, LR at level L-2,
OpenFHE EvalAdd handles level matching automatically).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedBlockDiagLowRank(EncryptedModule):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bd_weight: torch.Tensor,
        U_cols: torch.Tensor,
        V_cols: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        rank: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.bd_weight = bd_weight.detach().to(dtype=torch.float64, device="cpu")
        self.U_cols = U_cols.detach().to(dtype=torch.float64, device="cpu")
        self.V_cols = V_cols.detach().to(dtype=torch.float64, device="cpu")
        self.bias = bias.detach().to(dtype=torch.float64, device="cpu") if bias is not None else None

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        # 1. BD matmul via BSGS (zero-diag skip gives speedup) — 1 level
        bd_result = x.matmul(self.bd_weight, bias=None)

        # 2. Low-rank: sum_k u_k * (v_k^T @ x) — 2 levels total
        lr_result = None
        slot_count = x._cipher.size

        for k in range(self.rank):
            v_k: List[float] = [0.0] * slot_count
            for i in range(self.in_features):
                v_k[i] = float(self.V_cols[i, k])

            # cipher × plaintext(v_k) + rescale → level L-1
            inner = x.mul(v_k).rescale()

            # reduce-sum via rotate-and-add (replicates to all slots)
            step = 1
            while step < slot_count:
                rotated = inner.rotate(step)
                inner = inner + rotated
                step *= 2

            # cipher × plaintext(u_k) + rescale → level L-2
            u_k: List[float] = [0.0] * slot_count
            for i in range(self.out_features):
                u_k[i] = float(self.U_cols[i, k])

            rank_term = inner.mul(u_k).rescale()

            if lr_result is None:
                lr_result = rank_term
            else:
                lr_result = lr_result + rank_term

        # 3. Combine BD + LR (OpenFHE EvalAdd handles level matching)
        if lr_result is not None:
            result = bd_result + lr_result
        else:
            result = bd_result

        # 4. Bias
        if self.bias is not None:
            bias_padded: List[float] = [0.0] * slot_count
            for i, v in enumerate(self.bias.tolist()):
                bias_padded[i] = v
            result = result.add(bias_padded)

        return result

    def mult_depth(self) -> int:
        return 2 if self.rank > 0 else 1

    @classmethod
    def from_module(cls, module: "BlockDiagLowRankLinear") -> "EncryptedBlockDiagLowRank":
        from .block_diagonal_low_rank import BlockDiagLowRankLinear

        # Build dense BD weight matrix (zeros in off-blocks → zero-diag skip)
        bd_weight = torch.zeros(
            module.out_features, module.in_features,
            dtype=module.blocks.dtype,
        )
        bs = module.block_size
        for i in range(module.num_blocks):
            r = i * bs
            bd_weight[r : r + bs, r : r + bs] = module.blocks[i].T

        return cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bd_weight=bd_weight,
            U_cols=module.U_cols.detach(),
            V_cols=module.V_cols.detach(),
            bias=module.bias.detach() if module.bias is not None else None,
            rank=module.rank,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}"
        )
