"""
BlockDiagLowRankLinear — post-training decomposition of a dense linear layer
into block-diagonal + low-rank residual.

    W ≈ BD(W) + U @ V^T

The block-diagonal part gets CKKS zero-diagonal skip (fewer EvalMult).
The low-rank part uses inner-product + broadcast (2r EvalMult, r reductions).

Usage::

    # Decompose a pre-trained dense layer (no fine-tuning needed)
    dense_linear = model.fc2  # nn.Linear(256, 256)
    bd_lr = BlockDiagLowRankLinear.from_dense(dense_linear, block_size=32, rank=8)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BlockDiagLowRankLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        rank: int,
        blocks: torch.Tensor,
        U_cols: torch.Tensor,
        V_cols: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if in_features != out_features:
            raise ValueError("BlockDiagLowRankLinear requires square weight matrix")
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = in_features // block_size
        self.rank = rank

        self.blocks = nn.Parameter(blocks)   # (num_blocks, bs, bs)
        self.U_cols = nn.Parameter(U_cols)   # (out_features, rank)
        self.V_cols = nn.Parameter(V_cols)   # (in_features, rank)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
        block_size: int,
        rank: int,
    ) -> "BlockDiagLowRankLinear":
        W = linear.weight.detach().float()  # (out, in)
        n = W.shape[0]
        if W.shape[1] != n:
            raise ValueError(f"from_dense requires square weight, got {W.shape}")
        if n % block_size != 0:
            raise ValueError(f"in_features ({n}) not divisible by block_size ({block_size})")

        num_blocks = n // block_size
        bs = block_size

        # Extract diagonal blocks (as nn.Linear convention: W is (out, in))
        bd_weight = torch.zeros_like(W)
        # blocks stored transposed for einsum "...nb,nbo->...no" (y = x @ B)
        blocks = torch.zeros(num_blocks, bs, bs, dtype=W.dtype)
        for i in range(num_blocks):
            r = i * bs
            block_w = W[r : r + bs, r : r + bs]   # (out_block, in_block)
            bd_weight[r : r + bs, r : r + bs] = block_w
            blocks[i] = block_w.T  # transpose for einsum convention

        # Off-block residual
        E = W - bd_weight

        # SVD of residual, absorb S into U
        U_full, S_full, Vt_full = torch.linalg.svd(E, full_matrices=False)
        actual_rank = min(rank, len(S_full))
        U_cols = U_full[:, :actual_rank] * S_full[:actual_rank].unsqueeze(0)  # (n, r)
        V_cols = Vt_full[:actual_rank, :].T  # (n, r)

        bias = linear.bias.detach().float() if linear.bias is not None else None
        return cls(n, n, block_size, actual_rank, blocks, U_cols, V_cols, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, _ = x.shape
        # BD part
        xb = x.view(*batch, self.num_blocks, self.block_size)
        bd_out = torch.einsum("...nb,nbo->...no", xb, self.blocks)
        bd_out = bd_out.reshape(*batch, self.out_features)
        # Low-rank part: U @ V^T @ x
        lr_out = x @ self.V_cols                # (..., rank)
        lr_out = lr_out @ self.U_cols.T         # (..., out_features)
        result = bd_out + lr_out
        if self.bias is not None:
            result = result + self.bias
        return result

    def to_dense_weight(self) -> torch.Tensor:
        """Reconstruct the approximate weight matrix BD(W) + U @ V^T."""
        w = torch.zeros(
            self.out_features, self.in_features,
            dtype=self.blocks.dtype, device=self.blocks.device,
        )
        bs = self.block_size
        for i in range(self.num_blocks):
            r = i * bs
            w[r : r + bs, r : r + bs] = self.blocks[i].T
        w = w + self.U_cols @ self.V_cols.T
        return w

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"block_size={self.block_size}, "
            f"rank={self.rank}, "
            f"num_blocks={self.num_blocks}"
        )
