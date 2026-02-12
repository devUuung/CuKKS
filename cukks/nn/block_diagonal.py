"""
BlockDiagonalLinear — a fully-connected layer whose weight is constrained to
block-diagonal structure.

In the CKKS diagonal method, a block-diagonal weight matrix has only
``2 * block_size - 1`` non-zero diagonals (out of ``dim``), which directly
translates to fewer rotations and plaintext–ciphertext multiplications during
encrypted inference.

Usage (training)::

    model = nn.Sequential(
        BlockDiagonalLinear(256, 256, block_size=32),
        nn.ReLU(),
        BlockDiagonalLinear(256, 10),
    )

The layer is converted to ``EncryptedLinear`` via the standard ``convert()``
pipeline — no special handling is needed because the weight matrix is
*structurally* block-diagonal (zeros everywhere else).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class BlockDiagonalLinear(nn.Module):
    """Linear layer with block-diagonal weight constraint.

    The weight matrix ``W`` of shape ``(out_features, in_features)`` is
    constrained to be block-diagonal:

    .. math::
        W = \\mathrm{diag}(W_1, W_2, \\ldots, W_b)

    where each ``W_i`` is ``(block_size, block_size)`` and
    ``b = in_features // block_size``.

    Parameters
    ----------
    in_features : int
        Input dimension.  Must be divisible by *block_size*.
    out_features : int
        Output dimension.  Must equal *in_features* for a square block-diagonal.
    block_size : int
        Side length of each dense block.
    bias : bool
        If ``True``, add a learnable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if block_size is None:
            block_size = in_features  # no constraint → dense
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"block_size ({block_size})"
            )
        if out_features != in_features:
            raise ValueError(
                f"BlockDiagonalLinear requires in_features == out_features, "
                f"got {in_features} vs {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = in_features // block_size

        # Each block is a (block_size, block_size) parameter
        self.blocks = nn.Parameter(
            torch.empty(self.num_blocks, block_size, block_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_blocks):
            nn.init.kaiming_uniform_(self.blocks[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.block_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        *batch, _ = x.shape
        # Reshape to (..., num_blocks, block_size)
        x = x.view(*batch, self.num_blocks, self.block_size)
        # Batched matmul: each block independently
        # blocks: (num_blocks, block_size, block_size)
        # x:      (..., num_blocks, block_size)
        # out:    (..., num_blocks, block_size)
        out = torch.einsum("...nb,nbo->...no", x, self.blocks)
        out = out.reshape(*batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dense_weight(self) -> torch.Tensor:
        """Reconstruct the full ``(out_features, in_features)`` weight matrix.

        The einsum ``"...nb,nbo->...no"`` computes ``y = x @ B`` per block,
        so the equivalent ``nn.Linear`` weight (which computes ``y = x @ W^T``)
        needs ``W_block = B^T``.
        """
        w = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=self.blocks.dtype,
            device=self.blocks.device,
        )
        bs = self.block_size
        for i in range(self.num_blocks):
            r = i * bs
            w[r : r + bs, r : r + bs] = self.blocks[i].T
        return w

    def to_linear(self) -> nn.Linear:
        """Convert to a standard ``nn.Linear`` (for encrypted conversion)."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        )
        with torch.no_grad():
            linear.weight.copy_(self.to_dense_weight())
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"block_size={self.block_size}, "
            f"num_blocks={self.num_blocks}, "
            f"bias={self.bias is not None}"
        )
