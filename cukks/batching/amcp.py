from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cukks.tensor import EncryptedTensor

if TYPE_CHECKING:
    from cukks.context import CKKSInferenceContext


class AMCPacker:
    def __init__(self, num_heads: int, seq_len: int, batch_size: int, total_slots: int) -> None:
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if total_slots <= 0:
            raise ValueError(f"total_slots must be positive, got {total_slots}")

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.total_slots = total_slots
        self.k = self.compute_packing_factor(
            num_heads=num_heads,
            seq_len=seq_len,
            batch_size=batch_size,
            total_slots=total_slots,
        )
        self.stride = total_slots // seq_len
        self.width = self.k * batch_size

    @staticmethod
    def compute_packing_factor(num_heads: int, seq_len: int, batch_size: int, total_slots: int) -> int:
        divisors: list[int] = []
        candidate = 1
        while candidate * candidate <= num_heads:
            if num_heads % candidate == 0:
                divisors.append(candidate)
                pair = num_heads // candidate
                if pair != candidate:
                    divisors.append(pair)
            candidate += 1

        valid = [
            factor
            for factor in divisors
            if factor * batch_size * seq_len <= total_slots
        ]
        if not valid:
            raise ValueError(
                "No valid AMCP packing factor: requires k * batch_size * seq_len <= total_slots "
                f"for some k dividing num_heads={num_heads}; got seq_len={seq_len}, "
                f"batch_size={batch_size}, total_slots={total_slots}"
            )
        return max(valid)

    def build_masks(self) -> tuple[list[float], list[float]]:
        t = self.batch_size
        phi = self.stride
        width = self.width

        u_block = [1.0] * (width - t) + [0.0] * t + [0.0] * (phi - width)
        v_block = [0.0] * t + [1.0] * (width - t) + [0.0] * (phi - width)

        u_mask = u_block * self.seq_len
        v_mask = v_block * self.seq_len
        return u_mask, v_mask

    def group_wise_rotation(
        self,
        ciphertext: EncryptedTensor,
        ctx: CKKSInferenceContext,
    ) -> list[EncryptedTensor]:
        u_mask, v_mask = self.build_masks()
        u_enc = ctx.encrypt(torch.tensor(u_mask, dtype=torch.float64))
        v_enc = ctx.encrypt(torch.tensor(v_mask, dtype=torch.float64))

        aligned = [ciphertext]
        for _ in range(self.k - 1):
            current = aligned[-1]
            left = current.rotate(self.batch_size)
            right = current.rotate(-(self.width - self.batch_size))
            next_state = left.mul(u_enc).add(right.mul(v_enc))
            aligned.append(next_state)
        return aligned

    def scale_rotation(self, logical_step: int) -> int:
        return logical_step * self.stride
