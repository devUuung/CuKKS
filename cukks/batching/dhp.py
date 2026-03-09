from __future__ import annotations

import torch

from cukks.tensor import EncryptedTensor


class DHPacker:
    def __init__(self, num_heads: int) -> None:
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_heads % 2 != 0:
            raise ValueError(f"num_heads must be even for DHP, got {num_heads}")
        self.num_heads = num_heads
        self.num_pairs = num_heads // 2

    @staticmethod
    def precode_weights(w_h: torch.Tensor, w_h1: torch.Tensor) -> torch.Tensor:
        if w_h.shape != w_h1.shape:
            raise ValueError(
                "w_h and w_h1 must have same shape, "
                f"got {tuple(w_h.shape)} and {tuple(w_h1.shape)}"
            )
        w_h_complex = w_h.to(dtype=torch.complex128)
        w_h1_complex = w_h1.to(dtype=torch.complex128)
        return w_h_complex + 1j * w_h1_complex

    def parallel_projection(
        self,
        inputs: list[EncryptedTensor],
        precoded_weights: list[list[float | complex]],
    ) -> list[EncryptedTensor]:
        if len(inputs) != len(precoded_weights):
            raise ValueError(
                "inputs and precoded_weights must have same length, "
                f"got {len(inputs)} and {len(precoded_weights)}"
            )
        if not inputs:
            return []
        if not precoded_weights[0]:
            return []

        out_dim = len(precoded_weights[0])
        for row_idx, row in enumerate(precoded_weights):
            if len(row) != out_dim:
                raise ValueError(
                    "all rows in precoded_weights must have same length, "
                    f"row 0 has {out_dim} and row {row_idx} has {len(row)}"
                )

        packed_outputs: list[EncryptedTensor] = []
        for col in range(out_dim):
            first_weight = complex(precoded_weights[0][col])
            acc = inputs[0].mul(first_weight.real)
            if first_weight.imag != 0.0:
                acc = acc.add(inputs[0].mul(first_weight.imag).mul_by_i())

            for row in range(1, len(inputs)):
                weight = complex(precoded_weights[row][col])
                term = inputs[row].mul(weight.real)
                if weight.imag != 0.0:
                    term = term.add(inputs[row].mul(weight.imag).mul_by_i())
                acc = acc.add(term)
            packed_outputs.append(acc)
        return packed_outputs

    def unpack_heads(
        self,
        packed: list[EncryptedTensor],
    ) -> tuple[list[EncryptedTensor], list[EncryptedTensor]]:
        head_h = [col.extract_real() for col in packed]
        head_h1 = [col.extract_imag() for col in packed]
        return head_h, head_h1

    def repack_after_attention(
        self,
        att_h: list[EncryptedTensor],
        att_h1: list[EncryptedTensor],
    ) -> list[EncryptedTensor]:
        if len(att_h) != len(att_h1):
            raise ValueError(
                "att_h and att_h1 must have same length, "
                f"got {len(att_h)} and {len(att_h1)}"
            )
        repacked: list[EncryptedTensor] = []
        for col_h, col_h1 in zip(att_h, att_h1):
            repacked.append(col_h.add(col_h1.mul_by_i()))
        return repacked

    def extract_final(self, accumulated: EncryptedTensor) -> EncryptedTensor:
        return accumulated.extract_real()
