from __future__ import annotations

from cukks.tensor import EncryptedTensor


class RIHPacker:
    def __init__(self, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len % 2 != 0:
            raise ValueError(f"seq_len must be even for RIHP, got {seq_len}")
        self.seq_len = seq_len
        self.half_seq_len = seq_len // 2

    def pack_hybrid(self, keys: list[EncryptedTensor]) -> list[EncryptedTensor]:
        packed: list[EncryptedTensor] = []
        for key_col in keys:
            rotated = key_col.rotate(self.half_seq_len)
            packed.append(key_col.add(rotated.mul_by_i()))
        return packed

    def halved_ccmm(
        self,
        queries: list[EncryptedTensor],
        keys_hybrid: list[EncryptedTensor],
    ) -> list[EncryptedTensor]:
        if len(queries) != len(keys_hybrid):
            raise ValueError(
                "queries and keys_hybrid must have same length, "
                f"got {len(queries)} and {len(keys_hybrid)}"
            )
        if not queries:
            return []

        packed_diagonals: list[EncryptedTensor] = []
        for rotation in range(self.half_seq_len):
            acc = queries[0].mul(keys_hybrid[0].rotate(rotation)).rescale()
            for col in range(1, len(queries)):
                term = queries[col].mul(keys_hybrid[col].rotate(rotation)).rescale()
                acc = acc.add(term)
            packed_diagonals.append(acc)
        return packed_diagonals

    def unpack_diagonals(self, hybrid_results: list[EncryptedTensor]) -> list[EncryptedTensor]:
        if len(hybrid_results) != self.half_seq_len:
            raise ValueError(
                "hybrid_results must have length seq_len/2, "
                f"got {len(hybrid_results)} for seq_len={self.seq_len}"
            )

        real_diagonals: list[EncryptedTensor] = []
        imag_diagonals: list[EncryptedTensor] = []
        for hybrid in hybrid_results:
            real_diag = hybrid.add(hybrid.conjugate()).mul(0.5)
            imag_diag = hybrid.extract_imag()
            real_diagonals.append(real_diag)
            imag_diagonals.append(imag_diag)
        return real_diagonals + imag_diagonals
