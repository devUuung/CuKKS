from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PackingLayout:
    seq_len: int
    d_model: int
    num_heads: int
    block_size: int
    complex_packing: str = "real_only"

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.block_size < self.d_model:
            raise ValueError(
                f"block_size ({self.block_size}) must be >= d_model ({self.d_model})"
            )
        if self.complex_packing not in {"real_only", "rihp", "dhp"}:
            raise ValueError(
                "complex_packing must be 'real_only', 'rihp', or 'dhp', "
                f"got {self.complex_packing!r}"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    @property
    def total_slots_needed(self) -> int:
        return self.seq_len * self.block_size

    def token_offset(self, offset: int) -> int:
        return offset * self.block_size

    def head_range(self, head_idx: int) -> tuple[int, int]:
        if head_idx < 0 or head_idx >= self.num_heads:
            raise ValueError(f"head_idx must be in [0, {self.num_heads}), got {head_idx}")
        start = head_idx * self.d_head
        return (start, start + self.d_head)


@dataclass
class TokenBlockPacker:
    layout: PackingLayout
    total_slots: int
    _cached_masks: list[list[float]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total_slots < self.layout.total_slots_needed:
            raise ValueError(
                f"total_slots ({self.total_slots}) must be >= "
                f"layout.total_slots_needed ({self.layout.total_slots_needed})"
            )

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout.complex_packing != "real_only":
            raise NotImplementedError(
                f"TokenBlockPacker does not yet implement {self.layout.complex_packing} packing"
            )
        tensor = x.detach().to(dtype=torch.float64, device="cpu")
        expected_shape = (self.layout.seq_len, self.layout.d_model)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Expected tensor shape {expected_shape}, got {tuple(tensor.shape)}")

        packed = torch.zeros(self.total_slots, dtype=torch.float64)
        for token_idx in range(self.layout.seq_len):
            start = token_idx * self.layout.block_size
            end = start + self.layout.d_model
            packed[start:end] = tensor[token_idx]
        return packed

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        flat = packed.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if flat.numel() < self.total_slots:
            raise ValueError(
                f"Packed tensor has {flat.numel()} values, expected at least {self.total_slots}"
            )
        unpacked = torch.zeros((self.layout.seq_len, self.layout.d_model), dtype=torch.float64)
        for token_idx in range(self.layout.seq_len):
            start = token_idx * self.layout.block_size
            end = start + self.layout.d_model
            unpacked[token_idx] = flat[start:end]
        return unpacked

    def head_mask(self, head_idx: int) -> list[float]:
        head_start, head_end = self.layout.head_range(head_idx)
        mask = [0.0] * self.total_slots
        for token_idx in range(self.layout.seq_len):
            base = token_idx * self.layout.block_size
            for slot_idx in range(base + head_start, base + head_end):
                mask[slot_idx] = 1.0
        return mask

    def all_head_masks(self) -> list[list[float]]:
        if self._cached_masks is None:
            self._cached_masks = [self.head_mask(head_idx) for head_idx in range(self.layout.num_heads)]
        return self._cached_masks
