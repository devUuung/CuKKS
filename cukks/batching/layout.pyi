from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class PackingLayout:
    seq_len: int
    d_model: int
    num_heads: int
    block_size: int
    complex_packing: str = "real_only"

    @property
    def d_head(self) -> int: ...

    @property
    def total_slots_needed(self) -> int: ...

    def token_offset(self, offset: int) -> int: ...
    def head_range(self, head_idx: int) -> Tuple[int, int]: ...


@dataclass
class TokenBlockPacker:
    layout: PackingLayout
    total_slots: int

    def pack(self, x: torch.Tensor) -> torch.Tensor: ...
    def unpack(self, packed: torch.Tensor) -> torch.Tensor: ...
    def head_mask(self, head_idx: int) -> List[float]: ...
    def all_head_masks(self) -> List[List[float]]: ...
