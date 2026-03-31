"""Encrypted InstanceNorm layers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from cukks.tensor import EncryptedTensor

EncryptedGroupNorm = importlib.import_module("cukks.nn.groupnorm").EncryptedGroupNorm


class _EncryptedInstanceNormBase(EncryptedGroupNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_features = num_features
        self.affine = affine
        self.track_running_stats = track_running_stats
        super().__init__(
            num_groups=num_features,
            num_channels=num_features,
            weight=weight if affine else None,
            bias=bias if affine else None,
            eps=eps,
        )

    def forward(self, x: "EncryptedTensor"):
        return super().forward(x)

    def mult_depth(self) -> int:
        return 18

    @classmethod
    def _from_torch_common(
        cls,
        module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d],
    ):
        affine = bool(getattr(module, "affine", False))
        return cls(
            num_features=int(module.num_features),
            eps=float(module.eps),
            affine=affine,
            track_running_stats=bool(getattr(module, "track_running_stats", False)),
            weight=module.weight.data if affine else None,
            bias=module.bias.data if affine else None,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


class EncryptedInstanceNorm1d(_EncryptedInstanceNormBase):
    @classmethod
    def from_torch(cls, module: nn.InstanceNorm1d) -> "EncryptedInstanceNorm1d":
        return cls._from_torch_common(module)


class EncryptedInstanceNorm2d(_EncryptedInstanceNormBase):
    @classmethod
    def from_torch(cls, module: nn.InstanceNorm2d) -> "EncryptedInstanceNorm2d":
        return cls._from_torch_common(module)
