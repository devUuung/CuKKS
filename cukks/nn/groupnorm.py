"""Encrypted GroupNorm layer."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .module import EncryptedModule
from ..stats.crypto_inv_sqrt import _compute_inv_sqrt_coeffs

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedGroupNorm(EncryptedModule):
    """Encrypted GroupNorm using polynomial inverse-square-root approximation."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        inv_sqrt_domain: Tuple[float, float] = (0.01, 10.0),
    ) -> None:
        super().__init__()

        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels={num_channels} must be divisible by num_groups={num_groups}"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.eps = eps
        self.inv_sqrt_domain = inv_sqrt_domain
        self.inv_sqrt_coeffs = _compute_inv_sqrt_coeffs(inv_sqrt_domain, degree=15)

        self.weight = self._prepare_param(weight, default=1.0)
        self.bias = self._prepare_param(bias, default=0.0)
        self._layout_cache: Dict[Tuple[Tuple[int, ...], int], Tuple[torch.Tensor, list[float], list[float]]] = {}

        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.register_parameter("inv_sqrt_coeffs", self.inv_sqrt_coeffs)

    def _prepare_param(self, value: Optional[torch.Tensor], *, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((self.num_channels,), default, dtype=torch.float64, device="cpu")

        prepared = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if prepared.shape != (self.num_channels,):
            raise ValueError(
                f"expected parameter shape ({self.num_channels},), got {tuple(prepared.shape)}"
            )
        return prepared

    def _infer_sample_layout(self, x: "EncryptedTensor") -> Tuple[Tuple[int, ...], int]:
        layout = getattr(x, "_cnn_layout", None)
        if layout is not None:
            patch_features = int(layout["patch_features"])
            if patch_features != self.num_channels:
                raise RuntimeError(
                    "EncryptedGroupNorm CNN path requires patch_features to equal num_channels, "
                    f"got patch_features={patch_features} and num_channels={self.num_channels}."
                )
            num_patches = int(
                layout.get("num_patches_per_image", layout["num_patches"])
                if getattr(x, "_packed_batch", False)
                else layout["num_patches"]
            )
            return (num_patches, self.num_channels), 1

        sample_shape = tuple(getattr(x, "_packed_sample_shape", None) or x.shape)
        if not sample_shape:
            raise RuntimeError("EncryptedGroupNorm requires a non-empty input shape")

        if sample_shape[0] == self.num_channels:
            return sample_shape, 0
        if sample_shape[-1] == self.num_channels:
            return sample_shape, len(sample_shape) - 1
        if len(sample_shape) == 1 and sample_shape[0] == self.num_channels:
            return sample_shape, 0

        raise RuntimeError(
            "EncryptedGroupNorm requires the channel dimension to be the first or last axis, "
            f"got sample_shape={sample_shape} and num_channels={self.num_channels}."
        )

    def _layout_tensors(
        self,
        sample_shape: Tuple[int, ...],
        channel_axis: int,
    ) -> Tuple[torch.Tensor, list[float], list[float]]:
        key = (sample_shape, channel_axis)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        sample_size = math.prod(sample_shape)
        spatial_shape = list(sample_shape)
        spatial_shape.pop(channel_axis)
        group_size = self.channels_per_group * int(math.prod(spatial_shape))

        channel_view_shape = [1] * len(sample_shape)
        channel_view_shape[channel_axis] = self.num_channels
        channel_ids = torch.arange(self.num_channels, dtype=torch.int64).view(*channel_view_shape).expand(*sample_shape)
        group_ids = (channel_ids // self.channels_per_group).reshape(-1)

        avg_block = (group_ids[:, None] == group_ids[None, :]).to(torch.float64)
        avg_block /= float(group_size)

        weight_view_shape = [1] * len(sample_shape)
        weight_view_shape[channel_axis] = self.num_channels
        weight_pattern = self.weight.view(*weight_view_shape).expand(*sample_shape).reshape(sample_size).tolist()
        bias_pattern = self.bias.view(*weight_view_shape).expand(*sample_shape).reshape(sample_size).tolist()

        cached = (avg_block, weight_pattern, bias_pattern)
        self._layout_cache[key] = cached
        return cached

    def _restore_metadata(self, result: "EncryptedTensor", x: "EncryptedTensor") -> None:
        result._cnn_layout = copy.deepcopy(getattr(x, "_cnn_layout", None))
        result._packed_batch = getattr(x, "_packed_batch", False)
        result._batch_size = getattr(x, "_batch_size", None)
        result._slots_per_sample = getattr(x, "_slots_per_sample", None)
        result._packed_sample_shape = getattr(x, "_packed_sample_shape", None)
        result._sigma_factor = None

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        sample_shape, channel_axis = self._infer_sample_layout(x)
        sample_size = math.prod(sample_shape)
        avg_block, weight_pattern, bias_pattern = self._layout_tensors(sample_shape, channel_axis)

        a, b = self.inv_sqrt_domain
        alpha = 2.0 / (b - a)
        beta_map = -(a + b) / (b - a)

        if getattr(x, "_packed_batch", False):
            batch_size = getattr(x, "_batch_size", None)
            if batch_size is None:
                raise RuntimeError("EncryptedGroupNorm packed path requires batch_size metadata")
            reshaped = x.view(batch_size, sample_size)
        else:
            reshaped = x.view(1, sample_size)

        mean_broadcast = reshaped.matmul(avg_block)
        centered = reshaped.sub(mean_broadcast)

        sq = centered.square().rescale()
        var_broadcast = sq.matmul(avg_block)
        var_eps = var_broadcast.add(self.eps)

        t = var_eps.mul(alpha).rescale().add(beta_map)
        inv_std = t.poly_eval(self.inv_sqrt_coeffs)
        normalized = centered.mul(inv_std).rescale()

        output = normalized.mul(weight_pattern).rescale()
        output = output.add(bias_pattern)
        result = output.view(*x.shape)
        self._restore_metadata(result, x)
        return result

    @classmethod
    def from_torch(cls, module: nn.GroupNorm) -> "EncryptedGroupNorm":
        return cls(
            num_groups=module.num_groups,
            num_channels=module.num_channels,
            weight=module.weight.data,
            bias=module.bias.data,
            eps=module.eps,
        )

    def mult_depth(self) -> int:
        return 18

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
            f"eps={self.eps}"
        )
