"""Encrypted GroupNorm layer."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import torch
import torch.nn as nn

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedGroupNorm(EncryptedModule):
    """Encrypted GroupNorm with fixed per-group statistics.

    Uses pre-computed per-group mean and inverse standard deviation,
    analogous to BatchNorm inference mode.  This avoids the expensive
    dynamic mean/variance computation and inverse-square-root polynomial
    evaluation that would otherwise be required in CKKS.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        group_mean: torch.Tensor,
        group_inv_std: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
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

        self.group_mean = self._prepare_stat_param(group_mean, "group_mean")
        self.group_inv_std = self._prepare_stat_param(group_inv_std, "group_inv_std")
        self.weight = self._prepare_param(weight, default=1.0)
        self.bias = self._prepare_param(bias, default=0.0)

        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def _prepare_stat_param(self, value: torch.Tensor, name: str) -> torch.Tensor:
        prepared = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if prepared.shape != (self.num_channels,):
            raise ValueError(
                f"{name} shape must be ({self.num_channels},), got {tuple(prepared.shape)}"
            )
        return prepared

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
    ) -> Tuple[list[float], list[float]]:
        key = (sample_shape, channel_axis)
        cached = getattr(self, "_layout_cache", None)
        if cached is not None and key in cached:
            return cached[key]

        sample_size = int(__import__("math").prod(sample_shape))
        weight_view_shape = [1] * len(sample_shape)
        weight_view_shape[channel_axis] = self.num_channels
        weight_pattern = self.weight.view(*weight_view_shape).expand(*sample_shape).reshape(sample_size).tolist()
        bias_pattern = self.bias.view(*weight_view_shape).expand(*sample_shape).reshape(sample_size).tolist()

        if not hasattr(self, "_layout_cache"):
            self._layout_cache: dict = {}
        self._layout_cache[key] = (weight_pattern, bias_pattern)
        return weight_pattern, bias_pattern

    def _restore_metadata(self, result: "EncryptedTensor", x: "EncryptedTensor") -> None:
        result_any = cast(Any, result)
        x_any = cast(Any, x)
        result_any._cnn_layout = copy.deepcopy(getattr(x_any, "_cnn_layout", None))
        result_any._packed_batch = getattr(x_any, "_packed_batch", False)
        result_any._batch_size = getattr(x_any, "_batch_size", None)
        result_any._slots_per_sample = getattr(x_any, "_slots_per_sample", None)
        result_any._packed_sample_shape = getattr(x_any, "_packed_sample_shape", None)
        result_any._sigma_factor = None

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        sample_shape, channel_axis = self._infer_sample_layout(x)
        sample_size = int(__import__("math").prod(sample_shape))
        weight_pattern, bias_pattern = self._layout_tensors(sample_shape, channel_axis)

        if getattr(x, "_packed_batch", False):
            batch_size = getattr(x, "_batch_size", None)
            if batch_size is None:
                raise RuntimeError("EncryptedGroupNorm packed path requires batch_size metadata")
            reshaped = x.view(batch_size, sample_size)
        else:
            reshaped = x.view(1, sample_size)

        mean_list = self.group_mean.tolist()
        inv_std_list = self.group_inv_std.tolist()

        centered = reshaped.sub(mean_list)
        normalized = centered.mul(inv_std_list)

        output = normalized.mul(weight_pattern).rescale()
        output = output.add(bias_pattern)
        result = output.view(*x.shape)
        self._restore_metadata(result, x)
        return result

    @classmethod
    def from_torch(
        cls,
        module: nn.GroupNorm,
        group_mean: Optional[torch.Tensor] = None,
        group_inv_std: Optional[torch.Tensor] = None,
        inv_sqrt_degree: Optional[int] = None,
    ) -> "EncryptedGroupNorm":
        del inv_sqrt_degree
        if group_mean is None:
            group_mean = torch.zeros(module.num_channels, dtype=torch.float64)
        if group_inv_std is None:
            group_inv_std = torch.ones(module.num_channels, dtype=torch.float64)
        return cls(
            num_groups=module.num_groups,
            num_channels=module.num_channels,
            group_mean=group_mean,
            group_inv_std=group_inv_std,
            weight=module.weight.data,
            bias=module.bias.data,
            eps=module.eps,
        )

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
            f"eps={self.eps}"
        )
