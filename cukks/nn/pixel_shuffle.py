from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _infer_hw(layout: dict) -> tuple[int, int]:
    if "height" in layout and "width" in layout:
        return int(layout["height"]), int(layout["width"])

    batch_size = int(layout.get("batch_size", 1))
    num_patches = int(layout["num_patches"])
    num_patches_per_image = int(layout.get("num_patches_per_image", num_patches // batch_size))
    side = int(math.isqrt(num_patches_per_image))
    if side * side != num_patches_per_image:
        raise ValueError(f"Cannot infer spatial dimensions from num_patches_per_image={num_patches_per_image}")
    return side, side


def _flatten_without_layout(x: "EncryptedTensor", total_in: int) -> "EncryptedTensor":
    from ..tensor import EncryptedTensor as _ET

    flat = _ET(x._cipher, (total_in,), x._context, x._depth)
    flat._needs_rescale = x._needs_rescale
    return flat


def _repeat_per_image_weight(
    per_image_weight: torch.Tensor,
    *,
    batch_size: int,
    total_in: int,
    total_out: int,
) -> torch.Tensor:
    if batch_size == 1:
        return per_image_weight

    weight = torch.zeros(total_out, total_in, dtype=torch.float64)
    image_out = per_image_weight.shape[0]
    image_in = per_image_weight.shape[1]
    for batch_idx in range(batch_size):
        row_start = batch_idx * image_out
        row_end = row_start + image_out
        col_start = batch_idx * image_in
        col_end = col_start + image_in
        weight[row_start:row_end, col_start:col_end] = per_image_weight
    return weight


class EncryptedPixelShuffle(EncryptedModule):
    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        if upscale_factor <= 0:
            raise ValueError("upscale_factor must be positive")
        self.upscale_factor = int(upscale_factor)

    def _build_weight(self, in_h: int, in_w: int, in_channels: int) -> tuple[torch.Tensor, int, int, int]:
        r = self.upscale_factor
        factor_sq = r * r
        if in_channels % factor_sq != 0:
            raise ValueError(
                f"PixelShuffle requires channels divisible by upscale_factor^2={factor_sq}, got {in_channels}"
            )

        out_channels = in_channels // factor_sq
        out_h = in_h * r
        out_w = in_w * r
        total_in = in_h * in_w * in_channels
        total_out = out_h * out_w * out_channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)

        for iy in range(in_h):
            for ix in range(in_w):
                in_patch = iy * in_w + ix
                for out_channel in range(out_channels):
                    for dy in range(r):
                        for dx in range(r):
                            in_channel = out_channel * factor_sq + dy * r + dx
                            oy = iy * r + dy
                            ox = ix * r + dx
                            out_patch = oy * out_w + ox
                            row = out_patch * out_channels + out_channel
                            col = in_patch * in_channels + in_channel
                            weight[row, col] = 1.0

        return weight, out_channels, out_h, out_w

    def _updated_layout(self, x: "EncryptedTensor", out_channels: int, out_h: int, out_w: int) -> dict:
        layout = copy.deepcopy(x._cnn_layout)
        assert layout is not None
        batch_size = int(layout.get("batch_size", 1))
        out_patches_per_image = out_h * out_w
        layout["patch_features"] = out_channels
        layout["height"] = out_h
        layout["width"] = out_w
        layout["num_patches_per_image"] = out_patches_per_image
        layout["num_patches"] = out_patches_per_image * batch_size

        original_shape = layout.get("original_shape")
        if isinstance(original_shape, tuple) and len(original_shape) >= 3:
            original_shape = list(original_shape)
            original_shape[-3] = out_channels
            original_shape[-2] = out_h
            original_shape[-1] = out_w
            layout["original_shape"] = tuple(original_shape)

        return layout

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        layout = getattr(x, "_cnn_layout", None)
        if layout is None:
            raise RuntimeError("EncryptedPixelShuffle requires _cnn_layout metadata for CNN-packed input")

        batch_size = int(layout.get("batch_size", 1))
        in_h, in_w = _infer_hw(layout)
        in_channels = int(layout["patch_features"])
        per_image_weight, out_channels, out_h, out_w = self._build_weight(in_h, in_w, in_channels)

        image_in = in_h * in_w * in_channels
        image_out = out_h * out_w * out_channels
        total_in = batch_size * image_in
        total_out = batch_size * image_out
        weight = _repeat_per_image_weight(
            per_image_weight,
            batch_size=batch_size,
            total_in=total_in,
            total_out=total_out,
        )

        result = _flatten_without_layout(x, total_in).matmul(weight)
        result._shape = (batch_size * out_h * out_w, out_channels)
        result._cnn_layout = self._updated_layout(x, out_channels, out_h, out_w)

        if batch_size > 1:
            result._packed_batch = True
            result._batch_size = batch_size
            result._slots_per_sample = image_out
            result._packed_sample_shape = (out_h * out_w, out_channels)

        return result

    @classmethod
    def from_torch(cls, module: nn.PixelShuffle) -> "EncryptedPixelShuffle":
        return cls(upscale_factor=module.upscale_factor)

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"


class EncryptedPixelUnshuffle(EncryptedModule):
    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        if downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive")
        self.downscale_factor = int(downscale_factor)

    def _build_weight(self, in_h: int, in_w: int, in_channels: int) -> tuple[torch.Tensor, int, int, int]:
        r = self.downscale_factor
        if in_h % r != 0 or in_w % r != 0:
            raise ValueError(
                f"PixelUnshuffle requires spatial dimensions divisible by downscale_factor={r}, got {(in_h, in_w)}"
            )

        out_channels = in_channels * r * r
        out_h = in_h // r
        out_w = in_w // r
        total_in = in_h * in_w * in_channels
        total_out = out_h * out_w * out_channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)

        for oy in range(out_h):
            for ox in range(out_w):
                out_patch = oy * out_w + ox
                for in_channel in range(in_channels):
                    for dy in range(r):
                        for dx in range(r):
                            iy = oy * r + dy
                            ix = ox * r + dx
                            in_patch = iy * in_w + ix
                            out_channel = in_channel * (r * r) + dy * r + dx
                            row = out_patch * out_channels + out_channel
                            col = in_patch * in_channels + in_channel
                            weight[row, col] = 1.0

        return weight, out_channels, out_h, out_w

    def _updated_layout(self, x: "EncryptedTensor", out_channels: int, out_h: int, out_w: int) -> dict:
        layout = copy.deepcopy(x._cnn_layout)
        assert layout is not None
        batch_size = int(layout.get("batch_size", 1))
        out_patches_per_image = out_h * out_w
        layout["patch_features"] = out_channels
        layout["height"] = out_h
        layout["width"] = out_w
        layout["num_patches_per_image"] = out_patches_per_image
        layout["num_patches"] = out_patches_per_image * batch_size

        original_shape = layout.get("original_shape")
        if isinstance(original_shape, tuple) and len(original_shape) >= 3:
            original_shape = list(original_shape)
            original_shape[-3] = out_channels
            original_shape[-2] = out_h
            original_shape[-1] = out_w
            layout["original_shape"] = tuple(original_shape)

        return layout

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        layout = getattr(x, "_cnn_layout", None)
        if layout is None:
            raise RuntimeError("EncryptedPixelUnshuffle requires _cnn_layout metadata for CNN-packed input")

        batch_size = int(layout.get("batch_size", 1))
        in_h, in_w = _infer_hw(layout)
        in_channels = int(layout["patch_features"])
        per_image_weight, out_channels, out_h, out_w = self._build_weight(in_h, in_w, in_channels)

        image_in = in_h * in_w * in_channels
        image_out = out_h * out_w * out_channels
        total_in = batch_size * image_in
        total_out = batch_size * image_out
        weight = _repeat_per_image_weight(
            per_image_weight,
            batch_size=batch_size,
            total_in=total_in,
            total_out=total_out,
        )

        result = _flatten_without_layout(x, total_in).matmul(weight)
        result._shape = (batch_size * out_h * out_w, out_channels)
        result._cnn_layout = self._updated_layout(x, out_channels, out_h, out_w)

        if batch_size > 1:
            result._packed_batch = True
            result._batch_size = batch_size
            result._slots_per_sample = image_out
            result._packed_sample_shape = (out_h * out_w, out_channels)

        return result

    @classmethod
    def from_torch(cls, module: nn.PixelUnshuffle) -> "EncryptedPixelUnshuffle":
        return cls(downscale_factor=module.downscale_factor)

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"
