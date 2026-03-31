from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Tuple, Union

import torch
import torch.nn as nn

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


PaddingArg = Union[int, Tuple[int, int, int, int]]


def _normalize_padding(padding: PaddingArg) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        if padding < 0:
            raise ValueError("padding must be non-negative")
        return padding, padding, padding, padding
    if len(padding) != 4:
        raise ValueError(f"padding must be an int or 4-tuple, got {padding}")
    left, right, top, bottom = (int(v) for v in padding)
    if min(left, right, top, bottom) < 0:
        raise ValueError("padding must be non-negative")
    return left, right, top, bottom


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


class _EncryptedPad2dBase(EncryptedModule):
    mode = "zeros"

    def __init__(self, padding: PaddingArg) -> None:
        super().__init__()
        self.padding = _normalize_padding(padding)

    def _map_index(self, index: int, size: int) -> int | None:
        raise NotImplementedError

    def _validate_input(self, in_h: int, in_w: int) -> None:
        return None

    def _padded_value(self) -> float:
        return 0.0

    def _build_weight_and_bias(
        self,
        in_h: int,
        in_w: int,
        channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        self._validate_input(in_h, in_w)
        left, right, top, bottom = self.padding
        out_h = in_h + top + bottom
        out_w = in_w + left + right
        total_in = in_h * in_w * channels
        total_out = out_h * out_w * channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        bias = None
        pad_value = self._padded_value()
        if abs(pad_value) > 1e-30:
            bias = torch.zeros(total_out, dtype=torch.float64)

        for oy in range(out_h):
            for ox in range(out_w):
                iy = self._map_index(oy - top, in_h)
                ix = self._map_index(ox - left, in_w)
                out_patch = oy * out_w + ox
                if iy is None or ix is None:
                    if bias is not None:
                        start = out_patch * channels
                        bias[start:start + channels] = pad_value
                    continue
                in_patch = iy * in_w + ix
                for channel in range(channels):
                    row = out_patch * channels + channel
                    col = in_patch * channels + channel
                    weight[row, col] = 1.0

        return weight, bias, out_h, out_w

    def _updated_layout(self, x: "EncryptedTensor", out_h: int, out_w: int) -> dict:
        layout = copy.deepcopy(x._cnn_layout)
        assert layout is not None
        batch_size = int(layout.get("batch_size", 1))
        out_patches_per_image = out_h * out_w
        layout["height"] = out_h
        layout["width"] = out_w
        layout["num_patches_per_image"] = out_patches_per_image
        layout["num_patches"] = out_patches_per_image * batch_size

        original_shape = layout.get("original_shape")
        if isinstance(original_shape, tuple) and len(original_shape) >= 2:
            original_shape = list(original_shape)
            original_shape[-2] = out_h
            original_shape[-1] = out_w
            layout["original_shape"] = tuple(original_shape)

        return layout

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        layout = getattr(x, "_cnn_layout", None)
        if layout is None:
            raise RuntimeError(f"{type(self).__name__} requires _cnn_layout metadata for CNN-packed input")

        batch_size = int(layout.get("batch_size", 1))
        in_h, in_w = _infer_hw(layout)
        channels = int(layout["patch_features"])
        per_image_weight, per_image_bias, out_h, out_w = self._build_weight_and_bias(in_h, in_w, channels)

        image_in = in_h * in_w * channels
        image_out = out_h * out_w * channels
        total_in = batch_size * image_in
        total_out = batch_size * image_out
        weight = _repeat_per_image_weight(
            per_image_weight,
            batch_size=batch_size,
            total_in=total_in,
            total_out=total_out,
        )

        bias = None if per_image_bias is None else per_image_bias.repeat(batch_size)
        result = _flatten_without_layout(x, total_in).matmul(weight, bias)
        result._shape = (batch_size * out_h * out_w, channels)
        result._cnn_layout = self._updated_layout(x, out_h, out_w)

        if batch_size > 1:
            result._packed_batch = True
            result._batch_size = batch_size
            result._slots_per_sample = image_out
            result._packed_sample_shape = (out_h * out_w, channels)

        return result

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class EncryptedZeroPad2d(_EncryptedPad2dBase):
    def _map_index(self, index: int, size: int) -> int | None:
        if 0 <= index < size:
            return index
        return None

    @classmethod
    def from_torch(cls, module: nn.ZeroPad2d) -> "EncryptedZeroPad2d":
        return cls(padding=module.padding)


class EncryptedConstantPad2d(_EncryptedPad2dBase):
    def __init__(self, padding: PaddingArg, value: float) -> None:
        super().__init__(padding)
        self.value = float(value)

    def _map_index(self, index: int, size: int) -> int | None:
        if 0 <= index < size:
            return index
        return None

    def _padded_value(self) -> float:
        return self.value

    @classmethod
    def from_torch(cls, module: nn.ConstantPad2d) -> "EncryptedConstantPad2d":
        return cls(padding=module.padding, value=module.value)

    def extra_repr(self) -> str:
        return f"padding={self.padding}, value={self.value}"


class EncryptedReflectionPad2d(_EncryptedPad2dBase):
    def _validate_input(self, in_h: int, in_w: int) -> None:
        left, right, top, bottom = self.padding
        if in_h <= 1 or in_w <= 1:
            raise ValueError("ReflectionPad2d requires input height and width greater than 1")
        if top >= in_h or bottom >= in_h or left >= in_w or right >= in_w:
            raise ValueError("Padding size must be less than the corresponding input dimension for reflection")

    def _map_index(self, index: int, size: int) -> int | None:
        while index < 0 or index >= size:
            if index < 0:
                index = -index
            else:
                index = 2 * size - index - 2
        return index

    @classmethod
    def from_torch(cls, module: nn.ReflectionPad2d) -> "EncryptedReflectionPad2d":
        return cls(padding=module.padding)


class EncryptedReplicationPad2d(_EncryptedPad2dBase):
    def _map_index(self, index: int, size: int) -> int | None:
        return min(max(index, 0), size - 1)

    @classmethod
    def from_torch(cls, module: nn.ReplicationPad2d) -> "EncryptedReplicationPad2d":
        return cls(padding=module.padding)
