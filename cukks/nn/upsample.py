"""EncryptedUpsample - Spatial upsampling via plaintext sparse matmul."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _to_2tuple(value: int | float | tuple[int | float, int | float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected length-2 tuple, got {value}")
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _normalize_size(size: object) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"size must be an int or length-2 tuple, got {size}")


def _normalize_scale_factor(scale_factor: object) -> float | tuple[float, float] | None:
    if scale_factor is None:
        return None
    if isinstance(scale_factor, (int, float)):
        return float(scale_factor)
    if isinstance(scale_factor, tuple) and len(scale_factor) == 2:
        return float(scale_factor[0]), float(scale_factor[1])
    raise ValueError(f"scale_factor must be a number or length-2 tuple, got {scale_factor}")


class EncryptedUpsample(EncryptedModule):
    """Encrypted 2D upsampling for CNN-packed tensors."""

    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None:
        super().__init__()

        if size is None and scale_factor is None:
            raise ValueError("EncryptedUpsample requires size or scale_factor")
        if mode not in {"nearest", "bilinear"}:
            raise ValueError(f"unsupported upsample mode: {mode}")

        self.size = _normalize_size(size)
        self.scale_factor = _normalize_scale_factor(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def _infer_input_hw(self, layout: dict) -> tuple[int, int]:
        if "height" in layout and "width" in layout:
            return int(layout["height"]), int(layout["width"])

        batch_size = int(layout.get("batch_size", 1))
        num_patches = int(layout["num_patches"])
        num_patches_per_image = int(layout.get("num_patches_per_image", num_patches // batch_size))
        side = int(math.isqrt(num_patches_per_image))
        if side * side != num_patches_per_image:
            raise ValueError(
                f"Cannot infer spatial dimensions from num_patches_per_image={num_patches_per_image}"
            )
        return side, side

    def _output_hw(self, in_h: int, in_w: int) -> tuple[int, int]:
        if self.size is not None:
            return int(self.size[0]), int(self.size[1])

        if self.scale_factor is None:
            raise RuntimeError("scale_factor must be set when size is None")
        scale_h, scale_w = _to_2tuple(self.scale_factor)
        out_h = int(math.floor(in_h * scale_h))
        out_w = int(math.floor(in_w * scale_w))
        return out_h, out_w

    def _source_index_nearest(self, out_idx: int, in_size: int, out_size: int) -> int:
        src = int(math.floor(out_idx * in_size / out_size))
        return min(max(src, 0), in_size - 1)

    def _source_index_linear(self, out_idx: int, in_size: int, out_size: int) -> float:
        if out_size <= 1:
            return 0.0
        if self.align_corners:
            src = out_idx * (in_size - 1) / (out_size - 1)
        else:
            src = ((out_idx + 0.5) * in_size / out_size) - 0.5
        return min(max(src, 0.0), float(in_size - 1))

    def _build_nearest_weight(
        self,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
        channels: int,
    ) -> torch.Tensor:
        total_in = in_h * in_w * channels
        total_out = out_h * out_w * channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)

        for oy in range(out_h):
            iy = self._source_index_nearest(oy, in_h, out_h)
            for ox in range(out_w):
                ix = self._source_index_nearest(ox, in_w, out_w)
                out_patch = oy * out_w + ox
                in_patch = iy * in_w + ix
                for channel in range(channels):
                    weight[out_patch * channels + channel, in_patch * channels + channel] = 1.0

        return weight

    def _build_bilinear_weight(
        self,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
        channels: int,
    ) -> torch.Tensor:
        total_in = in_h * in_w * channels
        total_out = out_h * out_w * channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)

        for oy in range(out_h):
            src_y = self._source_index_linear(oy, in_h, out_h)
            y0 = int(math.floor(src_y))
            y1 = min(y0 + 1, in_h - 1)
            wy1 = src_y - y0
            wy0 = 1.0 - wy1

            for ox in range(out_w):
                src_x = self._source_index_linear(ox, in_w, out_w)
                x0 = int(math.floor(src_x))
                x1 = min(x0 + 1, in_w - 1)
                wx1 = src_x - x0
                wx0 = 1.0 - wx1

                coeffs = (
                    (y0, x0, wy0 * wx0),
                    (y0, x1, wy0 * wx1),
                    (y1, x0, wy1 * wx0),
                    (y1, x1, wy1 * wx1),
                )
                out_patch = oy * out_w + ox
                for channel in range(channels):
                    row = out_patch * channels + channel
                    for iy, ix, coeff in coeffs:
                        if abs(coeff) <= 1e-30:
                            continue
                        col = (iy * in_w + ix) * channels + channel
                        weight[row, col] += coeff

        return weight

    def _build_weight(
        self,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
        channels: int,
    ) -> torch.Tensor:
        if self.mode == "nearest":
            return self._build_nearest_weight(in_h, in_w, out_h, out_w, channels)
        return self._build_bilinear_weight(in_h, in_w, out_h, out_w, channels)

    def _updated_layout(self, x: "EncryptedTensor", out_h: int, out_w: int) -> dict:
        if x._cnn_layout is None:
            raise RuntimeError("EncryptedUpsample requires _cnn_layout metadata")

        layout = copy.deepcopy(x._cnn_layout)
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
        if x._cnn_layout is None:
            raise RuntimeError("EncryptedUpsample requires _cnn_layout metadata for CNN-packed input")

        layout = x._cnn_layout
        channels = int(layout["patch_features"])
        batch_size = int(layout.get("batch_size", 1))
        in_h, in_w = self._infer_input_hw(layout)
        out_h, out_w = self._output_hw(in_h, in_w)
        weight = self._build_weight(in_h, in_w, out_h, out_w, channels)
        out_patches = out_h * out_w

        if getattr(x, "_packed_batch", False) and batch_size > 1:
            total_in = in_h * in_w * channels
            x_flat = x.view(batch_size, total_in)
            result = x_flat.matmul(weight).view(batch_size, out_patches, channels)
        else:
            result = x.matmul(weight)
            result._shape = (out_patches, channels)

        result._cnn_layout = self._updated_layout(x, out_h, out_w)
        return result

    @classmethod
    def from_torch(cls, module: torch.nn.Upsample) -> "EncryptedUpsample":
        return cls(
            size=_normalize_size(module.size),
            scale_factor=_normalize_scale_factor(module.scale_factor),
            mode=module.mode,
            align_corners=module.align_corners,
        )

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        parts = []
        if self.size is not None:
            parts.append(f"size={self.size}")
        if self.scale_factor is not None:
            parts.append(f"scale_factor={self.scale_factor}")
        parts.append(f"mode='{self.mode}'")
        if self.align_corners is not None:
            parts.append(f"align_corners={self.align_corners}")
        return ", ".join(parts)
