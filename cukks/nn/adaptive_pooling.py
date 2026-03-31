"""Encrypted adaptive pooling layers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple, Union, cast

import torch
import torch.nn as nn

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def _expand_patch_indices_with_channels(
    patch_indices: torch.Tensor,
    channels: int,
    *,
    patch_offset: int = 0,
) -> torch.Tensor:
    channel_offsets = torch.arange(channels, dtype=torch.long)
    return (patch_indices[..., None] + patch_offset) * channels + channel_offsets


class EncryptedAdaptiveAvgPool2d(EncryptedModule):
    """Encrypted 2D adaptive average pooling using HE-friendly reductions."""

    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        self.output_size = output_size

    @staticmethod
    def _infer_spatial_dims(num_patches: int) -> Tuple[int, int]:
        hw = int(math.sqrt(num_patches))
        if hw * hw != num_patches:
            raise ValueError(
                f"Cannot infer H, W from num_patches={num_patches} (not a perfect square). "
                "Set layout['height'] and layout['width'] explicitly for rectangular inputs."
            )
        return hw, hw

    @staticmethod
    def _compute_kernel_stride(input_size: int, output_size: int) -> Tuple[int, int]:
        kernel = int(math.ceil(input_size / output_size))
        stride = int(math.floor(input_size / output_size))
        return kernel, max(1, stride)

    @staticmethod
    def _output_indices_for_channels(
        num_out_patches: int,
        channels: int,
        *,
        patch_offset: int = 0,
    ) -> torch.Tensor:
        out_patch_indices = torch.arange(num_out_patches, dtype=torch.long) + patch_offset
        channel_offsets = torch.arange(channels, dtype=torch.long)
        return out_patch_indices[:, None] * channels + channel_offsets

    def _resolve_layout(self, x: "EncryptedTensor") -> Tuple[dict, int, int, int, int]:
        layout = getattr(x, "_cnn_layout", None)
        if layout is None:
            raise RuntimeError("EncryptedAdaptiveAvgPool2d requires _cnn_layout metadata for CNN-packed input")

        batch_size = layout.get("batch_size", 1)
        num_patches = int(layout["num_patches"])
        num_patches_per_image = int(layout.get("num_patches_per_image", num_patches // batch_size))

        if "height" in layout and "width" in layout:
            height, width = int(layout["height"]), int(layout["width"])
        else:
            height, width = self._infer_spatial_dims(num_patches_per_image)

        return layout, batch_size, num_patches_per_image, height, width

    def _forward_global_avg_fast(self, x: "EncryptedTensor", num_patches: int) -> "EncryptedTensor":
        averaged = x.sum_and_broadcast(num_patches).mul(1.0 / num_patches)
        if averaged._needs_rescale:
            averaged = averaged.rescale()

        averaged._shape = (1, 1)
        averaged._cnn_layout = {
            "num_patches": 1,
            "patch_features": 1,
            "height": 1,
            "width": 1,
            "original_shape": getattr(x, "_cnn_layout", {}).get("original_shape"),
        }
        return averaged

    def _build_pool_weight(
        self,
        *,
        batch_size: int,
        num_patches_per_image: int,
        channels: int,
        height: int,
        width: int,
        out_h: int,
        out_w: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
    ) -> torch.Tensor:
        out_patches_per_image = out_h * out_w
        total_in = num_patches_per_image * batch_size * channels
        total_out = out_patches_per_image * batch_size * channels
        weight = torch.zeros(total_out, total_in, dtype=torch.float64)

        for batch_idx in range(batch_size):
            rows = []
            for oy in range(out_h):
                start_y = oy * stride_h
                end_y = min(start_y + kernel_h, height)
                for ox in range(out_w):
                    start_x = ox * stride_w
                    end_x = min(start_x + kernel_w, width)
                    patch_indices = [y * width + x for y in range(start_y, end_y) for x in range(start_x, end_x)]
                    rows.append(torch.tensor(patch_indices, dtype=torch.long))

            out_indices = self._output_indices_for_channels(
                out_patches_per_image,
                channels,
                patch_offset=batch_idx * out_patches_per_image,
            )
            for out_patch_idx, patch_indices in enumerate(rows):
                area = float(len(patch_indices))
                in_indices = _expand_patch_indices_with_channels(
                    patch_indices.unsqueeze(0),
                    channels,
                    patch_offset=batch_idx * num_patches_per_image,
                ).reshape(-1)
                out_row = out_indices[out_patch_idx]
                repeated_out = out_row.unsqueeze(0).expand(patch_indices.numel(), -1).reshape(-1)
                weight[repeated_out, in_indices] = 1.0 / area

        return weight

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        if len(x.shape) != 2 or getattr(x, "_cnn_layout", None) is None:
            raise RuntimeError(
                "EncryptedAdaptiveAvgPool2d expects a 2D encrypted tensor with _cnn_layout metadata"
            )

        layout, batch_size, num_patches_per_image, height, width = self._resolve_layout(x)
        channels = int(layout["patch_features"])
        out_h, out_w = self.output_size
        kernel_h, stride_h = self._compute_kernel_stride(height, out_h)
        kernel_w, stride_w = self._compute_kernel_stride(width, out_w)

        if out_h == 1 and out_w == 1 and batch_size == 1 and channels == 1:
            return self._forward_global_avg_fast(x, num_patches_per_image)

        weight = self._build_pool_weight(
            batch_size=batch_size,
            num_patches_per_image=num_patches_per_image,
            channels=channels,
            height=height,
            width=width,
            out_h=out_h,
            out_w=out_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )

        total_in = int(layout["num_patches"]) * channels
        from ..tensor import EncryptedTensor as _ET

        x_flat = _ET(x._cipher, (total_in,), x._context, x._depth)
        x_flat._needs_rescale = x._needs_rescale
        result = x_flat.matmul(weight, None)

        out_patches_per_image = out_h * out_w
        out_patches = out_patches_per_image * batch_size
        result._cnn_layout = {
            "num_patches": out_patches,
            "num_patches_per_image": out_patches_per_image,
            "patch_features": channels,
            "height": out_h,
            "width": out_w,
            "original_shape": layout.get("original_shape"),
            "batch_size": batch_size,
        }
        result._shape = (out_patches, channels)
        return result

    @classmethod
    def from_torch(cls, module: nn.AdaptiveAvgPool2d) -> "EncryptedAdaptiveAvgPool2d":
        output_size = module.output_size
        if isinstance(output_size, tuple):
            out_h, out_w = output_size
            if out_h is None or out_w is None:
                raise ValueError("AdaptiveAvgPool2d with None output dimensions is not supported")
            return cls(output_size=(int(out_h), int(out_w)))
        scalar_output_size = cast(int, output_size)
        return cls(output_size=int(scalar_output_size))

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"
