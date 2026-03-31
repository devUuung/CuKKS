"""Encrypted transposed convolution layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedConvTranspose2d(EncryptedModule):
    """Encrypted 2D transposed convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_pair(kernel_size)
        self.stride = self._to_pair(stride)
        self.padding = self._to_pair(padding)
        self.output_padding = self._to_pair(output_padding)
        self.dilation = self._to_pair(dilation)
        self.groups = groups

        self.weight = weight.detach().to(dtype=torch.float64, device="cpu")
        self.bias = bias.detach().to(dtype=torch.float64, device="cpu") if bias is not None else None
        self.weight_matrix = self._build_transposed_im2col_weight_matrix()

        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    @staticmethod
    def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(value, int):
            return (value, value)
        return value

    def _build_transposed_im2col_weight_matrix(self) -> torch.Tensor:
        kH, kW = self.kernel_size
        out_per_group = self.out_channels // self.groups
        rows = self.out_channels * kH * kW
        matrix = torch.zeros(rows, self.in_channels, dtype=torch.float64)
        in_per_group = self.in_channels // self.groups

        for ic in range(self.in_channels):
            group_idx = ic // in_per_group
            out_base = group_idx * out_per_group
            for oc_local in range(out_per_group):
                oc = out_base + oc_local
                for ky in range(kH):
                    for kx in range(kW):
                        row = oc * (kH * kW) + ky * kW + kx
                        matrix[row, ic] = self.weight[ic, oc_local, ky, kx]

        return matrix

    def _infer_spatial_dims(self, num_patches_per_image: int, layout: dict) -> Tuple[int, int]:
        if "height" in layout and "width" in layout:
            return int(layout["height"]), int(layout["width"])

        side = int(math.isqrt(num_patches_per_image))
        if side * side != num_patches_per_image:
            raise ValueError(
                "Cannot infer ConvTranspose2d input height/width from "
                f"num_patches_per_image={num_patches_per_image}. "
                "Set layout['height'] and layout['width'] explicitly."
            )
        return side, side

    def _build_full_weight_matrix(self, input_height: int, input_width: int) -> torch.Tensor:
        out_h, out_w = self.get_output_size(input_height, input_width)
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        num_patches_in = input_height * input_width
        num_patches_out = out_h * out_w
        matrix = torch.zeros(
            num_patches_out * self.out_channels,
            num_patches_in * self.in_channels,
            dtype=torch.float64,
        )

        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups

        for in_y in range(input_height):
            for in_x in range(input_width):
                in_patch = in_y * input_width + in_x
                for ic in range(self.in_channels):
                    group_idx = ic // in_per_group
                    out_base = group_idx * out_per_group
                    col = in_patch * self.in_channels + ic
                    for oc_local in range(out_per_group):
                        oc = out_base + oc_local
                        for ky in range(kH):
                            out_y = in_y * sH - pH + ky * dH
                            if out_y < 0 or out_y >= out_h:
                                continue
                            for kx in range(kW):
                                out_x = in_x * sW - pW + kx * dW
                                if out_x < 0 or out_x >= out_w:
                                    continue
                                row = (out_y * out_w + out_x) * self.out_channels + oc
                                matrix[row, col] += self.weight[ic, oc_local, ky, kx]

        return matrix

    def _apply_cnn_matmul(
        self,
        x: "EncryptedTensor",
        *,
        batch_size: int,
        num_patches_per_image: int,
        input_height: int,
        input_width: int,
        layout: dict,
    ) -> "EncryptedTensor":
        from ..tensor import EncryptedTensor as _ET

        if layout["patch_features"] != self.in_channels:
            raise RuntimeError(
                "EncryptedConvTranspose2d requires CNN layout with "
                f"patch_features={self.in_channels}, got {layout['patch_features']}."
            )

        per_image_weight = self._build_full_weight_matrix(input_height, input_width)
        out_h, out_w = self.get_output_size(input_height, input_width)
        out_patches_per_image = out_h * out_w

        total_in = batch_size * num_patches_per_image * self.in_channels
        total_out = batch_size * out_patches_per_image * self.out_channels

        if batch_size == 1:
            weight = per_image_weight
            bias = None if self.bias is None else self.bias.repeat(out_patches_per_image)
        else:
            weight = torch.zeros(total_out, total_in, dtype=torch.float64)
            for batch_idx in range(batch_size):
                row_start = batch_idx * out_patches_per_image * self.out_channels
                row_end = (batch_idx + 1) * out_patches_per_image * self.out_channels
                col_start = batch_idx * num_patches_per_image * self.in_channels
                col_end = (batch_idx + 1) * num_patches_per_image * self.in_channels
                weight[row_start:row_end, col_start:col_end] = per_image_weight
            bias = None if self.bias is None else self.bias.repeat(out_patches_per_image * batch_size)

        x_flat = _ET(x._cipher, (total_in,), x._context, x._depth)
        x_flat._needs_rescale = x._needs_rescale
        result = x_flat.matmul(weight, bias)

        result._cnn_layout = {
            "num_patches": out_patches_per_image * batch_size,
            "num_patches_per_image": out_patches_per_image,
            "patch_features": self.out_channels,
            "original_shape": layout.get("original_shape"),
            "batch_size": batch_size,
            "height": out_h,
            "width": out_w,
        }
        result._shape = (out_patches_per_image * batch_size, self.out_channels)

        if batch_size > 1:
            result._packed_batch = True
            result._batch_size = batch_size
            result._slots_per_sample = out_patches_per_image * self.out_channels
            result._packed_sample_shape = (out_patches_per_image, self.out_channels)

        return result

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        if getattr(x, "_cnn_layout", None) is not None:
            layout = x._cnn_layout
            assert layout is not None
            batch_size = int(layout.get("batch_size", 1))
            num_patches = int(layout["num_patches"])
            num_patches_per_image = int(layout.get("num_patches_per_image", num_patches // batch_size))
            input_height, input_width = self._infer_spatial_dims(num_patches_per_image, layout)
            return self._apply_cnn_matmul(
                x,
                batch_size=batch_size,
                num_patches_per_image=num_patches_per_image,
                input_height=input_height,
                input_width=input_width,
                layout=layout,
            )

        if getattr(x, "_packed_batch", False):
            sample_shape = getattr(x, "_packed_sample_shape", None)
            batch_size = getattr(x, "_batch_size", None)
            if sample_shape is None or batch_size is None or len(sample_shape) != 2 or sample_shape[1] != self.in_channels:
                raise RuntimeError(
                    "EncryptedConvTranspose2d packed-batch input must have sample shape "
                    f"(num_patches, {self.in_channels})."
                )
            num_patches_per_image = int(sample_shape[0])
            input_height, input_width = self._infer_spatial_dims(num_patches_per_image, {})
            return self._apply_cnn_matmul(
                x,
                batch_size=int(batch_size),
                num_patches_per_image=num_patches_per_image,
                input_height=input_height,
                input_width=input_width,
                layout={
                    "patch_features": self.in_channels,
                    "batch_size": int(batch_size),
                    "num_patches_per_image": num_patches_per_image,
                    "num_patches": int(batch_size) * num_patches_per_image,
                },
            )

        raise RuntimeError(
            "EncryptedConvTranspose2d requires CNN layout metadata or packed CNN batch input."
        )

    def get_output_size(self, input_height: int, input_width: int) -> Tuple[int, int]:
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        oH, oW = self.output_padding
        dH, dW = self.dilation
        out_h = (input_height - 1) * sH - 2 * pH + dH * (kH - 1) + oH + 1
        out_w = (input_width - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1
        return out_h, out_w

    @classmethod
    def from_torch(cls, module: nn.ConvTranspose2d) -> "EncryptedConvTranspose2d":
        padding = module.padding
        if isinstance(padding, str):
            raise TypeError("String padding modes are not supported for EncryptedConvTranspose2d")

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=cast(Tuple[int, int], tuple(module.kernel_size)),
            weight=module.weight,
            bias=module.bias,
            stride=cast(Tuple[int, int], tuple(module.stride)),
            padding=cast(Tuple[int, int], tuple(padding)),
            output_padding=cast(Tuple[int, int], tuple(module.output_padding)),
            groups=module.groups,
            dilation=cast(Tuple[int, int], tuple(module.dilation)),
        )

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        args = [
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
            f"padding={self.padding}",
            f"output_padding={self.output_padding}",
        ]
        if self.groups != 1:
            args.append(f"groups={self.groups}")
        if self.dilation != (1, 1):
            args.append(f"dilation={self.dilation}")
        if self.bias is None:
            args.append("bias=False")
        return ", ".join(args)
