from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


@dataclass
class CNNLayoutAnalyzer:
    input_shape: Optional[Tuple[int, ...]] = None
    conv_params: List[Dict[str, Any]] = field(default_factory=list)
    pool_params: List[Dict[str, Tuple[int, int]]] = field(default_factory=list)

    def detect(self, model: nn.Module) -> bool:
        found_conv = False
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                found_conv = True
            if isinstance(module, nn.Linear) and found_conv:
                return True
        return False

    def analyze(self, model: nn.Module) -> None:
        self.conv_params = []
        self.pool_params = []

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.conv_params.append(
                    {
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "out_channels": module.out_channels,
                        "in_channels": module.in_channels,
                    }
                )
            elif isinstance(module, nn.AvgPool2d):
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                stride = module.stride or kernel_size
                if isinstance(stride, int):
                    stride = (stride, stride)
                self.pool_params.append(
                    {
                        "kernel_size": kernel_size,
                        "stride": stride,
                    }
                )

    def compute_layout_from_linear(self, linear: nn.Linear) -> Optional[Dict[str, Any]]:
        if not self.conv_params:
            return None

        last_conv = self.conv_params[-1]
        channels = last_conv["out_channels"]
        in_features = linear.in_features
        if in_features % channels != 0:
            return None

        num_patches = in_features // channels
        layout: Dict[str, Any] = {
            "num_patches": num_patches,
            "patch_features": channels,
        }

        if self.pool_params:
            last_pool = self.pool_params[-1]
            kh, kw = last_pool["kernel_size"]
            sh, sw = last_pool["stride"]
            if kh == 2 and kw == 2 and sh == 2 and sw == 2:
                pre_pool_h, pre_pool_w = self.compute_pre_pool_dimensions()
                if pre_pool_h is not None and pre_pool_w is not None:
                    out_h = pre_pool_h // 2
                    out_w = pre_pool_w // 2
                    sparse_positions = []
                    for out_y in range(out_h):
                        for out_x in range(out_w):
                            in_y = 2 * out_y
                            in_x = 2 * out_x
                            in_patch_idx = in_y * pre_pool_w + in_x
                            for channel in range(channels):
                                sparse_positions.append(in_patch_idx * channels + channel)

                    layout["sparse"] = True
                    layout["sparse_positions"] = sparse_positions
                    layout["total_slots"] = pre_pool_h * pre_pool_w * channels
                    layout["pre_pool_h"] = pre_pool_h
                    layout["pre_pool_w"] = pre_pool_w

        return layout

    def compute_pre_pool_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.conv_params:
            return None, None
        if self.input_shape is None:
            warnings.warn(
                "_compute_pre_pool_dimensions: input_shape is None, "
                "cannot compute pre-pool spatial dimensions. Pass input_shape to converter.",
                UserWarning,
                stacklevel=2,
            )
            return None, None

        shape = tuple(self.input_shape)
        if len(shape) < 2:
            return None, None

        h, w = int(shape[-2]), int(shape[-1])
        pool_idx = 0
        num_pools = len(self.pool_params)

        for conv in self.conv_params:
            kernel_size = conv["kernel_size"]
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            stride = conv["stride"]
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = conv["padding"]
            if isinstance(padding, int):
                padding = (padding, padding)

            kh, kw = kernel_size
            sh, sw = stride
            ph, pw = padding
            h = (h + 2 * ph - kh) // sh + 1
            w = (w + 2 * pw - kw) // sw + 1

            if pool_idx < num_pools - 1:
                pool = self.pool_params[pool_idx]
                psh, psw = pool["stride"]
                h //= psh
                w //= psw
                pool_idx += 1

        return h, w

    def compute_layout_before_flatten(self) -> Optional[Dict[str, int]]:
        if not self.conv_params:
            return None

        h, w = 8, 8
        if self.input_shape is not None and len(self.input_shape) >= 2:
            h, w = int(self.input_shape[-2]), int(self.input_shape[-1])

        channels = 1
        pool_idx = 0
        for conv in self.conv_params:
            kernel_size = conv["kernel_size"]
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            stride = conv["stride"]
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = conv["padding"]
            if isinstance(padding, int):
                padding = (padding, padding)

            kh, kw = kernel_size
            sh, sw = stride
            ph, pw = padding
            h = (h + 2 * ph - kh) // sh + 1
            w = (w + 2 * pw - kw) // sw + 1
            channels = conv["out_channels"]

            if pool_idx < len(self.pool_params):
                pool = self.pool_params[pool_idx]
                h //= pool["stride"][0]
                w //= pool["stride"][1]
                pool_idx += 1

        return {"num_patches": h * w, "patch_features": channels}

    def build_runtime_config(self) -> Optional[Dict[str, int]]:
        if not self.conv_params:
            return None

        last_conv = self.conv_params[-1]
        image_height = (
            int(self.input_shape[-2])
            if self.input_shape is not None and len(self.input_shape) >= 2
            else 8
        )
        image_width = (
            int(self.input_shape[-1])
            if self.input_shape is not None and len(self.input_shape) >= 2
            else 8
        )
        pool_size = 2
        pool_stride = 2
        if self.pool_params:
            last_pool = self.pool_params[-1]
            pool_size = last_pool["kernel_size"][0]
            pool_stride = last_pool["stride"][0]

        return {
            "image_height": image_height,
            "image_width": image_width,
            "channels": last_conv["out_channels"],
            "pool_size": pool_size,
            "pool_stride": pool_stride,
        }
