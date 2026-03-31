from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedConv1d(EncryptedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = weight.detach().to(dtype=torch.float64, device="cpu")
        self.weight_matrix = self._build_weight_matrix(self.weight)
        self.bias = bias.detach().to(dtype=torch.float64, device="cpu") if bias is not None else None

        self._wm_list: list | None = None
        self._wm_bias_list: list | None = None
        self._wm_hash: int = 0
        self._wm_diag_nonzero: list | None = None
        self._packed_compact_cache: dict[tuple[int, int, int], list[int]] = {}
        self._ensure_weight_matrix_cache()

        self.register_parameter("weight", self.weight)
        self.register_parameter("weight_matrix", self.weight_matrix)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

        self.input_shape: Optional[tuple[int, ...]] = None
        self.output_shape: Optional[tuple[int, ...]] = None

    def _build_weight_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        out_channels, channels_per_group, kernel_width = weight.shape
        full_width = self.in_channels * kernel_width

        if self.groups == 1:
            return weight.reshape(out_channels, full_width)

        weight_matrix = torch.zeros(out_channels, full_width, dtype=torch.float64)
        out_per_group = out_channels // self.groups
        in_per_group = self.in_channels // self.groups

        for out_idx in range(out_channels):
            group_idx = out_idx // out_per_group
            col_start = group_idx * in_per_group * kernel_width
            col_end = col_start + channels_per_group * kernel_width
            weight_matrix[out_idx, col_start:col_end] = weight[out_idx].reshape(-1)

        return weight_matrix

    def _ensure_weight_matrix_cache(self) -> None:
        if self._wm_list is not None and self._wm_diag_nonzero is not None:
            return

        wm_cpu = self.weight_matrix.detach().to(dtype=torch.float64, device="cpu")
        self._wm_list = wm_cpu.tolist()
        self._wm_bias_list = self.bias.reshape(-1).tolist() if self.bias is not None else None
        digest = hashlib.md5(wm_cpu.contiguous().numpy().tobytes()).digest()[:8]
        self._wm_hash = int.from_bytes(digest, "little")
        out_f, in_f = wm_cpu.shape
        rows = torch.arange(out_f)
        self._wm_diag_nonzero = [bool(wm_cpu[rows, (rows + d) % in_f].any()) for d in range(in_f)]

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        if getattr(x, "_packed_batch", False):
            sample_shape = getattr(x, "_packed_sample_shape", None) or x.shape[1:]
            if len(sample_shape) == 2 and sample_shape[1] == self.weight_matrix.shape[1]:
                self.input_shape = tuple(sample_shape)
                self.output_shape = (sample_shape[0], self.out_channels)
                self._ensure_weight_matrix_cache()
                return x.matmul(
                    self.weight_matrix,
                    self.bias,
                    weight_list=self._wm_list,
                    bias_list=self._wm_bias_list,
                    weight_hash=self._wm_hash,
                    diag_nonzero=self._wm_diag_nonzero,
                )

        input_ndim = len(x.shape)
        if input_ndim == 3:
            raise RuntimeError(
                "EncryptedConv1d does not support 3D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout. See: cukks.CKKSInferenceContext.encrypt_cnn_input()"
            )

        if input_ndim == 2:
            if hasattr(x, "_cnn_layout") and x._cnn_layout is not None:
                return self._forward_he_packed(x)
            raise RuntimeError(
                "EncryptedConv1d requires CNN layout for 2D input. "
                "Pre-process input using ctx.encrypt_cnn_input() to set "
                "_cnn_layout metadata."
            )

        if input_ndim == 1:
            self.input_shape = x.shape
            self.output_shape = (self.out_channels,)
            self._ensure_weight_matrix_cache()
            return x.matmul(
                self.weight_matrix,
                self.bias,
                weight_list=self._wm_list,
                bias_list=self._wm_bias_list,
                weight_hash=self._wm_hash,
                diag_nonzero=self._wm_diag_nonzero,
            )

        raise ValueError(f"Expected 1D-3D input, got {input_ndim}D with shape {x.shape}")

    def _forward_he_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv1d requires _cnn_layout metadata for packed forward")

        num_patches = layout["num_patches"]
        patch_features = layout["patch_features"]
        batch_size = layout.get("batch_size", 1)

        self.input_shape = (num_patches, patch_features)
        self.output_shape = (num_patches, self.out_channels)

        total_out = num_patches * self.out_channels
        total_in = num_patches * patch_features

        matrix_elements = total_out * total_in
        max_matrix_elements = 100_000_000
        if matrix_elements <= max_matrix_elements and total_in <= 2048:
            return self._forward_he_packed_dense(
                x,
                num_patches,
                patch_features,
                total_out,
                total_in,
                batch_size=batch_size,
            )

        return self._forward_he_packed_compact(
            x,
            num_patches,
            patch_features,
            batch_size=batch_size,
        )

    def _forward_he_packed_dense(
        self,
        x: "EncryptedTensor",
        num_patches: int,
        patch_features: int,
        total_out: int,
        total_in: int,
        *,
        batch_size: int = 1,
    ) -> "EncryptedTensor":
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv1d requires _cnn_layout metadata for dense packed forward")

        from ..tensor import EncryptedTensor as _ET

        x_flat = _ET(x._cipher, (total_in,), x._context, x._depth)
        x_flat._needs_rescale = x._needs_rescale

        block_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        for patch_idx in range(num_patches):
            row_start = patch_idx * self.out_channels
            row_end = (patch_idx + 1) * self.out_channels
            col_start = patch_idx * patch_features
            col_end = (patch_idx + 1) * patch_features
            block_weight[row_start:row_end, col_start:col_end] = self.weight_matrix

        block_bias: torch.Tensor | None = None
        if self.bias is not None:
            block_bias = self.bias.to(torch.float64).repeat(num_patches)

        result = x_flat.matmul(block_weight, block_bias)
        npp = layout.get("num_patches_per_image", num_patches // max(batch_size, 1))
        result._cnn_layout = {
            "num_patches": num_patches,
            "patch_features": self.out_channels,
            "original_shape": layout.get("original_shape"),
            "batch_size": batch_size,
            "num_patches_per_image": npp,
        }
        result._shape = (num_patches, self.out_channels)

        if batch_size > 1:
            result._packed_batch = True
            result._batch_size = batch_size
            result._slots_per_sample = (num_patches // batch_size) * self.out_channels

        return result

    def _forward_he_packed_compact(
        self,
        x: "EncryptedTensor",
        num_patches: int,
        patch_features: int,
        *,
        batch_size: int = 1,
    ) -> "EncryptedTensor":
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv1d requires _cnn_layout metadata for compact packed forward")

        total_out = num_patches * self.out_channels
        total_in = num_patches * patch_features
        slot_count = x._cipher.size
        out_channels = self.out_channels

        x_flat = x.view(total_in)
        diagonal_offsets = self._get_packed_compact_offsets(num_patches, patch_features, slot_count)
        weight_matrix = self.weight_matrix

        accumulator = None
        num_diags = len(diagonal_offsets)
        if num_diags == 0:
            raise RuntimeError("Conv1d: no non-zero diagonals found")

        group_size = max(1, math.ceil(math.sqrt(num_diags)))
        giant_groups: dict[int, list[int]] = defaultdict(list)
        for diagonal in diagonal_offsets:
            giant_groups[diagonal // group_size].append(diagonal)

        baby_ciphers: dict[int, EncryptedTensor] = {}
        for baby_step in sorted({diagonal % group_size for diagonal in diagonal_offsets}):
            baby_ciphers[baby_step] = x_flat.rotate(baby_step) if baby_step != 0 else x_flat

        for giant_idx, diagonals_in_group in sorted(giant_groups.items()):
            giant_step = giant_idx * group_size
            block_acc = None

            for diagonal in diagonals_in_group:
                baby_step = diagonal % group_size
                diag_vals = torch.zeros(slot_count, dtype=torch.float64)
                has_nonzero = False

                for row in range(min(total_out, slot_count)):
                    col = (row + diagonal) % total_in
                    row_patch = row // out_channels
                    col_patch = col // patch_features
                    if row_patch != col_patch:
                        continue
                    row_channel = row % out_channels
                    col_feature = col % patch_features
                    value = weight_matrix[row_channel, col_feature].item()
                    if abs(value) > 1e-30:
                        diag_vals[row] = value
                        has_nonzero = True

                if not has_nonzero:
                    continue

                if giant_step != 0:
                    diag_vals = torch.roll(diag_vals, giant_step)

                term = baby_ciphers[baby_step].mul(diag_vals.tolist()).rescale()
                block_acc = term if block_acc is None else block_acc.add(term)

            if block_acc is None:
                continue
            if giant_step != 0:
                block_acc = block_acc.rotate(giant_step)
            accumulator = block_acc if accumulator is None else accumulator.add(block_acc)

        if accumulator is None:
            raise RuntimeError("Conv1d: all weight diagonals are zero")

        if self.bias is not None:
            bias_plain = [0.0] * slot_count
            for patch_idx in range(num_patches):
                for channel_idx in range(out_channels):
                    index = patch_idx * out_channels + channel_idx
                    if index < slot_count:
                        bias_plain[index] = self.bias[channel_idx].item()
            accumulator = accumulator.add(bias_plain)

        npp = layout.get("num_patches_per_image", num_patches // max(batch_size, 1))
        accumulator._cnn_layout = {
            "num_patches": num_patches,
            "patch_features": self.out_channels,
            "original_shape": layout.get("original_shape"),
            "batch_size": batch_size,
            "num_patches_per_image": npp,
        }
        accumulator._shape = (num_patches, self.out_channels)

        if batch_size > 1:
            accumulator._packed_batch = True
            accumulator._batch_size = batch_size
            accumulator._slots_per_sample = (num_patches // batch_size) * self.out_channels

        return accumulator

    def _get_packed_compact_offsets(
        self,
        num_patches: int,
        patch_features: int,
        slot_count: int,
    ) -> list[int]:
        cache_key = (num_patches, patch_features, slot_count)
        cached = self._packed_compact_cache.get(cache_key)
        if cached is not None:
            return cached

        total_in = num_patches * patch_features
        out_channels = self.out_channels
        nonzero_diags = set()
        for patch_idx in range(num_patches):
            for delta in range(-(out_channels - 1), patch_features):
                nonzero_diags.add((patch_idx * (patch_features - out_channels) + delta) % total_in)

        cached = sorted(nonzero_diags)
        self._packed_compact_cache[cache_key] = cached
        return cached

    def get_output_size(self, input_length: int) -> int:
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        return (input_length + 2 * self.padding - effective_kernel) // self.stride + 1

    @staticmethod
    def unfold_input(
        x: torch.Tensor,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x_2d = x.unsqueeze(2)
        patches = F.unfold(
            x_2d,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, padding),
            stride=(1, stride),
        )
        patches = patches.transpose(1, 2)
        return patches.squeeze(0)

    @classmethod
    def from_torch(cls, module: nn.Conv1d) -> "EncryptedConv1d":
        kernel_size = cast(tuple[int], module.kernel_size)[0]
        stride = cast(tuple[int], module.stride)[0]
        dilation = cast(tuple[int], module.dilation)[0]

        if isinstance(module.padding, str):
            if module.padding == "same":
                padding = ((kernel_size - 1) * dilation) // 2
            else:
                padding = 0
        else:
            padding = cast(tuple[int], module.padding)[0]

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=kernel_size,
            weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            stride=stride,
            padding=padding,
            groups=module.groups,
            dilation=dilation,
        )

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}"
        )
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        return s
