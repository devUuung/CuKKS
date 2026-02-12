"""
EncryptedConv2d - Encrypted 2D convolution layer.

Convolution in CKKS is implemented by unrolling the convolution into
matrix-vector multiplication (im2col method).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedConv2d(EncryptedModule):
    """Encrypted 2D convolution layer.
    
    Implements convolution using the im2col method:
    1. Unfold input into patches (done on plaintext before encryption)
    2. Perform matrix multiplication with reshaped kernel
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        weight: Convolution kernel of shape (out_channels, in_channels, kH, kW).
        bias: Optional bias of shape (out_channels,).
        stride: Stride of the convolution. Default: 1
        padding: Padding added to input. Default: 0
        
    Note:
        For encrypted inference, the input must be pre-processed:
        - Image is unfolded into patches using im2col
        - Each patch is encrypted separately or packed together
        
        This layer expects the input to already be in unfolded format.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation
        
        self.groups = groups
        
        # Store weight reshaped for matrix multiplication
        # Original: (out_channels, in_channels, kH, kW)
        # Reshaped: (out_channels, in_channels * kH * kW)
        self.weight = weight.detach().to(dtype=torch.float64, device="cpu")
        self.weight_matrix = self.weight.reshape(out_channels, -1)
        self.bias = bias.detach().to(dtype=torch.float64, device="cpu") if bias is not None else None

        self._wm_list: list | None = None
        self._wm_bias_list: list | None = None
        self._wm_hash: int = 0
        self._wm_diag_nonzero: list | None = None
        if hasattr(self, "weight_matrix"):
            self._ensure_weight_matrix_cache()
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)
        
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None

    def _ensure_weight_matrix_cache(self) -> None:
        if self._wm_list is not None and self._wm_diag_nonzero is not None:
            return
        weight_matrix = getattr(self, "weight_matrix", None)
        if weight_matrix is None:
            return

        wm_cpu = weight_matrix.detach().to(dtype=torch.float64, device="cpu")
        self._wm_list = wm_cpu.tolist()
        self._wm_bias_list = self.bias.reshape(-1).tolist() if self.bias is not None else None
        digest = hashlib.md5(wm_cpu.contiguous().numpy().tobytes()).digest()[:8]
        self._wm_hash = int.from_bytes(digest, 'little')
        out_f, in_f = wm_cpu.shape
        rows = torch.arange(out_f)
        self._wm_diag_nonzero = [bool(wm_cpu[rows, (rows + d) % in_f].any()) for d in range(in_f)]
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass on encrypted input using pure HE operations.
        
        Requires input pre-processed via ctx.encrypt_cnn_input() for CNN
        inference. Direct 4D/3D image inputs are not supported in pure HE mode.
        
        Args:
            x: Encrypted input tensor.
               - 2D with _cnn_layout: (num_patches, patch_features) - HE matmul
               - 1D: (flattened,) - single patch matmul
               
        Returns:
            Encrypted output after convolution.
            
        Raises:
            RuntimeError: If input is 4D, 3D, or 2D without CNN layout.
        """
        input_ndim = len(x.shape)
        
        if input_ndim == 4:
            raise RuntimeError(
                "EncryptedConv2d does not support 4D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout. See: cukks.CKKSInferenceContext.encrypt_cnn_input()"
            )
            
        elif input_ndim == 3:
            raise RuntimeError(
                "EncryptedConv2d does not support 3D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout. See: cukks.CKKSInferenceContext.encrypt_cnn_input()"
            )
            
        elif input_ndim == 2:
            # 2D input: (num_patches, patch_features) - pre-unfolded via encrypt_cnn_input
            # Check if this is a CNN layout (already im2col processed)
            if hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
                # Real HE: Use packed matmul for im2col patches
                # Each row of patches needs to be multiplied by weight matrix
                return self._forward_he_packed(x)
            else:
                raise RuntimeError(
                    "EncryptedConv2d requires CNN layout for 2D input. "
                    "Pre-process input using ctx.encrypt_cnn_input() to set "
                    "_cnn_layout metadata."
                )
        
        elif input_ndim == 1:
            # 1D input: single flattened patch
            self.input_shape = x.shape
            self.output_shape = (self.out_channels,)
            self._ensure_weight_matrix_cache()
            return x.matmul(self.weight_matrix, self.bias,
                            weight_list=self._wm_list,
                            bias_list=self._wm_bias_list,
                            weight_hash=self._wm_hash,
                            diag_nonzero=self._wm_diag_nonzero)
            
        else:
            raise ValueError(
                f"Expected 1D-4D input, got {input_ndim}D with shape {x.shape}"
            )
    
    def mult_depth(self) -> int:
        """Convolution uses 1 multiplication."""
        return 1
    
    def get_output_size(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int]:
        """Compute the output spatial dimensions.
        
        Args:
            input_height: Height of the input.
            input_width: Width of the input.
            
        Returns:
            Tuple of (output_height, output_width).
        """
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        
        # Compute effective kernel size with dilation
        effective_kH = dH * (kH - 1) + 1
        effective_kW = dW * (kW - 1) + 1
        
        out_h = (input_height + 2 * pH - effective_kH) // sH + 1
        out_w = (input_width + 2 * pW - effective_kW) // sW + 1
        
        return out_h, out_w
    
    def _forward_he_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass using real HE matmul for packed im2col patches.

        Input layout (flattened):
            [p0_f0 .. p0_fK, p1_f0 .. p1_fK, ...]  (K = patch_features)

        Output layout (flattened):
            [p0_o0 .. p0_oC, p1_o0 .. p1_oC, ...]  (C = out_channels)

        For small inputs the block-diagonal matrix is materialized directly.
        For large inputs (>100M elements) we use the compact BSGS matmul on
        the small per-patch weight matrix W (out_channels × patch_features)
        which avoids allocating the full N×N block-diagonal.
        """
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv2d requires _cnn_layout metadata for packed forward")
        num_patches = layout['num_patches']
        patch_features = layout['patch_features']

        self.input_shape = (num_patches, patch_features)
        self.output_shape = (num_patches, self.out_channels)

        total_out = num_patches * self.out_channels
        total_in = num_patches * patch_features

        matrix_elements = total_out * total_in
        MAX_MATRIX_ELEMENTS = 100_000_000

        if matrix_elements <= MAX_MATRIX_ELEMENTS and total_in <= 2048:
            return self._forward_he_packed_dense(x, num_patches, patch_features, total_out, total_in)

        return self._forward_he_packed_compact(x, num_patches, patch_features)

    def _forward_he_packed_dense(
        self,
        x: "EncryptedTensor",
        num_patches: int,
        patch_features: int,
        total_out: int,
        total_in: int,
    ) -> "EncryptedTensor":
        """Small-input path: materialize the full block-diagonal matrix."""
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv2d requires _cnn_layout metadata for dense packed forward")

        x = x.view(total_in)

        block_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        for p in range(num_patches):
            row_start = p * self.out_channels
            row_end = (p + 1) * self.out_channels
            col_start = p * patch_features
            col_end = (p + 1) * patch_features
            block_weight[row_start:row_end, col_start:col_end] = self.weight_matrix.to(torch.float64)

        block_bias: torch.Tensor | None = None
        if self.bias is not None:
            block_bias = self.bias.to(torch.float64).repeat(num_patches)

        result = x.matmul(block_weight, block_bias)

        result._cnn_layout = {
            'num_patches': num_patches,
            'patch_features': self.out_channels,
            'original_shape': layout.get('original_shape'),
        }
        result._shape = (num_patches, self.out_channels)
        return result

    def _forward_he_packed_compact(
        self,
        x: "EncryptedTensor",
        num_patches: int,
        patch_features: int,
    ) -> "EncryptedTensor":
        """Large-input path: diagonal method without materializing block-diagonal.

        For block-diagonal B (total_out × total_in) with repeated block W (C × K):
            B[p*C+c, p*K+k] = W[c, k]   (same block p)
            B[i, j] = 0                  (different blocks)

        Standard diagonal encoding for rectangular M×N matrix:
            diag_d[i] = M[i % M_rows, (i + d) % M_cols]

        We compute each diagonal analytically from W, never allocating B.
        Number of non-zero diagonals ≤ K × ceil(total_out / total_in + 1).
        Memory per diagonal: O(num_slots).
        """
        layout = x._cnn_layout
        if layout is None:
            raise RuntimeError("EncryptedConv2d requires _cnn_layout metadata for compact packed forward")
        total_out = num_patches * self.out_channels
        total_in = num_patches * patch_features

        slot_count = x._cipher.size
        W = self.weight_matrix.to(torch.float64)
        C = self.out_channels
        K = patch_features

        x_flat = x.view(total_in)

        nonzero_diags = set()
        for p in range(num_patches):
            for delta in range(-(C - 1), K):
                d = (p * (K - C) + delta) % total_in
                nonzero_diags.add(d)

        accumulator = None

        for d in sorted(nonzero_diags):
            diag_vals = torch.zeros(slot_count, dtype=torch.float64)
            has_nonzero = False

            for i in range(min(total_out, slot_count)):
                row = i
                col = (i + d) % total_in

                row_patch = row // C
                row_c = row % C
                col_patch = col // K
                col_k = col % K

                if row_patch == col_patch:
                    val = W[row_c, col_k].item()
                    if abs(val) > 1e-30:
                        diag_vals[i] = val
                        has_nonzero = True

            if not has_nonzero:
                continue

            rotated = x_flat.rotate(d) if d != 0 else x_flat
            term = rotated.mul(diag_vals.tolist())
            term = term.rescale()

            if accumulator is None:
                accumulator = term
            else:
                accumulator = accumulator.add(term)

        if accumulator is None:
            raise RuntimeError("Conv2d: all weight diagonals are zero")

        if self.bias is not None:
            bias_plain = [0.0] * slot_count
            for p in range(num_patches):
                for c in range(C):
                    idx = p * C + c
                    if idx < slot_count:
                        bias_plain[idx] = self.bias[c].item()
            accumulator = accumulator.add(bias_plain)

        accumulator._cnn_layout = {
            'num_patches': num_patches,
            'patch_features': self.out_channels,
            'original_shape': layout.get('original_shape'),
        }
        accumulator._shape = (num_patches, self.out_channels)
        return accumulator

    @staticmethod
    def unfold_input(
        x: torch.Tensor,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Unfold input tensor for im2col convolution.
        
        This should be called BEFORE encryption.
        
        Args:
            x: Input tensor of shape (C, H, W) or (1, C, H, W).
            kernel_size: Kernel size (kH, kW).
            stride: Stride (sH, sW).
            padding: Padding (pH, pW).
            
        Returns:
            Unfolded tensor of shape (num_patches, C * kH * kW).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply padding
        if padding != (0, 0):
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
        
        # Unfold: (N, C, H, W) -> (N, C*kH*kW, num_patches)
        patches = F.unfold(x, kernel_size, stride=stride)
        
        # Transpose: (N, C*kH*kW, num_patches) -> (N, num_patches, C*kH*kW)
        patches = patches.transpose(1, 2)
        
        # Remove batch: (num_patches, C*kH*kW)
        return patches.squeeze(0)
    
    @classmethod
    def from_torch(cls, conv: torch.nn.Conv2d) -> "EncryptedConv2d":
        """Create from a PyTorch Conv2d layer.
        
        Args:
            conv: The PyTorch Conv2d layer to convert.
            
        Returns:
            EncryptedConv2d with copied weights.
        """
        kernel_size = cast(Tuple[int, int], conv.kernel_size)
        stride = cast(Tuple[int, int], conv.stride)
        dilation = cast(Tuple[int, int], conv.dilation)
        
        if isinstance(conv.padding, str):
            if conv.padding == 'same':
                padding: Tuple[int, int] = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = (0, 0)
        else:
            padding = cast(Tuple[int, int], conv.padding)
            
        return cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_size,
            weight=conv.weight.data,
            bias=conv.bias.data if conv.bias is not None else None,
            stride=stride,
            padding=padding,
            groups=conv.groups,
            dilation=dilation,
        )
    
    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}"
        )
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        return s
