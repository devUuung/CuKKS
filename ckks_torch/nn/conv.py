"""
EncryptedConv2d - Encrypted 2D convolution layer.

Convolution in CKKS is implemented by unrolling the convolution into
matrix-vector multiplication (im2col method).
"""

from __future__ import annotations

import math
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
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)
        
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
    
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
                "2D CNN layout. See: ckks_torch.CKKSInferenceContext.encrypt_cnn_input()"
            )
            
        elif input_ndim == 3:
            raise RuntimeError(
                "EncryptedConv2d does not support 3D input in pure HE mode. "
                "Pre-process input using ctx.encrypt_cnn_input() to create a "
                "2D CNN layout. See: ckks_torch.CKKSInferenceContext.encrypt_cnn_input()"
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
            return x.matmul(self.weight_matrix, self.bias)
            
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
        
        This performs convolution entirely in encrypted domain without decryption.
        Uses block-diagonal matrix multiplication approach.
        
        Input: flattened patches of shape (num_patches * patch_features,)
        Output: flattened output of shape (num_patches * out_channels,)
        
        The block-diagonal structure:
        | W  0  0  |   | patch0 |   | out0 |
        | 0  W  0  | @ | patch1 | = | out1 |
        | 0  0  W  |   | patch2 |   | out2 |
        
        Args:
            x: Encrypted im2col patches with _cnn_layout metadata.
            
        Returns:
            Encrypted output of shape (num_patches * out_channels,).
        """
        layout = x._cnn_layout
        num_patches = layout['num_patches']
        patch_features = layout['patch_features']
        
        self.input_shape = (num_patches, patch_features)
        self.output_shape = (num_patches, self.out_channels)
        
        total_out = num_patches * self.out_channels
        total_in = num_patches * patch_features
        
        # Update shape for matmul to understand the flat structure
        x._shape = (total_in,)
        
        # Create block-diagonal weight matrix
        block_weight = torch.zeros(total_out, total_in, dtype=torch.float64)
        for p in range(num_patches):
            row_start = p * self.out_channels
            row_end = (p + 1) * self.out_channels
            col_start = p * patch_features
            col_end = (p + 1) * patch_features
            block_weight[row_start:row_end, col_start:col_end] = self.weight_matrix.to(torch.float64)
        
        # Create block-diagonal bias
        if self.bias is not None:
            block_bias = self.bias.to(torch.float64).repeat(num_patches)
        else:
            block_bias = None
        
        # Use standard matmul with block-diagonal weights
        # matmul expects (out_features, in_features) weight
        result = x.matmul(block_weight, block_bias)
        
        # Update CNN layout for next layer
        result._cnn_layout = {
            'num_patches': num_patches,
            'patch_features': self.out_channels,
            'original_shape': layout.get('original_shape'),
        }
        result._shape = (num_patches, self.out_channels)
        
        return result

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
