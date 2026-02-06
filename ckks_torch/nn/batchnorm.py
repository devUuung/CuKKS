"""
Encrypted BatchNorm layers.

BatchNorm is typically "folded" into the preceding layer during inference,
eliminating the need for separate encrypted operations. These classes help
with that folding process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


def fold_batchnorm_into_linear(
    linear: torch.nn.Linear,
    bn: torch.nn.BatchNorm1d,
) -> torch.nn.Linear:
    """Fold BatchNorm1d parameters into a Linear layer.
    
    After training, BatchNorm can be folded into the preceding layer:
    
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    For a Linear layer W*x + b:
    y = gamma * (W*x + b - mean) / sqrt(var + eps) + beta
      = gamma/sqrt(var+eps) * W * x + gamma/sqrt(var+eps) * (b - mean) + beta
      = W_folded * x + b_folded
    
    Args:
        linear: The Linear layer.
        bn: The BatchNorm1d layer.
        
    Returns:
        New Linear layer with folded parameters.
    """
    # Validate running stats exist
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm must have running_mean and running_var for inference")
    
    gamma = bn.weight.data if bn.weight is not None else torch.ones(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
    beta = bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # Compute scale factor
    std = torch.sqrt(var + eps)
    scale = gamma / std
    
    W = linear.weight.data.clone()
    b = linear.bias.data.clone() if linear.bias is not None else torch.zeros(W.shape[0])  # pyright: ignore[reportUnnecessaryComparison]
    
    # Fold
    W_folded = W * scale.view(-1, 1)
    b_folded = scale * (b - mean) + beta
    
    # Create new linear
    folded = torch.nn.Linear(linear.in_features, linear.out_features, bias=True)
    folded.weight.data = W_folded
    assert folded.bias is not None  # bias=True guarantees this
    folded.bias.data = b_folded
    
    return folded


def fold_batchnorm_into_conv(
    conv: torch.nn.Conv2d,
    bn: torch.nn.BatchNorm2d,
) -> torch.nn.Conv2d:
    """Fold BatchNorm2d parameters into a Conv2d layer.
    
    Args:
        conv: The Conv2d layer.
        bn: The BatchNorm2d layer.
        
    Returns:
        New Conv2d layer with folded parameters.
    """
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm must have running_mean and running_var for inference")
    
    gamma = bn.weight.data if bn.weight is not None else torch.ones(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
    beta = bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    std = torch.sqrt(var + eps)
    scale = gamma / std
    
    W = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(W.shape[0])  # pyright: ignore[reportUnnecessaryComparison]
    
    W_folded = W * scale.view(-1, 1, 1, 1)
    b_folded = scale * (b - mean) + beta
    
    kernel_size = cast(Tuple[int, int], conv.kernel_size)
    stride = cast(Tuple[int, int], conv.stride)
    padding = cast(Tuple[int, int], conv.padding)
    
    folded = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels, kernel_size,
        stride=stride, padding=padding, bias=True
    )
    folded.weight.data = W_folded
    assert folded.bias is not None
    folded.bias.data = b_folded
    
    return folded


class EncryptedBatchNorm1d(EncryptedModule):
    """Encrypted BatchNorm1d (identity when folded).
    
    During inference, BatchNorm is typically folded into the preceding layer.
    If not folded, it becomes a simple affine transformation:
    
    y = scale * x + shift
    
    where scale = gamma / sqrt(var + eps) and shift = beta - gamma * mean / sqrt(var + eps)
    
    Args:
        num_features: Number of features.
        scale: Pre-computed scale factors.
        shift: Pre-computed shift factors.
    """
    
    def __init__(
        self,
        num_features: int,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.scale = scale.detach().to(dtype=torch.float64, device="cpu")
        self.shift = shift.detach().to(dtype=torch.float64, device="cpu")
        
        self.register_parameter("scale", self.scale)
        self.register_parameter("shift", self.shift)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply the affine transformation.
        
        Args:
            x: Encrypted input.
            
        Returns:
            Encrypted output after affine transformation.
        """
        # y = scale * x + shift
        slot_count = x._cipher.size
        scale_list = self.scale.tolist()
        shift_list = self.shift.tolist()
        if len(scale_list) < slot_count:
            scale_list = scale_list + [0.0] * (slot_count - len(scale_list))
        if len(shift_list) < slot_count:
            shift_list = shift_list + [0.0] * (slot_count - len(shift_list))
        result = x.mul(scale_list)
        result = result.rescale()
        result = result.add(shift_list)
        return result
    
    def mult_depth(self) -> int:
        """BatchNorm uses 1 multiplication."""
        return 1
    
    @classmethod
    def from_torch(cls, bn: torch.nn.BatchNorm1d) -> "EncryptedBatchNorm1d":
        """Create from a PyTorch BatchNorm1d layer.
        
        Args:
            bn: The BatchNorm1d layer.
            
        Returns:
            EncryptedBatchNorm1d.
        """
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError("BatchNorm must have running_mean and running_var for inference")
        
        gamma = bn.weight.data if bn.weight is not None else torch.ones(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
        beta = bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        std = torch.sqrt(var + eps)
        scale = gamma / std
        shift = beta - gamma * mean / std
        
        return cls(bn.num_features, scale, shift)
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"


class EncryptedBatchNorm2d(EncryptedModule):
    """Encrypted BatchNorm2d (identity when folded).
    
    Same as BatchNorm1d but for 2D inputs (images).
    """
    
    def __init__(
        self,
        num_features: int,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.scale = scale.detach().to(dtype=torch.float64, device="cpu")
        self.shift = shift.detach().to(dtype=torch.float64, device="cpu")
        
        self.register_parameter("scale", self.scale)
        self.register_parameter("shift", self.shift)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Apply the affine transformation."""
        if hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
            raise RuntimeError(
                "EncryptedBatchNorm2d does not support CNN packed layout. "
                "Use fold_batchnorm=True in convert() to fold BatchNorm into Conv2d."
            )
        slot_count = x._cipher.size
        scale_list = self.scale.tolist()
        shift_list = self.shift.tolist()
        if len(scale_list) < slot_count:
            scale_list = scale_list + [0.0] * (slot_count - len(scale_list))
        if len(shift_list) < slot_count:
            shift_list = shift_list + [0.0] * (slot_count - len(shift_list))
        result = x.mul(scale_list)
        result = result.rescale()
        result = result.add(shift_list)
        return result
    
    def mult_depth(self) -> int:
        return 1
    
    @classmethod
    def from_torch(cls, bn: torch.nn.BatchNorm2d) -> "EncryptedBatchNorm2d":
        """Create from a PyTorch BatchNorm2d layer."""
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError("BatchNorm must have running_mean and running_var for inference")
        
        gamma = bn.weight.data if bn.weight is not None else torch.ones(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
        beta = bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features)  # pyright: ignore[reportUnnecessaryComparison]
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        std = torch.sqrt(var + eps)
        scale = gamma / std
        shift = beta - gamma * mean / std
        
        return cls(bn.num_features, scale, shift)
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"
