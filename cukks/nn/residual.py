"""
EncryptedResidualBlock - Residual connection support for encrypted inference.

This module provides ResNet-style skip connection support, enabling patterns like:
    out = block(x) + x
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedResidualBlock(EncryptedModule):
    """Residual block with skip connection for encrypted inference.
    
    Implements the pattern: output = block(x) + identity
    where identity = downsample(x) if downsample is provided, else x.
    
    This enables ResNet-style architectures with encrypted data.
    
    Args:
        block: The main processing block (e.g., conv layers).
        downsample: Optional projection for matching dimensions.
            Used when input/output shapes differ.
    
    Example:
        >>> # Simple residual: out = linear(x) + x
        >>> linear = EncryptedLinear(64, 64, weight)
        >>> residual = EncryptedResidualBlock(linear)
        >>> out = residual(enc_input)
        
        >>> # With downsample for shape mismatch
        >>> block = EncryptedSequential(conv1, relu, conv2)
        >>> downsample = EncryptedConv2d(...)  # 1x1 conv for projection
        >>> residual = EncryptedResidualBlock(block, downsample)
    """
    
    def __init__(
        self,
        block: EncryptedModule,
        downsample: Optional[EncryptedModule] = None,
    ) -> None:
        super().__init__()
        self.block = block
        self.downsample = downsample
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass with residual connection.
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            block(x) + identity, where identity is x or downsample(x).
        """
        identity = x
        out = self.block(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        return out.add(identity)
    
    def mult_depth(self) -> int:
        """Estimate multiplicative depth.
        
        The residual addition is free (no multiplication).
        Depth is max of block and downsample paths.
        """
        block_depth = self.block.mult_depth()
        downsample_depth = self.downsample.mult_depth() if self.downsample else 0
        return max(block_depth, downsample_depth)
    
    def extra_repr(self) -> str:
        """Extra info for repr."""
        return f"has_downsample={self.downsample is not None}"
