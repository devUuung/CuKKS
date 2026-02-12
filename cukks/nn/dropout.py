"""
EncryptedDropout - Dropout layer for encrypted tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedDropout(EncryptedModule):
    """Dropout layer for encrypted inference.
    
    During inference, dropout is a no-op (returns input unchanged).
    This module mirrors torch.nn.Dropout for compatibility.
    
    Args:
        p: Dropout probability (default: 0.5). Stored for compatibility
           but has no effect during inference.
    
    Example:
        >>> dropout = EncryptedDropout(p=0.5)
        >>> enc_output = dropout(enc_input)  # Returns enc_input unchanged
    """
    
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass - returns input unchanged (no-op for inference).
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            The same encrypted tensor (dropout is disabled during inference).
        """
        return x
    
    def mult_depth(self) -> int:
        """Dropout has zero multiplicative depth."""
        return 0
    
    def extra_repr(self) -> str:
        """Return string representation with dropout probability."""
        return f"p={self.p}"
    
    @classmethod
    def from_torch(cls, module: torch.nn.Dropout) -> "EncryptedDropout":
        """Create from a PyTorch Dropout layer.
        
        Args:
            module: The PyTorch Dropout layer to convert.
            
        Returns:
            EncryptedDropout with the same dropout probability.
        """
        return cls(p=module.p)
