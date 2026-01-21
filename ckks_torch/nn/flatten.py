"""
EncryptedFlatten - Flatten operation for encrypted tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedFlatten(EncryptedModule):
    """Flattens encrypted tensor dimensions.
    
    This is a logical reshape - no encrypted operations are performed.
    Mirrors torch.nn.Flatten.
    
    Args:
        start_dim: First dimension to flatten (default: 0).
        end_dim: Last dimension to flatten (default: -1, all remaining).
    
    Note:
        For encrypted inference, we typically don't have batch dimensions
        (each sample is encrypted separately), so the default flattens
        everything to 1D.
    """
    
    def __init__(self, start_dim: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self._absorb_permutation = False  # Internal: set by converter
    
    @classmethod
    def _with_absorbed_permutation(cls, start_dim: int = 0, end_dim: int = -1) -> "EncryptedFlatten":
        """Internal factory for converter to create Flatten with absorbed permutation.
        
        When permutation is absorbed into FC weights, Flatten becomes a no-op.
        This is an internal API - users should not call this directly.
        """
        instance = cls(start_dim, end_dim)
        instance._absorb_permutation = True
        return instance
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Flatten the encrypted tensor.
        
        Args:
            x: Encrypted input tensor.
            
        Returns:
            Flattened encrypted tensor.
        """
        # Check if this is CNN-packed data
        if hasattr(x, '_cnn_layout') and x._cnn_layout is not None:
            return self._forward_he_packed(x)
        
        # For encrypted tensors, flatten is just a reshape
        return x.flatten()
    
    def _forward_he_packed(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Flatten CNN-packed tensor with correct memory layout.
        
        CNN layout: (num_patches, channels) = (H*W, C)
        PyTorch expects: (C, H, W) flattened = C first, then spatial
        
        If absorb_permutation is True, the permutation is already absorbed
        into FC weights, so we just reshape (no matmul needed).
        
        For sparse layouts (from pooling), keep the full slot count since
        FC weight matrix handles the gather operation.
        """
        layout = x._cnn_layout
        num_patches = layout['num_patches']  # H * W
        num_channels = layout['patch_features']  # C
        
        # Check if sparse layout from pooling
        is_sparse = layout.get('sparse', False)
        
        if is_sparse:
            total_slots = layout['total_slots']  # Full vector size for sparse
        else:
            total_slots = num_patches * num_channels
        
        # If permutation is absorbed into FC weights, just reshape
        if self._absorb_permutation:
            x._shape = (total_slots,)
            x._cnn_layout = None
            return x
        
        # Otherwise, apply permutation matmul
        import torch
        
        total_size = num_patches * num_channels
        
        # Current: [P0C0, P0C1, P0C2, P0C3, P1C0, P1C1, ...] (patches × channels)
        # Target:  [C0P0, C0P1, ..., C0Pn, C1P0, ...] (channels × patches)
        # This is a transpose operation: (H*W, C) -> (C, H*W) -> flatten
        
        # Build permutation matrix
        perm_matrix = torch.zeros(total_size, total_size, dtype=torch.float64)
        for c in range(num_channels):
            for p in range(num_patches):
                out_idx = c * num_patches + p
                in_idx = p * num_channels + c
                perm_matrix[out_idx, in_idx] = 1.0
        
        # Apply permutation via matmul (using EncryptedTensor.matmul method)
        perm_matrix = perm_matrix.to(x._cipher.device)
        result = x.matmul(perm_matrix)  # This handles the HE matmul
        
        # Update shape and clear CNN layout
        result._shape = (total_size,)
        result._cnn_layout = None  # Clear layout - now flat vector
        return result
    
    def mult_depth(self) -> int:
        # Flatten is free (no encrypted operations)
        return 0
    
    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
