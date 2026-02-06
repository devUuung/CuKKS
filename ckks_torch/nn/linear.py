"""
EncryptedLinear - Encrypted linear (fully connected) layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedLinear(EncryptedModule):
    """Encrypted linear (fully connected) layer.
    
    Computes: y = x @ W^T + b
    
    The weight and bias are stored in plaintext. The input is encrypted,
    and the output is also encrypted.
    
    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        weight: Weight matrix of shape (out_features, in_features).
        bias: Optional bias vector of shape (out_features,).
    
    Example:
        >>> # From a trained PyTorch layer
        >>> linear = nn.Linear(128, 64)
        >>> enc_linear = EncryptedLinear.from_torch(linear)
        >>> 
        >>> # Forward pass
        >>> enc_output = enc_linear(enc_input)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights as float64 for CKKS precision
        self.weight = weight.detach().to(dtype=torch.float64, device="cpu")
        self.bias = bias.detach().to(dtype=torch.float64, device="cpu") if bias is not None else None
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)
    
    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass on encrypted input.
        
        Args:
            x: Encrypted input of shape (*, in_features).
            
        Returns:
            Encrypted output of shape (*, out_features).
        """
        if getattr(self, '_sparse_input', False):
            return self._forward_sparse(x)
        return x.matmul(self.weight, self.bias)
    
    def _forward_sparse(self, x: "EncryptedTensor") -> "EncryptedTensor":
        """Forward pass for sparse input using row-by-row dot products.
        
        For out_features outputs, compute each independently:
        y[i] = sum_j W[i,j] * x[j]
        
        Uses O(out_features * log(in_features)) operations.
        """
        out_features = self.out_features
        in_features = self.in_features
        
        outputs = []
        for row in range(out_features):
            row_weights = self.weight[row].tolist()
            masked = x.mul(row_weights)
            masked = masked.rescale()
            dot_product = self._reduce_sum(masked, in_features)
            outputs.append(dot_product)
        
        result = self._pack_outputs(outputs, x._context)
        
        if self.bias is not None:
            bias_list = [0.0] * result._cipher.size
            for i, v in enumerate(self.bias.tolist()):
                if i < len(bias_list):
                    bias_list[i] = v
            result = result.add(bias_list)
        
        return result
    
    def _reduce_sum(self, x: "EncryptedTensor", length: int) -> "EncryptedTensor":
        """Sum elements using rotation-based reduction. O(log n) rotations."""
        result = x
        step = 1
        while step < length:
            rotated = result.rotate(step)
            result = result + rotated
            step *= 2
        return result
    
    def _pack_outputs(self, outputs: list, context) -> "EncryptedTensor":
        """Pack scalar outputs (sum in slot 0) into single ciphertext."""
        out_features = len(outputs)
        if out_features == 0:
            raise ValueError("Cannot pack empty outputs list")
        
        slot_count = outputs[0]._cipher.size
        
        result = None
        for i, out in enumerate(outputs):
            if i > 0:
                shifted = out.rotate(-i)
            else:
                shifted = out
            
            mask = [0.0] * slot_count
            mask[i] = 1.0
            masked = shifted.mul(mask)
            masked = masked.rescale()
            
            if result is None:
                result = masked
            else:
                result = result + masked
        
        # result is guaranteed to be non-None here since out_features > 0
        assert result is not None
        result._shape = (out_features,)
        return result
    
    def mult_depth(self) -> int:
        """Linear layer uses 1 multiplication."""
        return 1
    
    @classmethod
    def from_torch(cls, linear: torch.nn.Linear) -> "EncryptedLinear":
        """Create from a PyTorch Linear layer.
        
        Args:
            linear: The PyTorch Linear layer to convert.
            
        Returns:
            EncryptedLinear with copied weights.
        """
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight=linear.weight.data,
            bias=linear.bias.data if linear.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
        )
    
    @classmethod
    def from_torch_cnn(
        cls, 
        linear: torch.nn.Linear, 
        cnn_layout: dict,
    ) -> "EncryptedLinear":
        """Create from a PyTorch Linear layer following CNN layers.
        
        This absorbs the Flatten permutation into the FC weights, eliminating
        the need for an expensive permutation matmul at runtime.
        
        CNN layout stores data as (num_patches, channels) = (H*W, C),
        but PyTorch FC expects (C, H, W) flattened = channels first.
        
        Instead of permuting the encrypted data, we permute the weights:
            Original: y = (x @ P) @ W^T + b = x @ (P @ W^T) + b
            Optimized: y = x @ W'^T + b where W' = W @ P^T
        
        If cnn_layout has 'sparse' = True, also handles sparse pooling output
        by building a weight matrix that reads from sparse positions.
        
        Args:
            linear: The PyTorch Linear layer to convert.
            cnn_layout: Dictionary with 'num_patches' and 'patch_features'.
                        Optionally 'sparse', 'sparse_positions', 'total_slots'.
            
        Returns:
            EncryptedLinear with permuted weights (Flatten absorbed).
        """
        num_patches = cnn_layout['num_patches']    # H * W
        num_channels = cnn_layout['patch_features'] # C
        total_size = num_patches * num_channels
        
        # Check if input will be in sparse format (from pooling without gather)
        is_sparse = cnn_layout.get('sparse', False)
        
        if is_sparse:
            # Sparse format: values are at 'sparse_positions' in a larger vector
            sparse_positions = cnn_layout['sparse_positions']
            total_slots = cnn_layout['total_slots']
            
            # Build gather+permutation matrix
            # Maps: sparse_layout -> PyTorch FC expected layout
            # 1. Gather: pick values from sparse_positions
            # 2. Permute: reorder from (H*W, C) to (C, H*W)
            
            # Combined transformation: W_final = W @ P @ G
            # Where G is gather (64 x sparse_slots) and P is permute (64 x 64)
            
            # Build gather matrix G: (total_size x total_slots)
            # G[out_idx, sparse_pos] = 1 means: output[out_idx] = input[sparse_pos]
            gather_matrix = torch.zeros(total_size, total_slots, dtype=torch.float64)
            for out_idx, sparse_pos in enumerate(sparse_positions):
                gather_matrix[out_idx, sparse_pos] = 1.0
            
            # Build permutation matrix P: (C*H*W -> H*W*C)
            perm_matrix = torch.zeros(total_size, total_size, dtype=torch.float64)
            for c in range(num_channels):
                for p in range(num_patches):
                    out_idx = c * num_patches + p
                    in_idx = p * num_channels + c
                    perm_matrix[out_idx, in_idx] = 1.0
            
            # W_final = W @ P @ G
            W_original = linear.weight.data.to(torch.float64)
            W_permuted = W_original @ perm_matrix @ gather_matrix
            
            enc_linear = cls(
                in_features=total_slots,  # Input is the full sparse vector
                out_features=linear.out_features,
                weight=W_permuted,
                bias=linear.bias.data if linear.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            )
            enc_linear._sparse_input = True
            return enc_linear
        else:
            # Dense format: standard permutation only
            perm_matrix = torch.zeros(total_size, total_size, dtype=torch.float64)
            for c in range(num_channels):
                for p in range(num_patches):
                    out_idx = c * num_patches + p
                    in_idx = p * num_channels + c
                    perm_matrix[out_idx, in_idx] = 1.0
            
            W_original = linear.weight.data.to(torch.float64)
            W_permuted = W_original @ perm_matrix  # W @ P (not P^T!)
            
            return cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                weight=W_permuted,
                bias=linear.bias.data if linear.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
            )
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
