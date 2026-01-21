"""
EncryptedTensor - A tensor wrapper for CKKS-encrypted data.

This module provides a high-level tensor abstraction that feels familiar
to PyTorch users while operating on encrypted data.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

if TYPE_CHECKING:
    from .context import CKKSInferenceContext


class EncryptedTensor:
    """A tensor containing CKKS-encrypted data.
    
    This class wraps a CKKS ciphertext and provides tensor-like operations.
    It tracks the original shape and provides arithmetic operations that
    work on encrypted data.
    
    Note:
        This is NOT a torch.Tensor subclass. Encrypted operations have
        different semantics (e.g., rescaling) that don't map cleanly
        to PyTorch's autograd.
    
    Example:
        >>> ctx = CKKSInferenceContext()
        >>> x = ctx.encrypt(torch.randn(10))
        >>> y = x * 2.0  # Encrypted scalar multiplication
        >>> z = x + y    # Encrypted addition
        >>> result = ctx.decrypt(z)
    """
    
    def __init__(
        self,
        cipher: Any,
        shape: Tuple[int, ...],
        context: Any,
        depth: int = 0,
    ):
        """Initialize an EncryptedTensor.
        
        Args:
            cipher: The underlying CKKS ciphertext.
            shape: The logical shape of the tensor.
            context: The CKKS context that created this tensor.
            depth: The current multiplicative depth (default 0).
        """
        self._cipher = cipher
        self._shape = tuple(shape)
        self._context = context
        self._depth = depth
        self._original_size: Optional[int] = None
        self._cnn_layout: Optional[Dict[str, Any]] = None  # CNN im2col layout info
        self._needs_rescale: bool = False  # Lazy rescale flag
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """The logical shape of the tensor."""
        return self._shape
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(math.prod(self._shape)) if self._shape else 1
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)
    
    @property
    def context(self) -> "CKKSInferenceContext":
        """The CKKS context associated with this tensor."""
        return self._context
    
    @property
    def metadata(self) -> dict:
        """Get ciphertext metadata (scale, level, etc.)."""
        return self._cipher.metadata
    
    @property
    def level(self) -> int:
        """Current level (remaining multiplicative depth)."""
        return self.metadata.get("level", 0)
    
    @property
    def scale(self) -> float:
        """Current scale factor."""
        return self.metadata.get("scale", 0.0)
    
    @property
    def depth(self) -> int:
        """Current multiplicative depth consumed."""
        return self._depth
    
    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------
    
    def _to_plain_list(self, other: Union[torch.Tensor, float, int, List[float]]) -> List[float]:
        """Convert a value to a list of floats for the backend."""
        if isinstance(other, (int, float)):
            return [float(other)] * self.size
        elif isinstance(other, list):
            return other
        else:
            return torch.as_tensor(other, dtype=torch.float64).reshape(-1).tolist()
    
    def add(self, other: Union["EncryptedTensor", torch.Tensor, float, int, List[float]]) -> "EncryptedTensor":
        """Add another tensor or scalar.
        
        Args:
            other: EncryptedTensor, plain tensor, or scalar to add.
            
        Returns:
            New EncryptedTensor with the sum.
        """
        if isinstance(other, EncryptedTensor):
            new_cipher = self._cipher.add(other._cipher)
            new_depth = max(self._depth, other._depth)
        else:
            plain = self._to_plain_list(other)
            new_cipher = self._cipher.add(plain)
            new_depth = self._depth
        
        return EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
    
    def sub(self, other: Union["EncryptedTensor", torch.Tensor, float, int, List[float]]) -> "EncryptedTensor":
        """Subtract another tensor or scalar.
        
        Args:
            other: EncryptedTensor, plain tensor, or scalar to subtract.
            
        Returns:
            New EncryptedTensor with the difference.
        """
        # Try direct backend sub if available (more efficient than neg + add)
        if hasattr(self._cipher, 'sub'):
            if isinstance(other, EncryptedTensor):
                new_cipher = self._cipher.sub(other._cipher)
                new_depth = max(self._depth, other._depth)
            else:
                plain = self._to_plain_list(other)
                new_cipher = self._cipher.sub(plain)
                new_depth = self._depth
            return EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
        
        # Fallback: negate and add
        if isinstance(other, EncryptedTensor):
            neg_other = other.neg()
            return self.add(neg_other)
        else:
            if isinstance(other, (int, float)):
                return self.add(-float(other))
            else:
                return self.add(-torch.as_tensor(other, dtype=torch.float64))
    
    def mul(self, other: Union["EncryptedTensor", torch.Tensor, float, int, List[float]]) -> "EncryptedTensor":
        """Multiply by another tensor or scalar.
        
        Args:
            other: EncryptedTensor, plain tensor, or scalar to multiply.
            
        Returns:
            New EncryptedTensor with the product.
            
        Note:
            Cipher-cipher multiplication increases noise and consumes levels.
            Rescale is deferred (lazy) until needed by next operation.
        """
        if isinstance(other, EncryptedTensor):
            new_cipher = self._cipher.mul(other._cipher)
            new_depth = max(self._depth, other._depth) + 1
        else:
            plain = self._to_plain_list(other)
            new_cipher = self._cipher.mul(plain)
            new_depth = self._depth + 1
        
        result = EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
        result._needs_rescale = True  # Mark for lazy rescale
        return result
    
    def neg(self) -> "EncryptedTensor":
        """Negate the tensor."""
        neg_plain = [-1.0] * self.size
        new_cipher = self._cipher.mul(neg_plain)
        return EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
    
    def div(self, divisor: float) -> "EncryptedTensor":
        """Divide by plaintext scalar.
        
        Args:
            divisor: Non-zero scalar to divide by.
            
        Returns:
            New EncryptedTensor with the quotient.
            
        Note:
            Cipher-cipher division is not supported in CKKS.
        """
        if divisor == 0:
            raise ValueError("Division by zero")
        return self.mul(1.0 / divisor)
    
    def square(self) -> "EncryptedTensor":
        """Square the tensor (element-wise)."""
        new_cipher = self._cipher.mul(self._cipher)
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth + 1)
        result._needs_rescale = True  # Mark for lazy rescale
        # Propagate CNN layout if present
        if self._cnn_layout is not None:
            result._cnn_layout = self._cnn_layout.copy()
        return result
    
    # -------------------------------------------------------------------------
    # CKKS-specific Operations
    # -------------------------------------------------------------------------
    
    def rescale(self) -> "EncryptedTensor":
        """Rescale to reduce scale and consume a level.
        
        This should typically be called after multiplication to
        manage the scale growth.
        
        Returns:
            New EncryptedTensor with reduced scale.
        """
        new_cipher = self._cipher.rescale()
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        # Propagate CNN layout if present
        if self._cnn_layout is not None:
            result._cnn_layout = self._cnn_layout.copy()
        return result
    
    def rotate(self, steps: int) -> "EncryptedTensor":
        """Rotate slots by the given number of steps.
        
        Args:
            steps: Number of positions to rotate. Positive = left, negative = right.
            
        Returns:
            New EncryptedTensor with rotated slots.
        """
        new_cipher = self._cipher.rotate(steps)
        return EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
    
    def sum_slots(self) -> "EncryptedTensor":
        """Sum all slots into the first slot.
        
        Returns:
            New EncryptedTensor with sum in first position.
        """
        new_cipher = self._cipher.sum_slots()
        return EncryptedTensor(new_cipher, (1,), self._context, self._depth)
    
    def conjugate(self) -> "EncryptedTensor":
        """Apply complex conjugation to slots."""
        new_cipher = self._cipher.conjugate()
        return EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
    
    def bootstrap(self) -> "EncryptedTensor":
        """Bootstrap to refresh the ciphertext.
        
        This is expensive but allows continuing computation when
        levels are exhausted.
        
        Returns:
            New EncryptedTensor with refreshed levels.
        """
        new_cipher = self._cipher.bootstrap()
        return EncryptedTensor(new_cipher, self._shape, self._context, 0)
    
    def maybe_bootstrap(self, context: "CKKSInferenceContext") -> "EncryptedTensor":
        if context.auto_bootstrap and self._depth >= context.bootstrap_threshold:
            import logging
            logging.info(f"Auto-bootstrapping: depth={self._depth}")
            return self.bootstrap()
        return self
    
    # -------------------------------------------------------------------------
    # Matrix Operations
    # -------------------------------------------------------------------------
    
    def matmul(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> "EncryptedTensor":
        # Auto-rescale if previous operation left pending rescale
        tensor = self
        if self._needs_rescale:
            tensor = self.rescale()
        
        weight_2d = weight.detach().to(dtype=torch.float64, device="cpu")
        if weight_2d.ndim == 1:
            weight_2d = weight_2d.unsqueeze(0)
        
        use_bsgs = getattr(tensor._context, 'use_bsgs', False)
        
        if use_bsgs:
            weight_list = weight_2d.tolist()
            max_dim = getattr(tensor._context, '_max_rotation_dim', None)
            if max_dim is None:
                max_dim = weight_2d.shape[1]
            bsgs_n1 = max(1, int(math.ceil(math.sqrt(max_dim))))
            bsgs_n2 = (weight_2d.shape[1] + bsgs_n1 - 1) // bsgs_n1
            new_cipher = tensor._cipher.matmul_bsgs(weight_list, bsgs_n1, bsgs_n2)
            result = EncryptedTensor(new_cipher, (weight_2d.shape[0],), tensor._context, tensor._depth + 1)
        else:
            weight_list = weight_2d.tolist()
            new_cipher = tensor._cipher.matmul_dense(weight_list)
            result = EncryptedTensor(new_cipher, (weight_2d.shape[0],), tensor._context, tensor._depth + 1)
        
        result = result.rescale()
        
        if bias is not None:
            bias_flat = bias.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
            # Use result's slot count, not input's - output size may differ after matmul
            slot_count = result._cipher.size
            bias_padded = [0.0] * slot_count
            for i, v in enumerate(bias_flat.tolist()):
                bias_padded[i] = v
            result = result.add(bias_padded)
        
        return result
    
    def _matmul_bsgs(self, weight: torch.Tensor) -> "EncryptedTensor":
        """BSGS diagonal matmul: O(√n) rotations instead of O(n)."""
        out_features, in_features = weight.shape
        slot_count = self._cipher.size
        
        max_dim = getattr(self._context, '_max_rotation_dim', in_features)
        bsgs_n1 = max(1, int(math.ceil(math.sqrt(max_dim))))
        bsgs_n2 = (in_features + bsgs_n1 - 1) // bsgs_n1
        
        baby_ciphers = []
        for j in range(min(bsgs_n1, in_features)):
            if j == 0:
                baby_ciphers.append(self._cipher)
            else:
                baby_ciphers.append(self._cipher.rotate(j))
        
        indices = torch.arange(slot_count)
        
        acc = None
        for k in range(bsgs_n2):
            giant_step = k * bsgs_n1
            rows_base = (indices - giant_step) % out_features
            
            block = None
            for j in range(len(baby_ciphers)):
                d = giant_step + j
                if d >= in_features:
                    break
                cols = (indices + j) % in_features
                diag = weight[rows_base, cols].tolist()
                term = baby_ciphers[j].mul(diag)
                block = term if block is None else block.add(term)
            
            if block is None:
                continue
            if k > 0:
                block = block.rotate(giant_step)
            acc = block if acc is None else acc.add(block)
        
        return EncryptedTensor(acc, (out_features,), self._context, self._depth + 1)
    
    def _compute_bsgs_diagonal(self, weight: torch.Tensor, giant_step: int, baby_step: int, slot_count: int) -> List[float]:
        """Compute BSGS diagonal: W[(i - giant_step) % n, (i + baby_step) % m]."""
        out_features, in_features = weight.shape
        indices = torch.arange(slot_count)
        rows = (indices - giant_step) % out_features
        cols = (indices + baby_step) % in_features
        return weight[rows, cols].tolist()
    
    def poly_eval(self, coeffs: Sequence[float]) -> "EncryptedTensor":
        """Evaluate a polynomial on the encrypted values.
        
        Args:
            coeffs: Polynomial coefficients [a0, a1, a2, ...] for
                    a0 + a1*x + a2*x^2 + ...
                    
        Returns:
            New EncryptedTensor with polynomial evaluated element-wise.
        """
        new_cipher = self._cipher.poly_eval(list(coeffs))
        poly_depth = len(coeffs) - 1 if len(coeffs) > 1 else 0
        return EncryptedTensor(new_cipher, self._shape, self._context, self._depth + poly_depth)

    def inv_sqrt(
        self, domain: tuple[float, float] = (0.1, 100.0)
    ) -> "EncryptedTensor":
        """Compute 1/sqrt(x) using CryptoInvSqrt.

        Uses Chebyshev polynomial approximation followed by Newton-Raphson
        refinement to approximate the inverse square root.

        Args:
            domain: Input domain (a, b). Must be (0.1, 100.0) in v1.

        Returns:
            EncryptedTensor with 1/sqrt(x) approximation.

        Raises:
            NotImplementedError: If domain != (0.1, 100.0)
            RuntimeError: If enable_bootstrap=False

        Warning:
            v1 ONLY supports Mock backend.
            OpenFHE is NOT supported in v1.

        Note:
            Uses 2 bootstrap operations internally.

        Reference:
            Choi, H. (2025). PP-STAT. CIKM'25.
        """
        from ckks_torch.stats.crypto_inv_sqrt import crypto_inv_sqrt

        return crypto_inv_sqrt(self, domain=domain)

    def sqrt(
        self, domain: tuple[float, float] = (0.1, 100.0)
    ) -> "EncryptedTensor":
        """Compute sqrt(x) = x * inv_sqrt(x).

        Uses the identity sqrt(x) = x * (1/sqrt(x)) to compute the
        square root from the inverse square root.

        Args:
            domain: Input domain (a, b). Must be (0.1, 100.0) in v1.

        Returns:
            EncryptedTensor with sqrt(x) approximation.

        Raises:
            NotImplementedError: If domain != (0.1, 100.0)
            RuntimeError: If enable_bootstrap=False

        Warning:
            v1 ONLY supports Mock backend.
            OpenFHE is NOT supported in v1.

        Note:
            Error compounds from inv_sqrt; expect MRE ~1e-2.
            Uses 2 bootstrap operations internally (via inv_sqrt).

        Reference:
            Choi, H. (2025). PP-STAT. CIKM'25.
        """
        inv_sqrt_x = self.inv_sqrt(domain=domain)
        return self.mul(inv_sqrt_x)
    
    # -------------------------------------------------------------------------
    # CNN Operations
    # -------------------------------------------------------------------------
    
    def conv2d(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        output_shape: Tuple[int, ...],
    ) -> "EncryptedTensor":
        """Apply 2D convolution using im2col method.
        
        This is a high-level operation that:
        1. Unfolds input to patches (im2col)
        2. Performs matrix multiplication
        3. Reshapes to output spatial dimensions
        
        For mock backend, this decrypts, computes, and re-encrypts.
        For real HE backend, this would use optimized SIMD operations.
        
        Args:
            weight: Convolution weight matrix (out_channels, in_channels * kH * kW).
            bias: Optional bias (out_channels,).
            kernel_size: Kernel size (kH, kW).
            stride: Stride (sH, sW).
            padding: Padding (pH, pW).
            output_shape: Expected output shape (batch, out_channels, out_h, out_w).
            
        Returns:
            Encrypted output tensor with output_shape.
        """
        import torch.nn.functional as F
        
        # Decrypt for computation (mock backend approach)
        plaintext = self._context.decrypt(self)
        
        # Ensure 4D
        original_ndim = plaintext.ndim
        if plaintext.ndim == 3:
            plaintext = plaintext.unsqueeze(0)
        
        # Apply padding if needed
        if padding != (0, 0):
            plaintext = F.pad(plaintext, (padding[1], padding[1], padding[0], padding[0]))
        
        # Unfold: (N, C, H, W) -> (N, C*kH*kW, num_patches)
        patches = F.unfold(plaintext.to(torch.float64), kernel_size, stride=stride)
        
        # Transpose: (N, C*kH*kW, num_patches) -> (N, num_patches, C*kH*kW)
        patches = patches.transpose(1, 2)
        
        # Squeeze batch if single sample: (num_patches, C*kH*kW)
        patches = patches.squeeze(0)
        
        # Matrix multiply: (num_patches, patch_features) @ (out_channels, patch_features).T
        # = (num_patches, out_channels)
        # Ensure all tensors are on CPU for compatibility
        patches = patches.cpu()
        weight_2d = weight.to(dtype=torch.float64, device="cpu")
        result = torch.mm(patches, weight_2d.T)
        
        # Add bias if present
        if bias is not None:
            result = result + bias.to(dtype=torch.float64, device="cpu")
        
        # Reshape to output: (num_patches, out_channels) -> (batch, out_channels, out_h, out_w)
        batch_size = output_shape[0]
        out_channels = output_shape[1]
        out_h = output_shape[2]
        out_w = output_shape[3]
        
        # result is (num_patches, out_channels), need (batch, C, H, W)
        result = result.T.reshape(batch_size, out_channels, out_h, out_w)
        
        # Re-encrypt
        return self._context.encrypt(result)
    
    def matmul_2d(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> "EncryptedTensor":
        """Matrix multiply for 2D input (num_patches, features).
        
        Computes: input @ weight.T + bias
        
        For mock backend, decrypts and re-encrypts.
        
        Args:
            weight: Weight matrix (out_features, in_features).
            bias: Optional bias (out_features,).
            
        Returns:
            Encrypted result of shape (num_patches, out_features).
        """
        # Decrypt for computation
        plaintext = self._context.decrypt(self)
        
        # Matrix multiply
        weight_2d = weight.to(dtype=torch.float64)
        result = torch.mm(plaintext.to(torch.float64), weight_2d.T)
        
        # Add bias if present
        if bias is not None:
            result = result + bias.to(dtype=torch.float64)
        
        # Re-encrypt
        return self._context.encrypt(result)
    
    def avgpool2d(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int] = (0, 0),
    ) -> "EncryptedTensor":
        """Apply 2D average pooling.
        
        For mock backend, decrypts, applies pooling, and re-encrypts.
        
        Args:
            kernel_size: Pooling kernel size (kH, kW).
            stride: Pooling stride (sH, sW).
            padding: Pooling padding (pH, pW).
            
        Returns:
            Encrypted output after average pooling.
        """
        import torch.nn.functional as F
        
        # Decrypt for computation
        plaintext = self._context.decrypt(self)
        
        # Ensure 4D
        original_ndim = plaintext.ndim
        if plaintext.ndim == 3:
            plaintext = plaintext.unsqueeze(0)
        
        # Move to CPU for computation
        plaintext = plaintext.cpu().to(torch.float64)
        
        # Apply average pooling
        result = F.avg_pool2d(plaintext, kernel_size, stride=stride, padding=padding)
        
        # Re-encrypt
        return self._context.encrypt(result)
    
    # -------------------------------------------------------------------------
    # Decryption
    # -------------------------------------------------------------------------
    
    def decrypt(self, shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Decrypt this tensor.
        
        Args:
            shape: Optional shape for the output.
            
        Returns:
            Decrypted PyTorch tensor.
        """
        return self._context.decrypt(self, shape=shape)
    
    # -------------------------------------------------------------------------
    # Python Operators
    # -------------------------------------------------------------------------
    
    def __add__(self, other: Any) -> "EncryptedTensor":
        return self.add(other)
    
    def __radd__(self, other: Any) -> "EncryptedTensor":
        return self.add(other)
    
    def __sub__(self, other: Any) -> "EncryptedTensor":
        return self.sub(other)
    
    def __rsub__(self, other: Any) -> "EncryptedTensor":
        return self.neg().add(other)
    
    def __mul__(self, other: Any) -> "EncryptedTensor":
        return self.mul(other)
    
    def __rmul__(self, other: Any) -> "EncryptedTensor":
        return self.mul(other)
    
    def __neg__(self) -> "EncryptedTensor":
        return self.neg()
    
    def __matmul__(self, other: Any) -> "EncryptedTensor":
        if isinstance(other, torch.Tensor):
            return self.matmul(other)
        raise TypeError(f"matmul not supported with {type(other)}")
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def view(self, *shape: int) -> "EncryptedTensor":
        """Reshape the tensor (logical only, no data movement).
        
        Args:
            *shape: New shape dimensions.
            
        Returns:
            New EncryptedTensor with updated shape.
        """
        new_size = math.prod(shape)
        if new_size != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} to shape {shape}"
            )
        return EncryptedTensor(self._cipher, tuple(shape), self._context, self._depth)
    
    def reshape(self, shape: Union[int, Tuple[int, ...]]) -> "EncryptedTensor":
        if isinstance(shape, int):
            return self.view(shape)
        return self.view(*shape)
    
    def flatten(self) -> "EncryptedTensor":
        return self.view(self.size)
    
    def unfold(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> "EncryptedTensor":
        import torch.nn.functional as F
        
        original_shape = self._shape
        if len(original_shape) == 3:
            pass
        elif len(original_shape) == 4:
            pass
        else:
            raise ValueError(f"unfold requires 3D or 4D input, got {len(original_shape)}D")
        
        plaintext = self._context.decrypt(self)
        if len(original_shape) == 3:
            plaintext = plaintext.unsqueeze(0)
        
        if padding != (0, 0):
            plaintext = F.pad(plaintext, (padding[1], padding[1], padding[0], padding[0]))
        
        patches = F.unfold(plaintext.to(torch.float64), kernel_size, stride=stride)
        patches = patches.transpose(1, 2)
        patches = patches.squeeze(0)
        
        return self._context.encrypt(patches)
    
    def clone(self) -> "EncryptedTensor":
        """Create a copy of this tensor.
        
        Note: This creates a new Python wrapper but the underlying
        ciphertext may be shared (copy-on-write semantics).
        """
        return EncryptedTensor(self._cipher, self._shape, self._context, self._depth)
    
    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"EncryptedTensor(shape={self._shape}, "
            f"depth={self._depth}, "
            f"level={meta.get('level')}, "
            f"scale={meta.get('scale', 0):.2e})"
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the encrypted tensor to a file.
        
        Serializes the tensor metadata and underlying cipher data.
        The cipher serialization depends on the backend implementation.
        
        Args:
            path: Path to save the tensor.
        """
        cipher_data = None
        if hasattr(self._cipher, 'data'):
            cipher_data = self._cipher.data
        
        tensor_data = {
            "shape": self._shape,
            "depth": self._depth,
            "cipher_data": cipher_data,
        }
        with open(path, "wb") as f:
            pickle.dump(tensor_data, f)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        context: "CKKSInferenceContext",
    ) -> "EncryptedTensor":
        """Load an encrypted tensor from a file.
        
        Args:
            path: Path to the saved tensor.
            context: The CKKS context to associate with the loaded tensor.
            
        Returns:
            The loaded EncryptedTensor.
        """
        with open(path, "rb") as f:
            tensor_data = pickle.load(f)
        
        shape = tensor_data["shape"]
        depth = tensor_data["depth"]
        cipher_data = tensor_data["cipher_data"]
        
        if cipher_data is not None:
            dummy_tensor = torch.zeros(math.prod(shape) if shape else 1)
            cipher = context._ctx.encrypt(dummy_tensor)
            setattr(cipher, 'data', cipher_data)
            setattr(cipher, 'shape', shape)
            setattr(cipher, 'size', int(math.prod(shape)) if shape else 1)
        else:
            raise ValueError("Cannot load tensor: cipher_data not available")
        
        return cls(cipher, shape, context, depth)
