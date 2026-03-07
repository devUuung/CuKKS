"""
EncryptedTensor - A tensor wrapper for CKKS-encrypted data.

This module provides a high-level tensor abstraction that feels familiar
to PyTorch users while operating on encrypted data.
"""

from __future__ import annotations

import copy
import math
import pickle
import warnings
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
    
    Packed-Batch Layout Contract
    ----------------------------
    When ``_packed_batch=True``, the ciphertext slots contain multiple samples
    packed contiguously::
    
        [s0[0..K-1], s1[0..K-1], ..., s(B-1)[0..K-1], padding...]
    
    where *B* = ``_batch_size`` and *K* = ``_slots_per_sample``.
    
    All element-wise operations (add, sub, mul, neg, poly_eval),
    slot rotations (rotate), rescale, bootstrap, and matmul preserve
    this metadata automatically.  The I/O contract is::
    
        # Encryption
        enc_batch = ctx.encrypt_batch(samples)         # _packed_batch=True
    
        # Model forward (metadata preserved through every op)
        enc_out = enc_model(enc_batch)                  # _packed_batch=True
    
        # Decryption
        results = ctx.decrypt_batch(enc_out)            # list of B tensors
    
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
        self._packed_batch: bool = False
        self._batch_size: Optional[int] = None
        self._slots_per_sample: Optional[int] = None
        self._needs_rescale: bool = False  # Lazy rescale flag
    
    def _propagate_batch_meta(self, result: "EncryptedTensor") -> "EncryptedTensor":
        """Copy packed-batch metadata from *self* to *result*.
        
        Also copies ``_cnn_layout`` when present.  Every method that constructs
        a new :class:`EncryptedTensor` from an existing one should call this.
        """
        if self._packed_batch:
            result._packed_batch = True
            result._batch_size = self._batch_size
            result._slots_per_sample = self._slots_per_sample
        if self._cnn_layout is not None:
            result._cnn_layout = copy.deepcopy(self._cnn_layout)
        return result
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
        if isinstance(other, (int, float)):
            return [float(other)] * self._cipher.size
        elif isinstance(other, list):
            if len(other) < self._cipher.size:
                # Cyclically replicate to fill all slots (preserves BSGS invariant)
                slot_count = self._cipher.size
                n = len(other)
                if n == 0:
                    return [0.0] * slot_count
                return [other[i % n] for i in range(slot_count)]
            return other
        else:
            flat = torch.as_tensor(other, dtype=torch.float64).reshape(-1).tolist()
            slot_count = self._cipher.size
            n = len(flat)
            if n < slot_count:
                # Cyclically replicate to fill all slots (preserves BSGS invariant)
                if n == 0:
                    return [0.0] * slot_count
                return [flat[i % n] for i in range(slot_count)]
            return flat
    
    def add(self, other: Union["EncryptedTensor", torch.Tensor, float, int, List[float]]) -> "EncryptedTensor":
        """Add another tensor or scalar.
        
        Args:
            other: EncryptedTensor, plain tensor, or scalar to add.
            
        Returns:
            New EncryptedTensor with the sum.
        """
        if isinstance(other, EncryptedTensor):
            if self.size != other.size:
                raise ValueError(
                    f"Size mismatch in add: {self._shape} ({self.size} elements) vs "
                    f"{other._shape} ({other.size} elements)"
                )
            left = self
            right = other
            if left._needs_rescale and not right._needs_rescale:
                left = left.rescale()
            elif right._needs_rescale and not left._needs_rescale:
                right = right.rescale()
            new_cipher = left._cipher.add(right._cipher)
            new_depth = max(left._depth, right._depth)
            needs_rescale = left._needs_rescale and right._needs_rescale
            result = EncryptedTensor(new_cipher, left._shape, left._context, new_depth)
            result._needs_rescale = needs_rescale
            return left._propagate_batch_meta(result)
        else:
            plain = self._to_plain_list(other)
            new_cipher = self._cipher.add(plain)
            new_depth = self._depth
            needs_rescale = self._needs_rescale
            result = EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
            result._needs_rescale = needs_rescale
            return self._propagate_batch_meta(result)
    
    def sub(self, other: Union["EncryptedTensor", torch.Tensor, float, int, List[float]]) -> "EncryptedTensor":
        """Subtract another tensor or scalar.
        
        Args:
            other: EncryptedTensor, plain tensor, or scalar to subtract.
            
        Returns:
            New EncryptedTensor with the difference.
        """
        if hasattr(self._cipher, 'sub'):
            if isinstance(other, EncryptedTensor):
                if self.size != other.size:
                    raise ValueError(
                        f"Size mismatch in sub: {self._shape} ({self.size} elements) vs "
                        f"{other._shape} ({other.size} elements)"
                    )
                left = self
                right = other
                if left._needs_rescale and not right._needs_rescale:
                    left = left.rescale()
                elif right._needs_rescale and not left._needs_rescale:
                    right = right.rescale()
                new_cipher = left._cipher.sub(right._cipher)
                new_depth = max(left._depth, right._depth)
                needs_rescale = left._needs_rescale and right._needs_rescale
                result = EncryptedTensor(new_cipher, left._shape, left._context, new_depth)
                result._needs_rescale = needs_rescale
                return left._propagate_batch_meta(result)
            else:
                plain = self._to_plain_list(other)
                new_cipher = self._cipher.sub(plain)
                new_depth = self._depth
                needs_rescale = self._needs_rescale
                result = EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
                result._needs_rescale = needs_rescale
                return self._propagate_batch_meta(result)
        
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
            if self.size != other.size:
                raise ValueError(
                    f"Size mismatch in mul: {self._shape} ({self.size} elements) vs "
                    f"{other._shape} ({other.size} elements)"
                )
            new_cipher = self._cipher.mul(other._cipher)
            new_depth = max(self._depth, other._depth) + 1
        else:
            plain = self._to_plain_list(other)
            new_cipher = self._cipher.mul(plain)
            new_depth = self._depth + 1
        
        result = EncryptedTensor(new_cipher, self._shape, self._context, new_depth)
        result._needs_rescale = True
        return self._propagate_batch_meta(result)
    
    def neg(self) -> "EncryptedTensor":
        """Negate the tensor (free operation - doesn't consume multiplicative depth)."""
        if hasattr(self._cipher, 'neg'):
            new_cipher = self._cipher.neg()
            needs_rescale = self._needs_rescale
        else:
            neg_plain = [-1.0] * self._cipher.size
            new_cipher = self._cipher.mul(neg_plain)
            needs_rescale = True
        
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        result._needs_rescale = needs_rescale
        return self._propagate_batch_meta(result)
    
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
        """Square the tensor (element-wise).
        
        Uses dedicated GPU square operation when available, which is more
        efficient and stable than cipher-cipher multiplication with self.
        """
        # Use dedicated square() if available (GPU-optimized path)
        if hasattr(self._cipher, 'square'):
            new_cipher = self._cipher.square()
        else:
            # Fallback for backends without dedicated square
            new_cipher = self._cipher.mul(self._cipher)
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth + 1)
        result._needs_rescale = True  # Mark for lazy rescale
        return self._propagate_batch_meta(result)
    
    # -------------------------------------------------------------------------
    # CKKS-specific Operations
    # -------------------------------------------------------------------------
    
    def rescale(self) -> "EncryptedTensor":
        """Rescale to reduce scale and consume a level.
        
        This should typically be called after multiplication to
        manage the scale growth.
        
        Under FLEXIBLEAUTO scaling, some operations (e.g. CPU EvalMult,
        EvalPoly) auto-rescale internally. Calling rescale() again would
        waste a level. We detect this via noiseScaleDeg metadata: a value
        of 1 means the ciphertext is already at nominal scale.
        
        Returns:
            New EncryptedTensor with reduced scale.
        """
        # Check if FLEXIBLEAUTO already rescaled this ciphertext.
        # noiseScaleDeg == 1 means scale is already at the nominal level;
        # calling ModReduce/Rescale again would wastefully consume a level.
        meta = self.metadata
        noise_scale_deg = meta.get("noise_scale", None)
        if noise_scale_deg is not None and noise_scale_deg <= 1:
            # Already at nominal scale — skip the redundant rescale.
            result = EncryptedTensor(self._cipher, self._shape, self._context, self._depth)
            result._needs_rescale = False
            return self._propagate_batch_meta(result)

        new_cipher = self._cipher.rescale()
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        result._needs_rescale = False
        return self._propagate_batch_meta(result)

    def _ensure_rescaled(self) -> "EncryptedTensor":
        if self._needs_rescale:
            return self.rescale()
        return self
    
    def rotate(self, steps: int) -> "EncryptedTensor":
        """Rotate slots by the given number of steps.
        
        Args:
            steps: Number of positions to rotate. Positive = left, negative = right.
            
        Returns:
            New EncryptedTensor with rotated slots.
        """
        new_cipher = self._cipher.rotate(steps)
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        return self._propagate_batch_meta(result)
    
    def sum_slots(self) -> "EncryptedTensor":
        """Sum all slots into the first slot.
        
        Returns:
            New EncryptedTensor with sum in first position.
        """
        new_cipher = self._cipher.sum_slots()
        result = EncryptedTensor(new_cipher, (1,), self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        return self._propagate_batch_meta(result)
    
    def sum_and_broadcast(self, n: int) -> "EncryptedTensor":
        """Sum first n slots and replicate the result across all n positions.
        
        Uses log(n) rotation-and-add operations. In real CKKS, this naturally
        fills all slots with the sum. Required for LayerNorm and Attention
        operations that need reduction followed by element-wise operations.
        
        Args:
            n: Number of active elements to sum and broadcast.
            
        Returns:
            EncryptedTensor of same shape with sum replicated in first n slots.
        """
        new_cipher = self._cipher.sum_and_broadcast(n)
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        return self._propagate_batch_meta(result)
    
    def conjugate(self) -> "EncryptedTensor":
        """Apply complex conjugation to slots."""
        new_cipher = self._cipher.conjugate()
        result = EncryptedTensor(new_cipher, self._shape, self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        return self._propagate_batch_meta(result)
    
    def bootstrap(self) -> "EncryptedTensor":
        """Bootstrap to refresh the ciphertext.
        
        This is expensive but allows continuing computation when
        levels are exhausted.
        
        Returns:
            New EncryptedTensor with refreshed levels.
        """
        new_cipher = self._cipher.bootstrap()
        result = EncryptedTensor(new_cipher, self._shape, self._context, 0)
        return self._propagate_batch_meta(result)
    
    def maybe_bootstrap(
        self, context: Optional["CKKSInferenceContext"] = None
    ) -> "EncryptedTensor":
        """Bootstrap if auto_bootstrap is enabled and depth exceeds threshold.

        Args:
            context: CKKS context to check settings. If *None*, uses
                ``self._context`` (set during encryption).

        Returns:
            A bootstrapped tensor if threshold was exceeded, else *self*.
        """
        ctx = context if context is not None else self._context
        if ctx is None:
            return self
        if getattr(ctx, "auto_bootstrap", False) and self._depth >= getattr(
            ctx, "bootstrap_threshold", 2
        ):
            import logging
            logging.info(f"Auto-bootstrapping: depth={self._depth}")
            return self.bootstrap()
        return self
    
    # -------------------------------------------------------------------------
    # Matrix Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _rotate_plain_slots(values: List[float], steps: int) -> List[float]:
        n = len(values)
        if n == 0:
            return values
        steps = steps % n
        if steps == 0:
            return values
        return values[steps:] + values[:steps]

    @staticmethod
    def _build_packed_diag_split(
        weight_rows: List[List[float]],
        diagonal: int,
        *,
        out_features: int,
        slots_per_sample: int,
        batch_size: int,
        slot_count: int,
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        split = slots_per_sample - diagonal
        main_block = [0.0] * slots_per_sample
        wrap_block = [0.0] * slots_per_sample
        has_main = False
        has_wrap = False

        for row in range(out_features):
            value = float(weight_rows[row][(row + diagonal) % slots_per_sample])
            if abs(value) <= 1e-30:
                continue
            if row < split:
                main_block[row] = value
                has_main = True
            else:
                wrap_block[row] = value
                has_wrap = True

        main_plain = None
        if has_main:
            main_plain = [0.0] * slot_count
            for block in range(batch_size):
                start = block * slots_per_sample
                end = start + slots_per_sample
                main_plain[start:end] = main_block

        wrap_plain = None
        if has_wrap:
            wrap_plain = [0.0] * slot_count
            for block in range(batch_size):
                start = block * slots_per_sample
                end = start + slots_per_sample
                wrap_plain[start:end] = wrap_block

        return main_plain, wrap_plain

    def matmul_packed(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        weight_list: list | None = None,
        bias_list: list | None = None,
        weight_hash: int = 0,
        diag_nonzero: list | None = None,
    ) -> "EncryptedTensor":
        del weight_hash

        tensor = self
        if tensor._needs_rescale:
            tensor = tensor.rescale()

        if not tensor._packed_batch:
            raise ValueError("matmul_packed requires packed-batch input")

        if tensor._slots_per_sample is None:
            raise ValueError("Packed-batch tensor is missing slots_per_sample metadata")

        if tensor._batch_size is None:
            if len(tensor._shape) < 1:
                raise ValueError("Cannot infer packed batch size from tensor shape")
            batch_size = tensor._shape[0]
        else:
            batch_size = tensor._batch_size

        if batch_size <= 1:
            raise ValueError("matmul_packed requires batch_size > 1")

        slots_per_sample = tensor._slots_per_sample
        total_slots = batch_size * slots_per_sample
        slot_count = tensor._cipher.size
        if total_slots > slot_count:
            raise ValueError(
                f"Packed tensor uses {total_slots} active slots but ciphertext has only {slot_count}"
            )

        weight_2d = weight.detach().to(dtype=torch.float64, device="cpu")
        if weight_2d.ndim == 1:
            weight_2d = weight_2d.unsqueeze(0)
        out_features, in_features = weight_2d.shape

        if in_features != slots_per_sample:
            raise ValueError(
                "Packed matmul requires weight.in_features == slots_per_sample. "
                f"Got in_features={in_features}, slots_per_sample={slots_per_sample}."
            )
        if out_features > slots_per_sample:
            raise ValueError(
                "Packed matmul currently requires out_features <= slots_per_sample. "
                f"Got out_features={out_features}, slots_per_sample={slots_per_sample}."
            )

        x_flat = tensor.view(total_slots)
        weight_rows: List[List[float]] = (
            weight_list if weight_list is not None else weight_2d.tolist()
        )

        active_diags: list | None = None
        if diag_nonzero is not None and len(diag_nonzero) == slots_per_sample:
            active_diags = diag_nonzero

        use_bsgs = getattr(tensor._context, "use_bsgs", False)
        if use_bsgs:
            max_dim = getattr(tensor._context, "_max_rotation_dim", None)
            bsgs_dim = max_dim if max_dim else slots_per_sample
            bsgs_n1 = max(1, int(math.ceil(math.sqrt(bsgs_dim))))
        else:
            bsgs_n1 = 1
        bsgs_n2 = (slots_per_sample + bsgs_n1 - 1) // bsgs_n1
        baby_count = min(slots_per_sample, bsgs_n1)

        babies: List[EncryptedTensor] = [x_flat]
        for step in range(1, baby_count):
            babies.append(x_flat.rotate(step))

        accumulator: Optional[EncryptedTensor] = None

        for giant_idx in range(bsgs_n2):
            giant_step = giant_idx * bsgs_n1
            block_main: Optional[EncryptedTensor] = None
            block_wrap: Optional[EncryptedTensor] = None

            for baby_step, baby_cipher in enumerate(babies):
                diagonal = giant_step + baby_step
                if diagonal >= slots_per_sample:
                    break
                if active_diags is not None and not active_diags[diagonal]:
                    continue

                plain_main, plain_wrap = self._build_packed_diag_split(
                    weight_rows,
                    diagonal,
                    out_features=out_features,
                    slots_per_sample=slots_per_sample,
                    batch_size=batch_size,
                    slot_count=slot_count,
                )

                if plain_main is not None:
                    shifted_main = (
                        plain_main
                        if giant_step == 0
                        else self._rotate_plain_slots(plain_main, -giant_step)
                    )
                    main_term = baby_cipher.mul(shifted_main).rescale()
                    block_main = main_term if block_main is None else block_main.add(main_term)

                if plain_wrap is not None:
                    wrap_shift = slots_per_sample - giant_step
                    shifted_wrap = self._rotate_plain_slots(plain_wrap, wrap_shift)
                    wrap_term = baby_cipher.mul(shifted_wrap).rescale()
                    block_wrap = wrap_term if block_wrap is None else block_wrap.add(wrap_term)

            block_total: Optional[EncryptedTensor] = None
            if block_main is not None:
                if giant_step != 0:
                    block_main = block_main.rotate(giant_step)
                block_total = block_main

            if block_wrap is not None:
                if giant_step != 0:
                    block_wrap = block_wrap.rotate(giant_step)
                block_wrap = block_wrap.rotate(-slots_per_sample)
                block_total = block_wrap if block_total is None else block_total.add(block_wrap)

            if block_total is None:
                continue

            accumulator = block_total if accumulator is None else accumulator.add(block_total)

        if accumulator is None:
            raise RuntimeError("Packed matmul failed: all diagonals are zero")

        if bias is not None:
            bias_vals = (
                bias_list
                if bias_list is not None
                else bias.detach().to(dtype=torch.float64, device="cpu").reshape(-1).tolist()
            )
            if len(bias_vals) < out_features:
                raise ValueError(
                    f"Bias length {len(bias_vals)} is smaller than out_features={out_features}"
                )

            bias_plain = [0.0] * slot_count
            for block in range(batch_size):
                block_start = block * slots_per_sample
                for row in range(out_features):
                    idx = block_start + row
                    if idx < slot_count:
                        bias_plain[idx] = float(bias_vals[row])
            accumulator = accumulator.add(bias_plain)

        tensor._propagate_batch_meta(accumulator)
        accumulator._shape = (batch_size, slots_per_sample)
        return accumulator
    
    def matmul(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        weight_list: list | None = None,
        bias_list: list | None = None,
        weight_hash: int = 0,
        diag_nonzero: list | None = None,
    ) -> "EncryptedTensor":
        # Auto-rescale if previous operation left pending rescale
        tensor = self
        if self._needs_rescale:
            tensor = self.rescale()
        
        weight_2d = weight.detach().to(dtype=torch.float64, device="cpu")
        if weight_2d.ndim == 1:
            weight_2d = weight_2d.unsqueeze(0)

        if tensor._packed_batch and (tensor._batch_size or 0) > 1:
            return tensor.matmul_packed(
                weight_2d,
                bias,
                weight_list=weight_list,
                bias_list=bias_list,
                weight_hash=weight_hash,
                diag_nonzero=diag_nonzero,
            )

        use_bsgs = getattr(tensor._context, 'use_bsgs', False)
        
        if use_bsgs:
            weight_list = weight_list if weight_list is not None else weight_2d.tolist()
            max_dim = getattr(tensor._context, '_max_rotation_dim', None)
            in_features = weight_2d.shape[1]
            # IMPORTANT: bsgs_n1 must match the n1 used when generating rotation keys
            # to ensure the required rotations (baby steps 0..n1-1, giant steps n1, 2*n1, ...)
            # are available. Use max_dim for n1 calculation, not actual in_features.
            bsgs_dim = max_dim if max_dim else in_features
            bsgs_n1 = max(1, int(math.ceil(math.sqrt(bsgs_dim))))
            bsgs_n2 = (in_features + bsgs_n1 - 1) // bsgs_n1
            new_cipher = tensor._cipher.matmul_bsgs(weight_list, bsgs_n1, bsgs_n2,
                                                    weight_hash=weight_hash,
                                                    diag_nonzero=diag_nonzero)
            result = EncryptedTensor(new_cipher, (weight_2d.shape[0],), tensor._context, tensor._depth + 1)
            tensor._propagate_batch_meta(result)
        else:
            w_list = weight_list if weight_list is not None else weight_2d.tolist()
            new_cipher = tensor._cipher.matmul_dense(w_list)
            result = EncryptedTensor(new_cipher, (weight_2d.shape[0],), tensor._context, tensor._depth + 1)
            tensor._propagate_batch_meta(result)
        
        result = result.rescale()
        
        if bias is not None:
            if bias_list is not None:
                result = result.add(bias_list)
            else:
                bias_flat = bias.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
                result = result.add(bias_flat.tolist())
        
        return result
    
    def poly_eval(self, coeffs: Sequence[float]) -> "EncryptedTensor":
        """Evaluate a polynomial on the encrypted values.
        
        Args:
            coeffs: Polynomial coefficients [a0, a1, a2, ...] for
                    a0 + a1*x + a2*x^2 + ...
                    
        Returns:
            New EncryptedTensor with polynomial evaluated element-wise.
        """
        tensor = self._ensure_rescaled()
        coeff_list = list(coeffs)
        if not coeff_list:
            raise ValueError("coeffs must not be empty")

        # GPU backend falls back to CPU for general EvalPoly.
        # Keep sparse degree-4 (ReLU-like) on backend path; use Horner otherwise.
        is_sparse_degree4 = len(coeff_list) == 5 and abs(coeff_list[3]) <= 1e-12
        use_gpu_horner = (
            getattr(tensor._context, "_enable_gpu", False)
            and len(coeff_list) > 3
            and not is_sparse_degree4
        )
        if use_gpu_horner:
            result = tensor.mul(coeff_list[-1]).rescale()
            for i in range(len(coeff_list) - 2, -1, -1):
                result = result.add(coeff_list[i])
                if i > 0:
                    result = result.mul(tensor).rescale()
            tensor._propagate_batch_meta(result)
            return result

        new_cipher = tensor._cipher.poly_eval(coeff_list)
        degree = len(coeffs) - 1 if len(coeffs) > 1 else 0
        poly_depth = max(1, math.ceil(math.log2(degree + 1))) if degree > 0 else 0
        result = EncryptedTensor(new_cipher, tensor._shape, tensor._context, tensor._depth + poly_depth)
        if degree > 0 and not (getattr(tensor._context, "_enable_gpu", False) and is_sparse_degree4):
            result._needs_rescale = True
        return tensor._propagate_batch_meta(result)

    def inv_sqrt(
        self, domain: tuple[float, float] | None = None, shallow: bool = False
    ) -> "EncryptedTensor":
        """Compute 1/sqrt(x) using CryptoInvSqrt.

        Uses Chebyshev polynomial approximation followed by Newton-Raphson
        refinement to approximate the inverse square root. Supports both
        Mock and OpenFHE backends.

        Args:
            domain: Input domain (a, b). 
                - For shallow=False: Must be (0.1, 100.0) in v1 (default).
                - For shallow=True: Default (1.0, 10.0) for polynomial-only mode.
            shallow: If True, use polynomial-only approximation without bootstrap.
                Use this for OpenFHE GPU contexts where bootstrap is unavailable 
                or has known issues. If False (default), use full method
                with Newton refinement and bootstrap.

        Returns:
            EncryptedTensor with 1/sqrt(x) approximation.

        Raises:
            NotImplementedError: If domain != (0.1, 100.0) for shallow=False
            RuntimeError: If shallow=False and enable_bootstrap=False

        Example:
            >>> # Full method (requires bootstrap)
            >>> inv_sqrt_x = x.inv_sqrt()
            >>>
            >>> # Shallow method (no bootstrap, suitable for GPU)
            >>> inv_sqrt_x = x.inv_sqrt(domain=(1.0, 10.0), shallow=True)

        Note:
            - shallow=False: Uses 2 bootstrap operations internally.
            - shallow=True: No bootstrap required, narrower domain for accuracy.

        Reference:
            Choi, H. (2025). PP-STAT. CIKM'25.
        """
        if shallow:
            from cukks.stats.crypto_inv_sqrt import crypto_inv_sqrt_shallow
            
            if domain is None:
                domain = (1.0, 10.0)
            return crypto_inv_sqrt_shallow(self, domain=domain)
        else:
            from cukks.stats.crypto_inv_sqrt import crypto_inv_sqrt
            
            if domain is None:
                domain = (0.1, 100.0)
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
        
        This method is not available in pure HE mode.
        Use encrypt_cnn_input() to pre-apply im2col before encryption,
        then use EncryptedConv2d with _cnn_layout input.
        
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
        raise RuntimeError(
            "conv2d() is not available in pure HE mode. "
            "Use encrypt_cnn_input() to pre-apply im2col before encryption, "
            "then use EncryptedConv2d with _cnn_layout input."
        )
    
    def matmul_2d(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> "EncryptedTensor":
        """Matrix multiply for 2D input (num_patches, features).
        
        This method is not available in pure HE mode.
        Use encrypt_cnn_input() to pre-apply im2col before encryption,
        then use EncryptedLinear with packed BSGS matmul.
        
        Args:
            weight: Weight matrix (out_features, in_features).
            bias: Optional bias (out_features,).
            
        Returns:
            Encrypted result of shape (num_patches, out_features).
        """
        raise RuntimeError(
            "matmul_2d() is not available in pure HE mode. "
            "Use encrypt_cnn_input() to pre-apply im2col before encryption, "
            "then use EncryptedLinear with packed BSGS matmul."
        )
    
    def avgpool2d(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int] = (0, 0),
    ) -> "EncryptedTensor":
        """Apply 2D average pooling.
        
        This method is not available in pure HE mode.
        Use encrypt_cnn_input() to pre-apply im2col before encryption,
        then use EncryptedAvgPool2d with _cnn_layout input.
        
        Args:
            kernel_size: Pooling kernel size (kH, kW).
            stride: Pooling stride (sH, sW).
            padding: Pooling padding (pH, pW).
            
        Returns:
            Encrypted output after average pooling.
        """
        raise RuntimeError(
            "avgpool2d() is not available in pure HE mode. "
            "Use encrypt_cnn_input() to pre-apply im2col before encryption, "
            "then use EncryptedAvgPool2d with _cnn_layout input."
        )
    
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
        result = EncryptedTensor(self._cipher, tuple(shape), self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        # Propagate packed-batch meta (view doesn't change slot layout),
        # but NOT _cnn_layout (reshape invalidates spatial im2col layout).
        if self._packed_batch:
            result._packed_batch = True
            result._batch_size = self._batch_size
            result._slots_per_sample = self._slots_per_sample
        return result
    
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
        """Unfold tensor into patches (im2col).
        
        This method is not available in pure HE mode.
        Use encrypt_cnn_input() to pre-apply im2col before encryption.
        
        Args:
            kernel_size: Kernel size (kH, kW).
            stride: Stride (sH, sW).
            padding: Padding (pH, pW).
            
        Returns:
            Encrypted patches tensor.
        """
        raise RuntimeError(
            "unfold() is not available in pure HE mode. "
            "Use encrypt_cnn_input() to pre-apply im2col before encryption."
        )
    
    def clone(self) -> "EncryptedTensor":
        """Create a copy of this tensor.
        
        Note: This creates a new Python wrapper but the underlying
        ciphertext may be shared (copy-on-write semantics).
        """
        result = EncryptedTensor(self._cipher, self._shape, self._context, self._depth)
        result._needs_rescale = self._needs_rescale
        return self._propagate_batch_meta(result)
    
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
            "needs_rescale": self._needs_rescale,
            "cnn_layout": self._cnn_layout,
            "packed_batch": self._packed_batch,
            "batch_size": self._batch_size,
            "slots_per_sample": self._slots_per_sample,
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

        warnings.warn(
            "EncryptedTensor.load() reconstructs ciphertexts by setting attributes on a "
            "dummy encryption. This is only reliable with the mock backend. For real "
            "OpenFHE backends, use the native serialization API instead.",
            stacklevel=2,
        )
        
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
        
        result = cls(cipher, shape, context, depth)
        result._needs_rescale = tensor_data.get("needs_rescale", False)
        result._cnn_layout = tensor_data.get("cnn_layout", None)
        result._packed_batch = tensor_data.get("packed_batch", False)
        result._batch_size = tensor_data.get("batch_size", None)
        result._slots_per_sample = tensor_data.get("slots_per_sample", None)
        return result
