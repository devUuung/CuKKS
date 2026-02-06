"""Mock CKKS backend for unit testing without OpenFHE.

This module provides mock implementations of CKKSConfig, CKKSContext, and CKKSTensor
that perform operations in plaintext, allowing the Python layer to be tested
without the actual homomorphic encryption backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

import torch

if TYPE_CHECKING:
    pass


@dataclass
class MockCKKSConfig:
    """Mock configuration for CKKS context."""

    poly_mod_degree: int = 16384
    coeff_mod_bits: Sequence[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40
    security_level: str | int | None = "128_classic"
    enable_bootstrap: bool = False
    level_budget: Sequence[int] | None = None
    batch_size: int | None = None
    rotations: Sequence[int] = field(default_factory=list)
    relin: bool = True
    generate_conjugate_keys: bool = True

    def resolved_batch_size(self) -> int:
        if self.batch_size:
            return int(self.batch_size)
        return self.poly_mod_degree // 2

    def security_level_code(self) -> int:
        return 0


class MockCKKSContext:
    """Mock CKKS context that operates on plaintext data.
    
    This context encrypts/decrypts by simply wrapping/unwrapping tensors,
    allowing Python layer tests without actual HE operations.
    """

    def __init__(
        self,
        config: MockCKKSConfig | None = None,
        *,
        device: str | torch.device | None = None,
        enable_gpu: bool = True,
    ):
        self.config = config or MockCKKSConfig()
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cpu")
        )
        self._slots = self.config.resolved_batch_size()

    def encrypt(self, tensor: torch.Tensor) -> "MockCKKSTensor":
        """Wrap a tensor as a MockCKKSTensor (no actual encryption)."""
        flat = tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        padded = torch.zeros(self._slots, dtype=torch.float64)
        padded[: flat.numel()] = flat
        return MockCKKSTensor(
            context=self,
            data=padded,
            shape=tuple(tensor.shape),
            device=self.device,
        )

    def decrypt(
        self,
        tensor: "MockCKKSTensor",
        *,
        device: str | torch.device | None = None,
        shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Unwrap a MockCKKSTensor back to a regular tensor."""
        final_shape = shape or tensor.shape
        num_elements = int(math.prod(final_shape)) if final_shape else tensor.size
        values = tensor.data[:num_elements].clone()
        target_device = (
            torch.device(device)
            if device is not None
            else tensor.device
        )
        result = values.to(device=target_device, dtype=torch.float32)
        if final_shape:
            result = result.view(final_shape)
        return result


class MockCKKSTensor:
    """Mock encrypted tensor that operates on plaintext data.
    
    All operations are performed on the underlying plaintext data,
    mimicking the behavior of real CKKS operations without encryption.
    """

    def __init__(
        self,
        context: MockCKKSContext,
        data: torch.Tensor,
        shape: Sequence[int],
        device: torch.device | str,
        depth: int = 0,
    ):
        self.context = context
        self.data = data.to(dtype=torch.float64)
        self.shape = tuple(shape)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.size = int(math.prod(self.shape)) if len(self.shape) > 0 else 1
        self._level = 10
        self._depth = depth

    @property
    def metadata(self) -> dict:
        """Return mock metadata."""
        return {
            "scale": 2**40,
            "level": self._level,
            "noise_scale": 2,
        }

    def add(self, other: "MockCKKSTensor | torch.Tensor | float | int") -> "MockCKKSTensor":
        """Add another tensor or scalar."""
        if isinstance(other, MockCKKSTensor):
            new_data = self.data + other.data
            new_depth = max(self._depth, other._depth)
        else:
            other_tensor = self._to_tensor(other)
            new_data = self.data + other_tensor
            new_depth = self._depth
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, new_depth)
        result._level = self._level
        return result

    def sub(self, other: "MockCKKSTensor | torch.Tensor | float | int") -> "MockCKKSTensor":
        """Subtract another tensor or scalar."""
        if isinstance(other, MockCKKSTensor):
            new_data = self.data - other.data
            new_depth = max(self._depth, other._depth)
        else:
            other_tensor = self._to_tensor(other)
            new_data = self.data - other_tensor
            new_depth = self._depth
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, new_depth)
        result._level = self._level
        return result

    def mul(self, other: "MockCKKSTensor | torch.Tensor | float | int") -> "MockCKKSTensor":
        """Multiply by another tensor or scalar."""
        if isinstance(other, MockCKKSTensor):
            new_data = self.data * other.data
            new_level = min(self._level, other._level) - 1
            new_depth = max(self._depth, other._depth) + 1
        else:
            other_tensor = self._to_tensor(other)
            new_data = self.data * other_tensor
            new_level = self._level - 1
            new_depth = self._depth + 1
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, new_depth)
        result._level = max(0, new_level)
        return result

    def neg(self) -> "MockCKKSTensor":
        """Negate the tensor."""
        result = MockCKKSTensor(self.context, -self.data, self.shape, self.device, self._depth)
        result._level = self._level
        return result

    def square(self) -> "MockCKKSTensor":
        new_data = self.data * self.data
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, self._depth + 1)
        result._level = max(0, self._level - 1)
        return result

    def rotate(self, steps: int) -> "MockCKKSTensor":
        """Rotate slots by the given number of steps."""
        active_data = self.data[: self.size]
        rotated_active = torch.roll(active_data, shifts=-int(steps))
        new_data = self.data.clone()
        new_data[: self.size] = rotated_active
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, self._depth)
        result._level = self._level
        return result


    def sum_slots(self) -> "MockCKKSTensor":
        """Sum all slots into the first slot."""
        total = self.data[: self.size].sum()
        new_data = torch.zeros_like(self.data)
        new_data[0] = total
        result = MockCKKSTensor(self.context, new_data, (1,), self.device, self._depth)
        result._level = self._level
        return result

    def sum_and_broadcast(self, n: int) -> "MockCKKSTensor":
        """Sum first n active slots and replicate the sum to all n positions.
        
        In real CKKS, rotation-and-add naturally fills all slots with the sum.
        This mock replicates that behavior for testing.
        
        Args:
            n: Number of active elements to sum and broadcast.
            
        Returns:
            MockCKKSTensor with sum replicated in first n positions, same shape.
        """
        total = self.data[:n].sum()
        new_data = self.data.clone()
        new_data[:n] = total
        result = MockCKKSTensor(self.context, new_data, self.shape, self.device, self._depth)
        result._level = self._level
        return result

    def conjugate(self) -> "MockCKKSTensor":
        """Complex conjugate (identity for real values)."""
        result = MockCKKSTensor(self.context, self.data.clone(), self.shape, self.device, self._depth)
        result._level = self._level
        return result

    def rescale(self) -> "MockCKKSTensor":
        """Mock rescale (no-op for mock)."""
        result = MockCKKSTensor(self.context, self.data.clone(), self.shape, self.device, self._depth)
        result._level = max(0, self._level - 1)
        return result

    def matmul_dense(
        self, matrix: Sequence[Sequence[float]] | torch.Tensor
    ) -> "MockCKKSTensor":
        """Matrix-vector multiplication.
        
        Computes matrix @ vector where vector is the encrypted data.
        """
        if isinstance(matrix, torch.Tensor):
            mat = matrix.to(dtype=torch.float64)
        else:
            mat = torch.tensor(matrix, dtype=torch.float64)
        
        in_features = mat.shape[1]
        out_features = mat.shape[0]
        
        vec = self.data[:in_features]
        result_vec = mat @ vec
        
        new_data = torch.zeros_like(self.data)
        new_data[:out_features] = result_vec
        
        result = MockCKKSTensor(
            self.context, new_data, (out_features,), self.device, self._depth + 1
        )
        result._level = max(0, self._level - 1)
        return result

    def matmul_bsgs(
        self,
        matrix: Sequence[Sequence[float]] | torch.Tensor,
        bsgs_n1: int = 0,
        bsgs_n2: int = 0,
    ) -> "MockCKKSTensor":
        """BSGS matrix-vector multiplication (mock delegates to dense).

        In the real backend, BSGS reduces rotation count from O(n) to O(sqrt(n)).
        The mock doesn't need this optimization — it simply delegates to
        ``matmul_dense`` while accepting the extra parameters silently.
        """
        return self.matmul_dense(matrix)

    def poly_eval(self, coeffs: Sequence[float]) -> "MockCKKSTensor":
        """Evaluate a polynomial on the encrypted data.
        
        coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
        """
        coeffs_tensor = torch.tensor(coeffs, dtype=torch.float64)
        x = self.data
        result = torch.zeros_like(x)
        
        for c in reversed(coeffs_tensor):  # Horner's method
            result = result * x + c
        
        degree = len(coeffs) - 1
        poly_depth = max(1, int(math.ceil(math.log2(degree + 1)))) if degree > 0 else 0
        
        new_tensor = MockCKKSTensor(self.context, result, self.shape, self.device, self._depth + poly_depth)
        new_tensor._level = max(0, self._level - poly_depth)
        return new_tensor

    def bootstrap(self) -> "MockCKKSTensor":
        """Mock bootstrap - refreshes the level."""
        if not self.context.config.enable_bootstrap:
            raise RuntimeError("Bootstrapping was not enabled when the context was created")
        result = MockCKKSTensor(self.context, self.data.clone(), self.shape, self.device, 0)
        result._level = 10
        return result

    def decrypt(
        self,
        *,
        device: str | torch.device | None = None,
        shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Decrypt (unwrap) this tensor."""
        return self.context.decrypt(self, device=device, shape=shape)

    def _to_tensor(self, other: torch.Tensor | float | int | Iterable) -> torch.Tensor:
        """Convert plaintext to padded tensor matching slot size."""
        tensor = torch.as_tensor(other, dtype=torch.float64).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(self.data.numel())
        else:
            padded = torch.zeros_like(self.data)
            padded[: tensor.numel()] = tensor
            return padded

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.neg().add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return self.neg()

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"MockCKKSTensor(shape={self.shape}, device='{self.device}', "
            f"scale={meta.get('scale')}, level={meta.get('level')})"
        )
