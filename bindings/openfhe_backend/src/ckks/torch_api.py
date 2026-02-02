from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

_gpu_backend = None
_cpu_backend = None
_backend = None


def _load_backends():
    global _gpu_backend, _cpu_backend, _backend
    
    if _backend is not None:
        return
    
    try:
        from ckks.backends import ckks_openfhe_gpu_backend as gpu_mod
        _gpu_backend = gpu_mod
    except Exception:
        try:
            import ckks_openfhe_gpu_backend as gpu_mod
            _gpu_backend = gpu_mod
        except ImportError:
            _gpu_backend = None
    
    try:
        from ckks.backends import ckks_openfhe_backend as cpu_mod
        _cpu_backend = cpu_mod
    except Exception:
        try:
            import ckks_openfhe_backend as cpu_mod
            _cpu_backend = cpu_mod
        except ImportError:
            _cpu_backend = None
    
    if _gpu_backend is None and _cpu_backend is None:
        raise ImportError(
            "Neither ckks_openfhe_gpu_backend nor ckks_openfhe_backend native extension is available. "
            "Build with `pip install -e bindings/openfhe_backend` first."
        )
    
    use_gpu = os.environ.get("CKKS_USE_GPU", "1").lower() in ("1", "true", "yes")
    if use_gpu and _gpu_backend is not None:
        _backend = _gpu_backend
    elif _cpu_backend is not None:
        _backend = _cpu_backend
    else:
        _backend = _gpu_backend


_load_backends()


def _tensor_to_list(values: torch.Tensor | Iterable[float], expected: int | None = None) -> list[float]:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1).detach()
    if expected not in (None, tensor.numel()):
        if tensor.numel() == 1 and expected is not None:
            tensor = tensor.expand(expected)
        else:
            raise ValueError(f"Plaintext size {tensor.numel()} does not match ciphertext slots {expected}")
    return tensor.tolist()


@dataclass
class CKKSConfig:
    """Configuration for a CKKS context."""

    poly_mod_degree: int = 16384
    coeff_mod_bits: Sequence[int] = (60, 40, 40, 60)
    scale_bits: int = 40
    security_level: str | int | None = "128_classic"
    enable_bootstrap: bool = False
    level_budget: Sequence[int] | None = None
    batch_size: int | None = None
    rotations: Sequence[int] = ()
    relin: bool = True
    generate_conjugate_keys: bool = True

    def resolved_batch_size(self) -> int:
        if self.batch_size:
            return int(self.batch_size)
        return self.poly_mod_degree // 2

    def security_level_code(self) -> int:
        if isinstance(self.security_level, int):
            return int(self.security_level)
        if self.security_level is None:
            # Explicitly disable standards-based ring-dimension checks.
            return _SECURITY_LEVELS["notset"]
        key = str(self.security_level).lower()
        if key not in _SECURITY_LEVELS:
            raise ValueError(
                "Unknown security level. Use one of: "
                + ", ".join(sorted(_SECURITY_LEVELS.keys()))
            )
        return _SECURITY_LEVELS[key]


class CKKSContext:
    """Thin wrapper around the native backend, tuned for Torch tensors."""

    def __init__(self, config: CKKSConfig | None = None, *, device: str | torch.device | None = None, enable_gpu: bool = True):
        self.config = config or CKKSConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._enable_gpu = enable_gpu and _gpu_backend is not None
        
        active_backend = _gpu_backend if self._enable_gpu else (_cpu_backend or _backend)
        
        if hasattr(active_backend, "create_context"):
            if self._enable_gpu:
                self._ctx = active_backend.create_context(
                    int(self.config.poly_mod_degree),
                    list(self.config.coeff_mod_bits),
                    int(self.config.scale_bits),
                    int(self.config.security_level_code()),
                    bool(self.config.enable_bootstrap),
                    list(self.config.level_budget or []),
                    int(self.config.resolved_batch_size()),
                    True,
                )
            else:
                self._ctx = active_backend.create_context(
                    int(self.config.poly_mod_degree),
                    list(self.config.coeff_mod_bits),
                    int(self.config.scale_bits),
                    int(self.config.security_level_code()),
                    bool(self.config.enable_bootstrap),
                    list(self.config.level_budget or []),
                    int(self.config.resolved_batch_size()),
                )
        else:
            self._ctx = _backend.create_context(
                int(self.config.poly_mod_degree),
                list(self.config.coeff_mod_bits),
                int(self.config.scale_bits),
                int(self.config.security_level_code()),
                bool(self.config.enable_bootstrap),
                list(self.config.level_budget or []),
                int(self.config.resolved_batch_size()),
            )
        
        self._active_backend = active_backend if hasattr(active_backend, "keygen") else _backend
        
        if self._enable_gpu:
            self._keys = self._active_backend.keygen(
                self._ctx, list(self.config.rotations), bool(self.config.relin), bool(self.config.generate_conjugate_keys), True
            )
        else:
            self._keys = self._active_backend.keygen(
                self._ctx, list(self.config.rotations), bool(self.config.relin), bool(self.config.generate_conjugate_keys)
            )
    
    @property
    def gpu_enabled(self) -> bool:
        if hasattr(self._ctx, "gpu_enabled"):
            return self._ctx.gpu_enabled
        return False

    def encrypt(self, tensor: torch.Tensor) -> "CKKSTensor":
        flat = _tensor_to_list(tensor)
        cipher = self._active_backend.encrypt(self._ctx, self._keys, flat)
        return CKKSTensor(self, cipher, tuple(tensor.shape), tensor.device)

    def decrypt(
        self,
        tensor: "CKKSTensor",
        *,
        device: str | torch.device | None = None,
        shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        handle = tensor._cipher if isinstance(tensor, CKKSTensor) else tensor
        values = self._active_backend.decrypt(self._ctx, self._keys, handle)
        if isinstance(tensor, CKKSTensor) and tensor.size:
            values = values[: tensor.size]
        target_device = torch.device(device) if device is not None else (
            tensor.device if isinstance(tensor, CKKSTensor) else self.device
        )
        result = torch.tensor(values, dtype=torch.float64, device=target_device)
        final_shape = shape or (tensor.shape if isinstance(tensor, CKKSTensor) else None)
        if final_shape:
            result = result.view(final_shape)
        return result


class CKKSTensor:
    """Encrypted tensor that mirrors a Torch tensor shape."""

    def __init__(self, context: CKKSContext, cipher, shape: Sequence[int], device):
        self.context = context
        self._cipher = cipher
        self.shape = tuple(shape)
        self.device = torch.device(device) if device is not None else context.device
        self.size = int(math.prod(self.shape)) if len(self.shape) > 0 else 1
        self._backend = context._active_backend

    @property
    def metadata(self):
        return self._backend.cipher_metadata(self._cipher)

    def add(self, other: "CKKSTensor | torch.Tensor | float | int") -> "CKKSTensor":
        if isinstance(other, CKKSTensor):
            self._check_context(other)
            new_cipher = self._backend.add_cipher(self._cipher, other._cipher)
            return CKKSTensor(self.context, new_cipher, self.shape, self.device)
        plain = _tensor_to_list(other, expected=self.size)
        new_cipher = self._backend.add_plain(self._cipher, plain)
        return CKKSTensor(self.context, new_cipher, self.shape, self.device)

    def sub(self, other: "CKKSTensor | torch.Tensor | float | int") -> "CKKSTensor":
        if isinstance(other, CKKSTensor):
            self._check_context(other)
            new_cipher = self._backend.sub_cipher(self._cipher, other._cipher)
            return CKKSTensor(self.context, new_cipher, self.shape, self.device)
        plain = _tensor_to_list(other, expected=self.size)
        new_cipher = self._backend.sub_plain(self._cipher, plain)
        return CKKSTensor(self.context, new_cipher, self.shape, self.device)

    def mul(self, other: "CKKSTensor | torch.Tensor | float | int") -> "CKKSTensor":
        if isinstance(other, CKKSTensor):
            self._check_context(other)
            new_cipher = self._backend.mul_cipher(self._cipher, other._cipher)
            return CKKSTensor(self.context, new_cipher, self.shape, self.device)
        plain = _tensor_to_list(other, expected=self.size)
        new_cipher = self._backend.mul_plain(self._cipher, plain)
        return CKKSTensor(self.context, new_cipher, self.shape, self.device)

    def square(self) -> "CKKSTensor":
        """Square this ciphertext."""
        if hasattr(self._backend, 'square'):
            new_cipher = self._backend.square(self._cipher)
        else:
            new_cipher = self._backend.mul_cipher(self._cipher, self._cipher)
        return CKKSTensor(self.context, new_cipher, self.shape, self.device)

    def conjugate(self) -> "CKKSTensor":
        new_cipher = self._backend.conjugate(self._cipher)
        return CKKSTensor(self.context, new_cipher, self.shape, self.device)

    def rescale(self) -> "CKKSTensor":
        return CKKSTensor(self.context, self._backend.rescale(self._cipher), self.shape, self.device)

    def rotate(self, steps: int) -> "CKKSTensor":
        rotated = self._backend.rotate(self._cipher, int(steps))
        return CKKSTensor(self.context, rotated, self.shape, self.device)

    def sum_slots(self) -> "CKKSTensor":
        summed = self._backend.sum_slots(self._cipher)
        return CKKSTensor(self.context, summed, (1,), self.device)

    def matmul_diagonal(self, diagonals: Sequence[Sequence[float]]) -> "CKKSTensor":
        diag_plain = [ _tensor_to_list(diag, expected=self.size) for diag in diagonals ]
        result = self._backend.matvec_diag(self._cipher, diag_plain)
        return CKKSTensor(self.context, result, self.shape, self.device)

    def matmul_dense(self, matrix: Sequence[Sequence[float] | torch.Tensor]) -> "CKKSTensor":
        rows = [_tensor_to_list(torch.as_tensor(row, dtype=torch.float64, device="cpu")) for row in matrix]
        if not rows:
            raise ValueError("matrix must not be empty")
        row_len = len(rows[0])
        for r in rows:
            if len(r) != row_len:
                raise ValueError("matrix must be rectangular")
        if row_len > self.size:
            raise ValueError(f"matrix columns {row_len} exceed ciphertext size {self.size}")
        needed_rotations = set(range(1, row_len))
        have_rotations = {int(r) % row_len for r in self.context.config.rotations}
        missing = sorted(needed_rotations - have_rotations)
        if missing:
            raise ValueError(
                f"matmul_dense requires rotation keys for shifts {{1..{row_len-1}}}; missing {missing}. "
                "Add them to CKKSConfig.rotations."
            )
        result = self._backend.matmul_dense(self._cipher, rows)
        return CKKSTensor(self.context, result, (len(rows),), self.device)

    def matmul_bsgs(self, matrix: Sequence[Sequence[float]], bsgs_n1: int = 0, bsgs_n2: int = 0) -> "CKKSTensor":
        result = self._backend.matmul_bsgs(self._cipher, matrix, bsgs_n1, bsgs_n2)
        return CKKSTensor(self.context, result, (len(matrix),), self.device)

    def poly_eval(self, coeffs: Sequence[float]) -> "CKKSTensor":
        evaluated = self._backend.poly_eval(self._cipher, list(coeffs))
        return CKKSTensor(self.context, evaluated, self.shape, self.device)

    def bootstrap(self) -> "CKKSTensor":
        if not self.context.config.enable_bootstrap:
            raise RuntimeError("Bootstrapping was not enabled when the context was created")
        bootstrapped = self._backend.bootstrap(self._cipher)
        return CKKSTensor(self.context, bootstrapped, self.shape, self.device)

    def decrypt(self, *, device: str | torch.device | None = None, shape: Sequence[int] | None = None) -> torch.Tensor:
        return self.context.decrypt(self, device=device, shape=shape)

    def _check_context(self, other: "CKKSTensor") -> None:
        if self.context is not other.context:
            raise ValueError("Ciphertexts originate from different CKKS contexts or key sets")

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        # other - self = -(self - other) = -self + other
        return self.mul(-1.0).add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return self.mul(-1.0)

    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"CKKSTensor(shape={self.shape}, device='{self.device}', "
            f"scale={meta.get('scale')}, level={meta.get('level')})"
        )


_SECURITY_LEVELS = {
    "128_classic": 0,
    "192_classic": 1,
    "256_classic": 2,
    "128_quantum": 3,
    "192_quantum": 4,
    "256_quantum": 5,
    "notset": 0xFFFFFFFF,
    "none": 0xFFFFFFFF,
    "off": 0xFFFFFFFF,
}
