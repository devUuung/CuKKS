"""
Backend Protocol — Formal contracts between cukks/ and the CKKS backend.

This module defines :class:`typing.Protocol` classes that specify exactly
what a backend must implement.  The two consuming sites are:

* ``cukks.tensor.EncryptedTensor`` → uses ``CipherHandle``
* ``cukks.context.CKKSInferenceContext`` → uses ``BackendContext``

No runtime behaviour lives here — only static type contracts.  The real
backend (``ckks.CKKSTensor`` / ``ckks.CKKSContext``) and the test mock
(``tests.mocks.mock_backend``) both satisfy these protocols.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# CipherHandle — everything EncryptedTensor calls on self._cipher
# ---------------------------------------------------------------------------

@runtime_checkable
class CipherHandle(Protocol):
    """Protocol for an encrypted ciphertext handle.

    Every method listed here is called by ``EncryptedTensor`` on its
    ``_cipher`` attribute.  A conforming backend must implement all of
    them; optional GPU-only helpers (``mul_by_i``, ``extract_real``,
    ``extract_imag``) may raise ``AttributeError`` — ``EncryptedTensor``
    already guards those calls with ``hasattr``.
    """

    # -- Properties ----------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return ciphertext metadata (scale, level, noise, …)."""
        ...

    @property
    def size(self) -> int:
        """Number of usable slots in this ciphertext."""
        ...

    # -- Element-wise arithmetic ---------------------------------------------

    def add(self, other: "CipherHandle | Sequence[float] | float") -> "CipherHandle":
        ...

    def sub(self, other: "CipherHandle | Sequence[float] | float") -> "CipherHandle":
        ...

    def mul(self, other: "CipherHandle | Sequence[float] | float") -> "CipherHandle":
        ...

    def neg(self) -> "CipherHandle":
        ...

    def square(self) -> "CipherHandle":
        ...

    # -- Rescale / level management ------------------------------------------

    def rescale(self) -> "CipherHandle":
        ...

    def bootstrap(self) -> "CipherHandle":
        ...

    # -- Slot manipulation ---------------------------------------------------

    def rotate(self, steps: int) -> "CipherHandle":
        ...

    def sum_slots(self) -> "CipherHandle":
        ...

    def sum_and_broadcast(self, n: int) -> "CipherHandle":
        ...

    def conjugate(self) -> "CipherHandle":
        ...

    # -- Matrix operations ---------------------------------------------------

    def matmul_dense(
        self,
        matrix: Sequence[Sequence[float]] | torch.Tensor,
    ) -> "CipherHandle":
        ...

    def matmul_bsgs(
        self,
        matrix: Sequence[Sequence[float]] | torch.Tensor,
        bsgs_n1: int = 0,
        bsgs_n2: int = 0,
        weight_hash: int = 0,
        diag_nonzero: Sequence[bool] | None = None,
    ) -> "CipherHandle":
        ...

    def poly_eval(self, coeffs: Sequence[float]) -> "CipherHandle":
        ...

    # -- Decryption (convenience on handle) ----------------------------------

    def decrypt(
        self,
        *,
        device: str | torch.device | None = None,
        shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# BackendContext — everything CKKSInferenceContext calls on self._ctx
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendContext(Protocol):
    """Protocol for a backend CKKS context.

    ``CKKSInferenceContext`` stores a ``BackendContext`` as ``self._ctx``
    after initialisation and delegates encrypt / decrypt through it.
    """

    def encrypt(self, tensor: torch.Tensor) -> CipherHandle:
        ...

    def decrypt(
        self,
        cipher: CipherHandle,
        *,
        shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# BackendConfig — constructor signature expected by the backend context
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendConfig(Protocol):
    """Protocol for backend configuration.

    ``CKKSInferenceContext._ensure_initialized`` creates a config object
    with these fields and passes it to the ``BackendContext`` constructor.
    """

    poly_mod_degree: int
    coeff_mod_bits: Sequence[int]
    scale_bits: int
    security_level: str | int | None
    enable_bootstrap: bool
    level_budget: Sequence[int] | None
    rotations: Sequence[int]
    relin: bool
    generate_conjugate_keys: bool
