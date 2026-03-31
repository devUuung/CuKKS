"""EncryptedEmbedding - Encrypted embedding lookup via one-hot matmul."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .module import EncryptedModule

if TYPE_CHECKING:
    from ..tensor import EncryptedTensor


class EncryptedEmbedding(EncryptedModule):
    """Encrypted embedding layer.

    Expects one-hot encoded encrypted inputs of length ``num_embeddings`` and
    computes the embedding lookup as a plaintext matrix multiplication.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, weight: torch.Tensor) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = weight.detach().to(dtype=torch.float64, device="cpu")

        if self.weight.shape != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                "weight must have shape "
                f"({self.num_embeddings}, {self.embedding_dim}), got {tuple(self.weight.shape)}"
            )

        self.register_parameter("weight", self.weight)

    def forward(self, x: "EncryptedTensor") -> "EncryptedTensor":
        return x.matmul(self.weight.T)

    @classmethod
    def from_torch(cls, module: torch.nn.Embedding) -> "EncryptedEmbedding":
        return cls(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            weight=module.weight.data,
        )

    def mult_depth(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
        )
