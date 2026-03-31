import torch
import torch.nn as nn
from importlib import import_module

EncryptedEmbedding = import_module("cukks.nn.embedding").EncryptedEmbedding


class TestEncryptedEmbedding:
    def test_from_torch(self):
        module = nn.Embedding(12, 5)

        encrypted = EncryptedEmbedding.from_torch(module)

        assert encrypted.num_embeddings == 12
        assert encrypted.embedding_dim == 5

    def test_weight_shape(self):
        encrypted = EncryptedEmbedding(7, 3, torch.randn(7, 3))

        assert encrypted.weight.shape == (7, 3)

    def test_mult_depth(self):
        encrypted = EncryptedEmbedding(7, 3, torch.randn(7, 3))

        assert encrypted.mult_depth() == 1

    def test_embedding_dim(self):
        encrypted = EncryptedEmbedding(9, 4, torch.randn(9, 4))

        assert encrypted.embedding_dim == 4

    def test_num_embeddings(self):
        encrypted = EncryptedEmbedding(9, 4, torch.randn(9, 4))

        assert encrypted.num_embeddings == 9
