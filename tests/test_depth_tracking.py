"""Tests for multiplicative depth tracking in EncryptedTensor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from mocks.mock_backend import MockCKKSConfig, MockCKKSContext
from cukks.tensor import EncryptedTensor


@pytest.fixture
def mock_context() -> MockCKKSContext:
    config = MockCKKSConfig(enable_bootstrap=True)
    return MockCKKSContext(config)


@pytest.fixture
def encrypted_tensor(mock_context: MockCKKSContext) -> EncryptedTensor:
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cipher = mock_context.encrypt(tensor)
    return EncryptedTensor(cipher, tensor.shape, mock_context, depth=0)


class TestInitialDepth:
    def test_initial_depth_zero(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0, 3.0])
        cipher = mock_context.encrypt(tensor)
        
        # when
        enc = EncryptedTensor(cipher, tensor.shape, mock_context)
        
        # then
        assert enc.depth == 0

    def test_custom_initial_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0, 3.0])
        cipher = mock_context.encrypt(tensor)
        
        # when
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=5)
        
        # then
        assert enc.depth == 5


class TestMulIncrements:
    def test_mul_plain_increments_depth(self, encrypted_tensor: EncryptedTensor):
        # given
        initial_depth = encrypted_tensor.depth
        
        # when
        result = encrypted_tensor.mul(2.0)
        
        # then
        assert result.depth == initial_depth + 1

    def test_mul_cipher_increments_depth(self, encrypted_tensor: EncryptedTensor):
        # given
        other = encrypted_tensor.clone()
        initial_depth = encrypted_tensor.depth
        
        # when
        result = encrypted_tensor.mul(other)
        
        # then
        assert result.depth == initial_depth + 1

    def test_mul_cipher_uses_max_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher1 = mock_context.encrypt(tensor)
        cipher2 = mock_context.encrypt(tensor)
        enc1 = EncryptedTensor(cipher1, tensor.shape, mock_context, depth=2)
        enc2 = EncryptedTensor(cipher2, tensor.shape, mock_context, depth=5)
        
        # when
        result = enc1.mul(enc2)
        
        # then
        assert result.depth == 6  # max(2, 5) + 1

    def test_square_increments_depth(self, encrypted_tensor: EncryptedTensor):
        # given
        initial_depth = encrypted_tensor.depth
        
        # when
        result = encrypted_tensor.square()
        
        # then
        assert result.depth == initial_depth + 1


class TestAddPreserves:
    def test_add_plain_preserves_depth(self, encrypted_tensor: EncryptedTensor):
        # given
        initial_depth = encrypted_tensor.depth
        
        # when
        result = encrypted_tensor.add(5.0)
        
        # then
        assert result.depth == initial_depth

    def test_add_cipher_uses_max_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher1 = mock_context.encrypt(tensor)
        cipher2 = mock_context.encrypt(tensor)
        enc1 = EncryptedTensor(cipher1, tensor.shape, mock_context, depth=3)
        enc2 = EncryptedTensor(cipher2, tensor.shape, mock_context, depth=7)
        
        # when
        result = enc1.add(enc2)
        
        # then
        assert result.depth == 7  # max(3, 7)

    def test_sub_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher1 = mock_context.encrypt(tensor)
        cipher2 = mock_context.encrypt(tensor)
        enc1 = EncryptedTensor(cipher1, tensor.shape, mock_context, depth=4)
        enc2 = EncryptedTensor(cipher2, tensor.shape, mock_context, depth=2)
        
        # when
        result = enc1.sub(enc2)
        
        # then
        assert result.depth == 4  # max(4, 2)


class TestMatmulIncrements:
    def test_matmul_increments_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0, 3.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=2)
        weight = torch.randn(4, 3)
        
        # when
        result = enc.matmul(weight)
        
        # then
        # matmul: depth+1, rescale: depth unchanged
        assert result.depth == 3


class TestPolyEvalDepth:
    def test_poly_eval_depth_linear(self, mock_context: MockCKKSContext):
        # given: degree 1 polynomial (a0 + a1*x)
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=0)
        coeffs = [1.0, 2.0]  # degree 1
        
        # when
        result = enc.poly_eval(coeffs)
        
        # then: degree 1 -> depth = 0 + 1 = 1
        assert result.depth == 1

    def test_poly_eval_depth_quadratic(self, mock_context: MockCKKSContext):
        # given: degree 2 polynomial (a0 + a1*x + a2*x^2)
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=1)
        coeffs = [1.0, 2.0, 3.0]  # degree 2
        
        # when
        result = enc.poly_eval(coeffs)
        
        # then: degree 2 -> depth = 1 + 2 = 3
        assert result.depth == 3

    def test_poly_eval_depth_degree_4(self, mock_context: MockCKKSContext):
        # given: degree 4 polynomial
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=0)
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]  # degree 4
        
        # when
        result = enc.poly_eval(coeffs)
        
        # then: degree 4 -> poly_depth = ceil(log2(5)) = 3, total depth = 0 + 3 = 3
        assert result.depth == 3


class TestOtherOperationsPreserveDepth:
    def test_rescale_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=3)
        
        # when
        result = enc.rescale()
        
        # then
        assert result.depth == 3

    def test_rotate_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=5)
        
        # when
        result = enc.rotate(2)
        
        # then
        assert result.depth == 5

    def test_bootstrap_resets_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=8)
        
        # when
        result = enc.bootstrap()
        
        # then
        assert result.depth == 0

    def test_neg_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=4)
        
        # when
        result = enc.neg()
        
        # then
        assert result.depth == 4

    def test_view_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=3)
        
        # when
        result = enc.view(4)
        
        # then
        assert result.depth == 3

    def test_clone_preserves_depth(self, mock_context: MockCKKSContext):
        # given
        tensor = torch.tensor([1.0, 2.0])
        cipher = mock_context.encrypt(tensor)
        enc = EncryptedTensor(cipher, tensor.shape, mock_context, depth=6)
        
        # when
        result = enc.clone()
        
        # then
        assert result.depth == 6


class TestChainedOperations:
    def test_chained_muls_accumulate_depth(self, encrypted_tensor: EncryptedTensor):
        # given
        initial = encrypted_tensor  # depth 0
        
        # when
        r1 = initial.mul(2.0)  # depth 1
        r2 = r1.mul(3.0)       # depth 2
        r3 = r2.mul(4.0)       # depth 3
        
        # then
        assert r1.depth == 1
        assert r2.depth == 2
        assert r3.depth == 3

    def test_mixed_operations_track_correctly(self, encrypted_tensor: EncryptedTensor):
        # given
        enc = encrypted_tensor  # depth 0
        
        # when
        r1 = enc.mul(2.0)      # depth 1
        r2 = r1.add(1.0)       # depth 1 (add preserves)
        r3 = r2.mul(3.0)       # depth 2
        r4 = r3.rescale()      # depth 2 (rescale preserves)
        
        # then
        assert r1.depth == 1
        assert r2.depth == 1
        assert r3.depth == 2
        assert r4.depth == 2
