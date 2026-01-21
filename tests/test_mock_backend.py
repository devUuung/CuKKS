import math
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mocks.mock_backend import MockCKKSConfig, MockCKKSContext, MockCKKSTensor


class TestMockCKKSContext:
    def test_encrypt_decrypt_roundtrip(self):
        ctx = MockCKKSContext()
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        encrypted = ctx.encrypt(original)
        decrypted = ctx.decrypt(encrypted)
        
        torch.testing.assert_close(decrypted, original)
    
    def test_encrypt_preserves_shape(self):
        ctx = MockCKKSContext()
        original = torch.randn(3, 4)
        
        encrypted = ctx.encrypt(original)
        decrypted = ctx.decrypt(encrypted)
        
        assert decrypted.shape == original.shape


class TestMockCKKSTensor:
    @pytest.fixture
    def ctx(self):
        return MockCKKSContext()
    
    def test_add_cipher_cipher(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = ctx.encrypt(torch.tensor([4.0, 5.0, 6.0]))
        
        result = a.add(b)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([5.0, 7.0, 9.0]))
    
    def test_add_cipher_plain(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        plain = torch.tensor([10.0, 20.0, 30.0])
        
        result = a.add(plain)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([11.0, 22.0, 33.0]))
    
    def test_add_operator(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0]))
        b = ctx.encrypt(torch.tensor([3.0, 4.0]))
        
        result = a + b
        
        torch.testing.assert_close(result.decrypt(), torch.tensor([4.0, 6.0]))
    
    def test_mul_cipher_cipher(self, ctx):
        a = ctx.encrypt(torch.tensor([2.0, 3.0, 4.0]))
        b = ctx.encrypt(torch.tensor([5.0, 6.0, 7.0]))
        
        result = a.mul(b)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([10.0, 18.0, 28.0]))
    
    def test_mul_cipher_scalar(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        
        result = a.mul(2.0)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([2.0, 4.0, 6.0]))
    
    def test_mul_operator(self, ctx):
        a = ctx.encrypt(torch.tensor([2.0, 3.0]))
        
        result = a * 3.0
        
        torch.testing.assert_close(result.decrypt(), torch.tensor([6.0, 9.0]))
    
    def test_neg(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, -2.0, 3.0]))
        
        result = -a
        
        torch.testing.assert_close(result.decrypt(), torch.tensor([-1.0, 2.0, -3.0]))
    
    def test_sub(self, ctx):
        a = ctx.encrypt(torch.tensor([5.0, 6.0, 7.0]))
        b = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        
        result = a.sub(b)
        
        torch.testing.assert_close(result.decrypt(), torch.tensor([4.0, 4.0, 4.0]))
    
    def test_rotate(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        result = a.rotate(1)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([2.0, 3.0, 4.0, 1.0]))
    
    def test_sum_slots(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        result = a.sum_slots()
        
        assert result.shape == (1,)
        assert abs(result.decrypt().item() - 10.0) < 1e-9
    
    def test_matmul_dense(self, ctx):
        x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3 identity-like
        
        result = x.matmul_dense(matrix)
        decrypted = result.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([1.0, 2.0]))
    
    def test_poly_eval_linear(self, ctx):
        x = ctx.encrypt(torch.tensor([2.0, 3.0]))
        coeffs = [1.0, 2.0]  # 1 + 2*x
        
        result = x.poly_eval(coeffs)
        decrypted = result.decrypt()
        
        expected = torch.tensor([5.0, 7.0])  # 1 + 2*2, 1 + 2*3
        torch.testing.assert_close(decrypted, expected)
    
    def test_poly_eval_quadratic(self, ctx):
        x = ctx.encrypt(torch.tensor([2.0]))
        coeffs = [1.0, 0.0, 1.0]  # 1 + x^2
        
        result = x.poly_eval(coeffs)
        
        assert abs(result.decrypt().item() - 5.0) < 1e-9  # 1 + 4
    
    def test_bootstrap_requires_enabled(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0]))
        
        with pytest.raises(RuntimeError, match="Bootstrapping was not enabled"):
            a.bootstrap()
    
    def test_bootstrap_when_enabled(self):
        config = MockCKKSConfig(enable_bootstrap=True)
        ctx = MockCKKSContext(config)
        a = ctx.encrypt(torch.tensor([1.0, 2.0]))
        
        result = a.bootstrap()
        
        torch.testing.assert_close(result.decrypt(), torch.tensor([1.0, 2.0]))
    
    def test_level_decreases_after_mul(self, ctx):
        a = ctx.encrypt(torch.tensor([1.0]))
        initial_level = a._level
        
        b = a.mul(2.0)
        
        assert b._level < initial_level


class TestMockWithEncryptedModules:
    @pytest.fixture
    def ctx(self):
        return MockCKKSContext()
    
    def test_linear_forward_with_mock(self, ctx):
        from ckks_torch.nn import EncryptedLinear
        
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        bias = torch.tensor([0.5, 0.5])
        linear = EncryptedLinear(2, 2, weight, bias)
        
        x_plain = torch.tensor([1.0, 1.0])
        x_enc = ctx.encrypt(x_plain)
        
        result = x_enc.matmul_dense(weight)
        result = result.add(bias)
        decrypted = result.decrypt()
        
        expected = x_plain @ weight.T + bias
        torch.testing.assert_close(decrypted, expected)
    
    def test_square_activation_with_mock(self, ctx):
        x = ctx.encrypt(torch.tensor([2.0, 3.0, 4.0]))
        
        squared = x.mul(x)
        decrypted = squared.decrypt()
        
        torch.testing.assert_close(decrypted, torch.tensor([4.0, 9.0, 16.0]))
