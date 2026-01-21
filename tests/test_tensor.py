"""Tests for EncryptedTensor operations."""

import pytest
import torch


class TestEncryptedTensorSub:
    
    def test_sub_cipher_cipher(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([5.0, 6.0, 7.0]))
        b = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        
        result = a.sub(b)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([4.0, 4.0, 4.0]))
    
    def test_sub_cipher_plain_tensor(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([10.0, 20.0, 30.0]))
        plain = torch.tensor([1.0, 2.0, 3.0])
        
        result = a.sub(plain)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([9.0, 18.0, 27.0]))
    
    def test_sub_cipher_scalar(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([5.0, 10.0, 15.0]))
        
        result = a.sub(2.0)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([3.0, 8.0, 13.0]))
    
    def test_sub_operator(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([5.0, 6.0]))
        b = mock_enc_context.encrypt(torch.tensor([1.0, 2.0]))
        
        result = a - b
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([4.0, 4.0]))
    
    def test_sub_preserves_shape(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([0.5, 0.5, 0.5]))
        
        result = a.sub(b)
        
        assert result.shape == (3,)


class TestEncryptedTensorDiv:
    
    def test_div_scalar(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([10.0, 20.0, 30.0]))
        
        result = a.div(2.0)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([5.0, 10.0, 15.0]))
    
    def test_div_by_fraction(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 4.0]))
        
        result = a.div(0.5)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(decrypted, torch.tensor([2.0, 4.0, 8.0]))
    
    def test_div_by_zero_raises(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0]))
        
        with pytest.raises(ValueError, match="Division by zero"):
            a.div(0)
    
    def test_div_preserves_shape(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        result = a.div(2.0)
        
        assert result.shape == (4,)


class TestEncryptedTensorMatmulShape:
    
    def test_matmul_shape_basic(self, mock_enc_context):
        x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        weight = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        result = x.matmul(weight)
        
        assert result.shape == (2,)
    
    def test_matmul_shape_square(self, mock_enc_context):
        x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        weight = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        result = x.matmul(weight)
        
        assert result.shape == (3,)
    
    def test_matmul_shape_expansion(self, mock_enc_context):
        x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0]))
        weight = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        
        result = x.matmul(weight)
        
        assert result.shape == (4,)
    
    def test_matmul_shape_with_bias(self, mock_enc_context):
        x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        weight = torch.tensor([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        bias = torch.tensor([0.5, 0.5])
        
        result = x.matmul(weight, bias)
        
        assert result.shape == (2,)
    
    def test_matmul_correctness(self, mock_enc_context):
        x_plain = torch.tensor([1.0, 2.0, 3.0])
        x = mock_enc_context.encrypt(x_plain)
        weight = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        result = x.matmul(weight)
        decrypted = mock_enc_context.decrypt(result)
        
        expected = weight @ x_plain
        torch.testing.assert_close(decrypted, expected.to(torch.float32), rtol=1e-4, atol=1e-4)
