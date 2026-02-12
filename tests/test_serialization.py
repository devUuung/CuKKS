"""Tests for serialization of Context and EncryptedTensor."""

import tempfile
from pathlib import Path

import pytest
import torch

from cukks.context import CKKSInferenceContext, InferenceConfig
from cukks.tensor import EncryptedTensor


class TestContextSerialization:
    
    def test_context_save_load(self, tmp_path: Path):
        config = InferenceConfig(
            poly_mod_degree=8192,
            scale_bits=30,
            mult_depth=3,
        )
        ctx = CKKSInferenceContext(
            config=config,
            device="cpu",
            use_bsgs=True,
            auto_bootstrap=True,
            bootstrap_threshold=3,
        )
        
        save_path = tmp_path / "context.bin"
        ctx.save_context(save_path)
        
        loaded_ctx = CKKSInferenceContext.load_context(save_path)
        
        assert loaded_ctx.config.poly_mod_degree == 8192
        assert loaded_ctx.config.scale_bits == 30
        assert loaded_ctx.config.mult_depth == 3
        assert loaded_ctx.device == "cpu"
        assert loaded_ctx.use_bsgs is True
        assert loaded_ctx.auto_bootstrap is True
        assert loaded_ctx.bootstrap_threshold == 3
    
    def test_context_load_alias(self, tmp_path: Path):
        ctx = CKKSInferenceContext()
        save_path = tmp_path / "context.bin"
        ctx.save_context(save_path)
        
        loaded_ctx = CKKSInferenceContext.load(save_path)
        
        assert loaded_ctx.config.poly_mod_degree == ctx.config.poly_mod_degree
    
    def test_context_save_load_with_rotations(self, tmp_path: Path):
        rotations = [1, 2, 4, 8, -1, -2, -4, -8]
        ctx = CKKSInferenceContext(rotations=rotations)
        
        save_path = tmp_path / "context.bin"
        ctx.save_context(save_path)
        
        loaded_ctx = CKKSInferenceContext.load_context(save_path)
        
        assert loaded_ctx._rotations == rotations


class TestTensorSerialization:
    
    def test_tensor_save_load(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        enc_tensor.save(save_path)
        
        loaded_tensor = EncryptedTensor.load(save_path, mock_enc_context)
        
        assert loaded_tensor.shape == enc_tensor.shape
        assert loaded_tensor.depth == enc_tensor.depth
    
    def test_save_load_preserves_values(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        enc_tensor.save(save_path)
        
        loaded_tensor = EncryptedTensor.load(save_path, mock_enc_context)
        decrypted = mock_enc_context.decrypt(loaded_tensor)
        
        torch.testing.assert_close(decrypted, original)
    
    def test_tensor_save_load_multidim(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        enc_tensor.save(save_path)
        
        loaded_tensor = EncryptedTensor.load(save_path, mock_enc_context)
        
        assert loaded_tensor.shape == (2, 2)
        decrypted = mock_enc_context.decrypt(loaded_tensor)
        torch.testing.assert_close(decrypted, original)
    
    def test_tensor_save_load_after_operations(self, mock_enc_context, tmp_path: Path):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))
        result = a.add(b)
        
        save_path = tmp_path / "result.bin"
        result.save(save_path)
        
        loaded = EncryptedTensor.load(save_path, mock_enc_context)
        decrypted = mock_enc_context.decrypt(loaded)
        
        expected = torch.tensor([5.0, 7.0, 9.0])
        torch.testing.assert_close(decrypted, expected)


class TestRoundTrip:
    
    def test_context_and_tensor_roundtrip(self, mock_enc_context, tmp_path: Path):
        original = torch.randn(10)
        enc_tensor = mock_enc_context.encrypt(original)
        
        tensor_path = tmp_path / "tensor.bin"
        enc_tensor.save(tensor_path)
        
        loaded_tensor = EncryptedTensor.load(tensor_path, mock_enc_context)
        decrypted = mock_enc_context.decrypt(loaded_tensor)
        
        torch.testing.assert_close(decrypted, original, rtol=1e-5, atol=1e-5)
