"""Tests for serialization of Context and EncryptedTensor."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from cukks.batching import PackingLayout
from cukks.context import CKKSInferenceContext, InferenceConfig
from cukks.tensor import EncryptedTensor
from conftest import requires_real_backend


class TestContextSerialization:
    
    def test_context_save_load(self, use_mock_backend, tmp_path: Path):
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
        with pytest.warns(UserWarning, match="unsafe"):
            ctx.save_context(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle deserialization"):
            loaded_ctx = CKKSInferenceContext.load_context(save_path, allow_unsafe_pickle=True)
        
        assert loaded_ctx.config.poly_mod_degree == 8192
        assert loaded_ctx.config.scale_bits == 30
        assert loaded_ctx.config.mult_depth == 3
        assert loaded_ctx.device == "cpu"
        assert loaded_ctx.use_bsgs is True
        assert loaded_ctx.auto_bootstrap is True
        assert loaded_ctx.bootstrap_threshold == 3
    
    def test_context_load_alias(self, use_mock_backend, tmp_path: Path):
        ctx = CKKSInferenceContext()
        save_path = tmp_path / "context.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            ctx.save_context(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle deserialization"):
            loaded_ctx = CKKSInferenceContext.load(save_path, allow_unsafe_pickle=True)
        
        assert loaded_ctx.config.poly_mod_degree == ctx.config.poly_mod_degree
    
    def test_context_save_load_with_rotations(self, use_mock_backend, tmp_path: Path):
        rotations = [1, 2, 4, 8, -1, -2, -4, -8]
        ctx = CKKSInferenceContext(rotations=rotations)
        
        save_path = tmp_path / "context.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            ctx.save_context(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle deserialization"):
            loaded_ctx = CKKSInferenceContext.load_context(save_path, allow_unsafe_pickle=True)
        
        assert loaded_ctx._rotations == rotations

    def test_context_pickle_requires_explicit_opt_in(self, use_mock_backend, tmp_path: Path):
        ctx = CKKSInferenceContext()
        save_path = tmp_path / "context.bin"

        with pytest.raises(RuntimeError, match="allow_unsafe_pickle=True"):
            ctx.save_context(save_path)

        with pytest.warns(UserWarning, match="unsafe"):
            ctx.save_context(save_path, allow_unsafe_pickle=True)

        with pytest.raises(RuntimeError, match="allow_unsafe_pickle=True"):
            CKKSInferenceContext.load_context(save_path)


class TestTensorSerialization:
    
    def test_tensor_save_load(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            enc_tensor.save(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle"):
            loaded_tensor = EncryptedTensor.load(
                save_path,
                mock_enc_context,
                allow_unsafe_pickle=True,
            )
        
        assert loaded_tensor.shape == enc_tensor.shape
        assert loaded_tensor.depth == enc_tensor.depth
    
    def test_save_load_preserves_values(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            enc_tensor.save(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle"):
            loaded_tensor = EncryptedTensor.load(
                save_path,
                mock_enc_context,
                allow_unsafe_pickle=True,
            )
        decrypted = mock_enc_context.decrypt(loaded_tensor)
        
        torch.testing.assert_close(decrypted, original)
    
    def test_tensor_save_load_multidim(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        enc_tensor = mock_enc_context.encrypt(original)
        
        save_path = tmp_path / "tensor.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            enc_tensor.save(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle"):
            loaded_tensor = EncryptedTensor.load(
                save_path,
                mock_enc_context,
                allow_unsafe_pickle=True,
            )
        
        assert loaded_tensor.shape == (2, 2)
        decrypted = mock_enc_context.decrypt(loaded_tensor)
        torch.testing.assert_close(decrypted, original)
    
    def test_tensor_save_load_after_operations(self, mock_enc_context, tmp_path: Path):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))
        result = a.add(b)
        
        save_path = tmp_path / "result.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            result.save(save_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle"):
            loaded = EncryptedTensor.load(
                save_path,
                mock_enc_context,
                allow_unsafe_pickle=True,
            )
        decrypted = mock_enc_context.decrypt(loaded)
        
        expected = torch.tensor([5.0, 7.0, 9.0])
        torch.testing.assert_close(decrypted, expected)


class TestRoundTrip:
    
    def test_context_and_tensor_roundtrip(self, mock_enc_context, tmp_path: Path):
        original = torch.randn(10)
        enc_tensor = mock_enc_context.encrypt(original)
        
        tensor_path = tmp_path / "tensor.bin"
        with pytest.warns(UserWarning, match="unsafe"):
            enc_tensor.save(tensor_path, allow_unsafe_pickle=True)
        
        with pytest.warns(UserWarning, match="pickle"):
            loaded_tensor = EncryptedTensor.load(
                tensor_path,
                mock_enc_context,
                allow_unsafe_pickle=True,
            )
        decrypted = mock_enc_context.decrypt(loaded_tensor)

        torch.testing.assert_close(decrypted, original, rtol=1e-5, atol=1e-5)

    def test_tensor_pickle_requires_explicit_opt_in(self, mock_enc_context, tmp_path: Path):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc_tensor = mock_enc_context.encrypt(original)
        save_path = tmp_path / "tensor.bin"

        with pytest.raises(RuntimeError, match="allow_unsafe_pickle=True"):
            enc_tensor.save(save_path)

        with pytest.warns(UserWarning, match="unsafe"):
            enc_tensor.save(save_path, allow_unsafe_pickle=True)

        with pytest.raises(RuntimeError, match="allow_unsafe_pickle=True"):
            EncryptedTensor.load(save_path, mock_enc_context)


class TestNativeSerialization:
    @requires_real_backend
    def test_native_context_and_tensor_roundtrip(self, tmp_path: Path):
        config = InferenceConfig(
            poly_mod_degree=32768,
            scale_bits=40,
            mult_depth=4,
        )
        ctx = CKKSInferenceContext(
            config=config,
            device="cpu",
            rotations=[1, -1],
            use_bsgs=False,
            enable_gpu=False,
        )

        original = torch.tensor([0.25, -1.5, 2.75, 4.0], dtype=torch.float64)
        enc_tensor = ctx.encrypt(original)

        context_path = tmp_path / "native_context.bin"
        tensor_path = tmp_path / "native_tensor.bin"
        ctx.save_context(context_path)
        enc_tensor.save(tensor_path)

        assert context_path.exists()
        assert tensor_path.exists()
        assert Path(f"{context_path}.context.bin").exists()
        assert Path(f"{tensor_path}.ciphertext.bin").exists()

        loaded_ctx = CKKSInferenceContext.load_context(context_path)
        loaded_tensor = EncryptedTensor.load(tensor_path, loaded_ctx)

        decrypted = loaded_ctx.decrypt(loaded_tensor)
        torch.testing.assert_close(decrypted, original, rtol=1e-4, atol=1e-4)

    @requires_real_backend
    def test_native_roundtrip_preserves_runtime_metadata(self, tmp_path: Path):
        config = InferenceConfig(
            poly_mod_degree=32768,
            scale_bits=40,
            mult_depth=4,
        )
        ctx = CKKSInferenceContext(
            config=config,
            device="cpu",
            rotations=[1, -1],
            use_bsgs=False,
            enable_gpu=False,
            batch_size=2,
            architecture="stip",
        )

        original = torch.arange(8, dtype=torch.float64).reshape(2, 4)
        enc_tensor = ctx.encrypt(original)
        enc_tensor._needs_rescale = True
        enc_tensor._packed_batch = True
        enc_tensor._batch_size = 2
        enc_tensor._slots_per_sample = 4
        enc_tensor._packed_sample_shape = (4,)
        enc_tensor._cnn_layout = {
            "num_patches": 2,
            "patch_features": 4,
            "height": 1,
            "width": 2,
            "metadata": {"channels": 1},
        }
        enc_tensor._packing_layout = PackingLayout(
            seq_len=2,
            d_model=4,
            num_heads=2,
            block_size=4,
        )
        enc_tensor._stip_layout_fresh = True

        context_path = tmp_path / "native_context_meta.bin"
        tensor_path = tmp_path / "native_tensor_meta.bin"
        ctx.save_context(context_path)
        enc_tensor.save(tensor_path)

        loaded_ctx = CKKSInferenceContext.load_context(context_path)
        loaded_tensor = EncryptedTensor.load(tensor_path, loaded_ctx)

        assert loaded_ctx.batch_size == 2
        assert loaded_ctx._architecture == "stip"
        assert loaded_tensor._needs_rescale is True
        assert loaded_tensor._packed_batch is True
        assert loaded_tensor._batch_size == 2
        assert loaded_tensor._slots_per_sample == 4
        assert loaded_tensor._packed_sample_shape == (4,)
        assert loaded_tensor._cnn_layout == enc_tensor._cnn_layout
        assert loaded_tensor._packing_layout == enc_tensor._packing_layout
        assert loaded_tensor._stip_layout_fresh is True

    @requires_real_backend
    def test_native_load_supports_cpu_to_gpu_context_migration(self, tmp_path: Path):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = InferenceConfig(
            poly_mod_degree=32768,
            scale_bits=40,
            mult_depth=4,
        )
        ctx = CKKSInferenceContext(
            config=config,
            device="cpu",
            rotations=[1, -1],
            use_bsgs=False,
            enable_gpu=False,
        )

        original = torch.tensor([0.5, -2.25, 3.125, 7.0], dtype=torch.float64)
        enc_tensor = ctx.encrypt(original)

        context_path = tmp_path / "native_context_cpu.bin"
        tensor_path = tmp_path / "native_tensor_cpu.bin"
        ctx.save_context(context_path)
        enc_tensor.save(tensor_path)

        loaded_ctx = CKKSInferenceContext.load_context(
            context_path,
            device="cuda",
            enable_gpu=True,
        )
        loaded_tensor = EncryptedTensor.load(tensor_path, loaded_ctx)

        decrypted = loaded_ctx.decrypt(loaded_tensor)
        torch.testing.assert_close(decrypted.cpu(), original, rtol=1e-4, atol=1e-4)

    @requires_real_backend
    def test_native_backend_exit_cleanup_does_not_segfault(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [env.get("PYTHONPATH"), str(Path.cwd()), str(Path.cwd() / "cukks" / "_native")] if p]
        )
        script = """
import torch
from cukks.context import CKKSInferenceContext, InferenceConfig
from cukks.profiling import get_profiler

assert get_profiler() is None
ctx = CKKSInferenceContext(
    config=InferenceConfig(poly_mod_degree=32768, scale_bits=40, mult_depth=4),
    device="cpu",
    rotations=[1, -1],
    use_bsgs=False,
    enable_gpu=False,
)
x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
y = x.add(1.0)
assert y.shape == (4,)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout
