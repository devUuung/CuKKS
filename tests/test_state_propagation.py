"""Tests for state propagation in EncryptedTensor operations.

These tests verify that _needs_rescale and _cnn_layout are correctly
propagated through various tensor operations to prevent state loss.
"""

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import torch


def _require_backend() -> None:
    from cukks.backend_loader import load_backend

    try:
        load_backend()
    except RuntimeError as exc:
        pytest.skip(f"backend unavailable in this environment: {exc}")


class TestNeedsRescalePropagation:
    """Verify _needs_rescale flag is correctly propagated through operations."""

    def test_view_preserves_needs_rescale(self, mock_enc_context):
        """view() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        reshaped = a_mul.view(4, 4)
        assert reshaped._needs_rescale is True

    def test_reshape_preserves_needs_rescale(self, mock_enc_context):
        """reshape() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        reshaped = a_mul.reshape((4, 4))
        assert reshaped._needs_rescale is True

    def test_flatten_preserves_needs_rescale(self, mock_enc_context):
        """flatten() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._shape = (4, 4)
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        flattened = a_mul.flatten()
        assert flattened._needs_rescale is True

    def test_clone_preserves_needs_rescale(self, mock_enc_context):
        """clone() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        cloned = a_mul.clone()
        assert cloned._needs_rescale is True

    def test_sum_and_broadcast_preserves_needs_rescale(self, mock_enc_context):
        """sum_and_broadcast() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        broadcasted = a_mul.sum_and_broadcast(16)
        assert broadcasted._needs_rescale is True

    def test_sum_slots_preserves_needs_rescale(self, mock_enc_context):
        """sum_slots() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        summed = a_mul.sum_slots()
        assert summed._needs_rescale is True

    def test_conjugate_preserves_needs_rescale(self, mock_enc_context):
        """conjugate() should preserve _needs_rescale flag."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        conjugated = a_mul.conjugate()
        assert conjugated._needs_rescale is True

    def test_fresh_tensor_has_false_needs_rescale(self, mock_enc_context):
        """Freshly encrypted tensor should have _needs_rescale=False."""
        a = mock_enc_context.encrypt(torch.randn(16))
        assert a._needs_rescale is False

    def test_mul_sets_needs_rescale_true(self, mock_enc_context):
        """mul() should set _needs_rescale=True."""
        a = mock_enc_context.encrypt(torch.randn(16))
        result = a.mul(2.0)
        assert result._needs_rescale is True

    def test_square_sets_needs_rescale_true(self, mock_enc_context):
        """square() should set _needs_rescale=True."""
        a = mock_enc_context.encrypt(torch.randn(16))
        result = a.square()
        assert result._needs_rescale is True


class TestCnnLayoutPropagation:
    """Verify _cnn_layout is correctly deep-copied through operations."""

    def test_clone_deep_copies_cnn_layout(self, mock_enc_context):
        """clone() should deep copy _cnn_layout."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            "num_patches": 4,
            "patch_features": 4,
            "nested": {"key": "original"},
        }

        cloned = a.clone()
        cloned._cnn_layout["nested"]["key"] = "modified"

        assert a._cnn_layout["nested"]["key"] == "original"

    def test_conjugate_deep_copies_cnn_layout(self, mock_enc_context):
        """conjugate() should deep copy _cnn_layout."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            "num_patches": 4,
            "patch_features": 4,
            "data": [1, 2, 3],
        }

        conjugated = a.conjugate()
        conjugated._cnn_layout["data"].append(4)

        assert len(a._cnn_layout["data"]) == 3

    def test_sum_and_broadcast_deep_copies_cnn_layout(self, mock_enc_context):
        """sum_and_broadcast() should deep copy _cnn_layout."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            "num_patches": 4,
            "patch_features": 4,
            "info": {"value": 100},
        }

        broadcasted = a.sum_and_broadcast(16)
        broadcasted._cnn_layout["info"]["value"] = 999

        assert a._cnn_layout["info"]["value"] == 100

    def test_view_clears_cnn_layout(self, mock_enc_context):
        """view() should NOT propagate _cnn_layout (reshape invalidates it)."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {"num_patches": 4, "patch_features": 4}

        reshaped = a.view(4, 4)
        assert reshaped._cnn_layout is None


class TestSerializationState:
    """Verify _needs_rescale and _cnn_layout survive save/load cycles."""

    def test_save_load_preserves_needs_rescale(self, mock_enc_context, tmp_path):
        """save() and load() should preserve _needs_rescale flag."""
        from cukks.tensor import EncryptedTensor

        a = mock_enc_context.encrypt(torch.randn(16))
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True

        save_path = tmp_path / "tensor.pkl"
        a_mul.save(save_path)

        loaded = EncryptedTensor.load(save_path, mock_enc_context)
        assert loaded._needs_rescale is True

    def test_save_load_preserves_cnn_layout(self, mock_enc_context, tmp_path):
        """save() and load() should preserve _cnn_layout."""
        from cukks.tensor import EncryptedTensor

        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            "num_patches": 4,
            "patch_features": 4,
        }

        save_path = tmp_path / "tensor.pkl"
        a.save(save_path)

        loaded = EncryptedTensor.load(save_path, mock_enc_context)
        assert loaded._cnn_layout is not None
        assert loaded._cnn_layout["num_patches"] == 4
        assert loaded._cnn_layout["patch_features"] == 4

    def test_save_load_fresh_tensor(self, mock_enc_context, tmp_path):
        """save() and load() should work for fresh tensors (defaults)."""
        from cukks.tensor import EncryptedTensor

        a = mock_enc_context.encrypt(torch.randn(16))
        assert a._needs_rescale is False
        assert a._cnn_layout is None

        save_path = tmp_path / "tensor.pkl"
        a.save(save_path)

        loaded = EncryptedTensor.load(save_path, mock_enc_context)
        assert loaded._needs_rescale is False
        assert loaded._cnn_layout is None


class TestContextSerialization:
    def test_save_load_context_roundtrip(self, tmp_path: Path):
        from cukks.context import CKKSInferenceContext, InferenceConfig

        ctx = CKKSInferenceContext(
            config=InferenceConfig(
                poly_mod_degree=8192,
                scale_bits=35,
                security_level="128_classic",
                mult_depth=5,
                enable_bootstrap=True,
                level_budget=None,
            ),
            device="cpu",
            rotations=[1, 2, 8],
            use_bsgs=False,
            max_rotation_dim=32,
            auto_bootstrap=True,
            bootstrap_threshold=3,
            batch_size=4,
            architecture="stip",
        )

        save_path = tmp_path / "context.pkl"
        ctx.save_context(save_path)

        with pytest.warns(UserWarning, match="pickle deserialization"):
            loaded = CKKSInferenceContext.load_context(save_path)

        assert loaded.config.poly_mod_degree == 8192
        assert loaded.config.scale_bits == 35
        assert loaded.config.security_level == "128_classic"
        assert loaded.config.mult_depth == 5
        assert loaded.config.enable_bootstrap is True
        assert loaded.device == "cpu"
        assert loaded.use_bsgs is False
        assert loaded._rotations == [1, 2, 8]
        assert loaded._max_rotation_dim == 32
        assert loaded._auto_bootstrap is True
        assert loaded._bootstrap_threshold == 3
        assert loaded._batch_size == 4
        assert loaded._architecture == "stip"


class TestRescaleSpyVerification:
    """Verify that rescale() is actually called when expected using spy."""

    def test_add_calls_rescale_for_scaled_left_operand(self, mock_enc_context):
        """When left operand needs rescale, add() should call rescale() on it."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))

        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True
        assert b._needs_rescale is False

        with patch.object(a_mul, "rescale", wraps=a_mul.rescale) as mock_rescale:
            result = a_mul.add(b)
            mock_rescale.assert_called_once()

    def test_add_calls_rescale_for_scaled_right_operand(self, mock_enc_context):
        """When right operand needs rescale, add() should call rescale() on it."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))

        b_mul = b.mul(2.0)
        assert a._needs_rescale is False
        assert b_mul._needs_rescale is True

        with patch.object(b_mul, "rescale", wraps=b_mul.rescale) as mock_rescale:
            result = a.add(b_mul)
            mock_rescale.assert_called_once()

    def test_add_no_rescale_when_both_fresh(self, mock_enc_context):
        """When both operands are fresh, add() should not call rescale()."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))

        assert a._needs_rescale is False
        assert b._needs_rescale is False

        with patch.object(a, "rescale", wraps=a.rescale) as mock_a_rescale:
            with patch.object(b, "rescale", wraps=b.rescale) as mock_b_rescale:
                result = a.add(b)
                mock_a_rescale.assert_not_called()
                mock_b_rescale.assert_not_called()

    def test_add_no_rescale_when_both_need_rescale(self, mock_enc_context):
        """When both operands need rescale, add() should not rescale either (optimization)."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))

        a_mul = a.mul(2.0)
        b_mul = b.mul(3.0)
        assert a_mul._needs_rescale is True
        assert b_mul._needs_rescale is True

        with patch.object(a_mul, "rescale", wraps=a_mul.rescale) as mock_a_rescale:
            with patch.object(b_mul, "rescale", wraps=b_mul.rescale) as mock_b_rescale:
                result = a_mul.add(b_mul)
                mock_a_rescale.assert_not_called()
                mock_b_rescale.assert_not_called()
                assert result._needs_rescale is True

    def test_rescale_skips_when_noise_scale_is_already_nominal(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))

        with patch.object(type(a._cipher), "metadata", new_callable=PropertyMock) as mock_metadata:
            mock_metadata.return_value = {
                "scale": 2**40,
                "level": 10,
                "noise_scale": 1,
            }
            with patch.object(a._cipher, "rescale", wraps=a._cipher.rescale) as mock_cipher_rescale:
                result = a.rescale()

        mock_cipher_rescale.assert_not_called()
        assert result is not a
        assert result._needs_rescale is False
        torch.testing.assert_close(mock_enc_context.decrypt(result), torch.tensor([1.0, 2.0, 3.0]))

    def test_rescale_calls_cipher_rescale_when_noise_scale_exceeds_nominal(self, mock_enc_context):
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        a_mul = a.mul(2.0)

        with patch.object(
            type(a_mul._cipher), "metadata", new_callable=PropertyMock
        ) as mock_metadata:
            mock_metadata.return_value = {
                "scale": 2**80,
                "level": 9,
                "noise_scale": 2,
            }
            with patch.object(
                a_mul._cipher, "rescale", wraps=a_mul._cipher.rescale
            ) as mock_cipher_rescale:
                result = a_mul.rescale()

        mock_cipher_rescale.assert_called_once()
        assert result._needs_rescale is False


class TestThreadSafeInitializationRobust:
    """Robust thread safety tests using barriers for synchronization."""

    def test_concurrent_initialization_with_barrier(self):
        """Multiple threads hitting _ensure_initialized simultaneously should be safe."""
        import threading
        from cukks.context import CKKSInferenceContext

        _require_backend()
        ctx = CKKSInferenceContext()
        init_count = [0]
        errors = []
        barrier = threading.Barrier(10)

        def init_thread():
            try:
                barrier.wait()
                ctx._ensure_initialized()
                init_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=init_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during initialization: {errors}"
        assert init_count[0] == 10
        assert ctx._initialized is True
