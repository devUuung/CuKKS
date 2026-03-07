"""Tests for batched inference functionality."""

from typing import Any

import pytest
import torch


class TestSlotPacker:

    def test_pack_single_sample(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=10, total_slots=100)
        samples = [torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])]
        
        packed = packer.pack(samples)
        
        assert packed.shape == (10,)
        torch.testing.assert_close(packed, samples[0].to(torch.float64))

    def test_pack_multiple_samples(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=4, total_slots=100)
        samples = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
            torch.tensor([9.0, 10.0, 11.0, 12.0]),
        ]
        
        packed = packer.pack(samples)
        
        assert packed.shape == (12,)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=torch.float64)
        torch.testing.assert_close(packed, expected)

    def test_pack_with_padding(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=5, total_slots=100)
        samples = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
        ]
        
        packed = packer.pack(samples)
        
        assert packed.shape == (10,)
        expected = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(packed, expected)

    def test_unpack_samples(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=4, total_slots=100)
        packed = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        
        samples = packer.unpack(packed, num_samples=2)
        
        assert len(samples) == 2
        torch.testing.assert_close(samples[0], torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        torch.testing.assert_close(samples[1], torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64))

    def test_pack_unpack_roundtrip(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=10, total_slots=1000)
        original_samples = [torch.randn(10) for _ in range(5)]
        
        packed = packer.pack(original_samples)
        recovered = packer.unpack(packed, num_samples=5)
        
        assert len(recovered) == 5
        for orig, rec in zip(original_samples, recovered):
            torch.testing.assert_close(rec, orig.to(torch.float64))

    def test_max_batch_size(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=100, total_slots=1000)
        
        assert packer.max_batch_size == 10

    def test_pack_exceeds_capacity_raises(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=100, total_slots=200)
        samples = [torch.randn(100) for _ in range(5)]
        
        with pytest.raises(ValueError, match="exceeds max batch size"):
            packer.pack(samples)

    def test_pack_empty_list_raises(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=10, total_slots=100)
        
        with pytest.raises(ValueError, match="Cannot pack empty"):
            packer.pack([])

    def test_pack_sample_too_large_raises(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=5, total_slots=100)
        samples = [torch.randn(10)]
        
        with pytest.raises(ValueError, match="exceeds slots_per_sample"):
            packer.pack(samples)

    def test_pack_inconsistent_sizes_raises(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=10, total_slots=100)
        samples = [torch.randn(5), torch.randn(7)]
        
        with pytest.raises(ValueError, match="Inconsistent sample sizes"):
            packer.pack(samples)

    def test_unpack_invalid_num_samples_raises(self):
        from cukks.batching import SlotPacker
        
        packer = SlotPacker(slots_per_sample=10, total_slots=100)
        packed = torch.randn(100)
        
        with pytest.raises(ValueError, match="num_samples must be positive"):
            packer.unpack(packed, num_samples=0)


class TestEncryptDecryptBatch:

    def test_encrypt_decrypt_batch_basic(self, mock_enc_context: Any):
        from cukks.batching import SlotPacker
        
        samples = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
        ]
        
        packer = SlotPacker(slots_per_sample=3, total_slots=8192)
        packed = packer.pack(samples)
        
        enc = mock_enc_context.encrypt(packed)
        dec = mock_enc_context.decrypt(enc)
        
        recovered = packer.unpack(dec.to(torch.float64), num_samples=2)
        
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0].to(torch.float64), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1].to(torch.float64), rtol=1e-4, atol=1e-4)

    def test_encrypt_decrypt_batch_multiple(self, mock_enc_context: Any):
        from cukks.batching import SlotPacker
        
        num_samples = 8
        sample_size = 16
        samples = [torch.randn(sample_size) for _ in range(num_samples)]
        
        packer = SlotPacker(slots_per_sample=sample_size, total_slots=8192)
        packed = packer.pack(samples)
        
        enc = mock_enc_context.encrypt(packed)
        dec = mock_enc_context.decrypt(enc)
        
        recovered = packer.unpack(dec.to(torch.float64), num_samples=num_samples)
        
        assert len(recovered) == num_samples
        for orig, rec in zip(samples, recovered):
            torch.testing.assert_close(rec, orig.to(torch.float64), rtol=1e-4, atol=1e-4)


class TestContextBatchMethods:

    def test_context_encrypt_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
        
        enc_batch = ctx.encrypt_batch(samples)

        assert enc_batch.shape == (2, 3)
        assert getattr(enc_batch, "_packed_batch", False) is True
        assert getattr(enc_batch, "_batch_size", None) == 2
        assert getattr(enc_batch, "_slots_per_sample", None) == 3

    def test_context_encrypt_batch_forward_with_identity(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        from cukks.nn import EncryptedIdentity
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        enc_batch = ctx.encrypt_batch(samples)
        enc_output = EncryptedIdentity()(enc_batch)

        assert not isinstance(enc_output, list)
        assert enc_output.shape == (2, 2)
        assert getattr(enc_output, "_packed_batch", False) is True
        recovered = ctx.decrypt_batch(enc_output)

        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1], rtol=1e-4, atol=1e-4)

    def test_standard_layer_accepts_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedLinear
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        enc_batch = ctx.encrypt_batch(samples)

        layer = EncryptedLinear(in_features=2, out_features=2, weight=torch.eye(2), bias=None)
        enc_output = layer(enc_batch)

        assert not isinstance(enc_output, list)
        assert enc_output.shape == (2, 2)
        assert getattr(enc_output, "_packed_batch", False) is True
        assert getattr(enc_output, "_slots_per_sample", None) == 2
        recovered = ctx.decrypt_batch(enc_output)

        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1], rtol=1e-4, atol=1e-4)

    def test_context_decrypt_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
        
        enc_batch = ctx.encrypt_batch(samples)
        recovered = ctx.decrypt_batch(enc_batch)
        
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1], rtol=1e-4, atol=1e-4)

    def test_context_batch_roundtrip_large(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        
        num_samples = 16
        sample_size = 128
        samples = [torch.randn(sample_size) for _ in range(num_samples)]
        
        enc_batch = ctx.encrypt_batch(samples)
        recovered = ctx.decrypt_batch(enc_batch)
        
        assert len(recovered) == num_samples
        for orig, rec in zip(samples, recovered):
            torch.testing.assert_close(rec, orig, rtol=1e-4, atol=1e-4)

    def test_context_batch_with_sample_shape(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        
        samples = [torch.randn(2, 3) for _ in range(4)]
        
        enc_batch = ctx.encrypt_batch(samples, slots_per_sample=6)
        recovered = ctx.decrypt_batch(enc_batch, sample_shape=(2, 3))
        
        assert len(recovered) == 4
        for orig, rec in zip(samples, recovered):
            torch.testing.assert_close(rec.reshape(-1), orig.reshape(-1), rtol=1e-4, atol=1e-4)


class TestBatchedInference:

    def test_batched_linear_inference(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        
        input_size = 4
        num_samples = 3
        samples = [torch.tensor([1.0, 2.0, 3.0, 4.0]) * (i + 1) for i in range(num_samples)]
        
        enc_batch = ctx.encrypt_batch(samples)
        
        scale_factor = 2.0
        scaled = enc_batch.mul([scale_factor] * input_size * num_samples)
        
        results = ctx.decrypt_batch(scaled)
        
        assert len(results) == num_samples
        for orig, res in zip(samples, results):
            expected = orig * scale_factor
            torch.testing.assert_close(res[:input_size], expected, rtol=1e-4, atol=1e-4)

    def test_batched_add_inference(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        
        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)
        
        ctx = CKKSInferenceContext(device="cpu")
        
        samples = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ]
        bias = [10.0, 20.0, 10.0, 20.0]
        
        enc_batch = ctx.encrypt_batch(samples)
        result = enc_batch.add(bias)
        
        recovered = ctx.decrypt_batch(result)
        
        torch.testing.assert_close(recovered[0][:2], torch.tensor([11.0, 22.0]), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1][:2], torch.tensor([13.0, 24.0]), rtol=1e-4, atol=1e-4)

    def test_batched_sequential_inference_stays_packed(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedLinear, EncryptedReLU, EncryptedSequential
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [
            torch.tensor([1.0, -2.0]),
            torch.tensor([3.0, -4.0]),
        ]
        enc_batch = ctx.encrypt_batch(samples)

        model = EncryptedSequential(
            EncryptedLinear(2, 2, weight=torch.eye(2), bias=torch.tensor([1.0, 0.5])),
            EncryptedReLU(degree=2, method="minimax", domain=(-1, 1)),
            EncryptedLinear(2, 1, weight=torch.tensor([[2.0, -1.0]]), bias=torch.tensor([0.25])),
        )

        enc_output = model(enc_batch)

        assert not isinstance(enc_output, list)
        assert enc_output.shape == (2, 1)
        assert getattr(enc_output, "_packed_batch", False) is True
        assert getattr(enc_output, "_slots_per_sample", None) == 1

        recovered = ctx.decrypt_batch(enc_output)
        expected = []
        for sample in samples:
            hidden = sample + torch.tensor([1.0, 0.5])
            hidden = 0.375 + 0.5 * hidden + 0.125 * hidden.square()
            expected.append((2.0 * hidden[0] - hidden[1] + 0.25).reshape(1))

        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_encrypt_batch_no_longer_warns_about_sample_splitting(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module
        import warnings

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx.encrypt_batch(samples)

        assert all("sample-splitting" not in str(item.message) for item in caught)

    def test_layernorm_rejects_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedLayerNorm
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])

        with pytest.raises(RuntimeError, match="does not yet support packed-batch"):
            EncryptedLayerNorm(normalized_shape=2)(enc_batch)

    def test_attention_rejects_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])

        with pytest.raises(RuntimeError, match="does not support packed-batch"):
            EncryptedApproxAttention(embed_dim=2, num_heads=1)(enc_batch)

    def test_batchnorm1d_applies_per_sample_to_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedBatchNorm1d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        enc_batch = ctx.encrypt_batch(samples)

        layer = EncryptedBatchNorm1d(
            num_features=2,
            scale=torch.tensor([2.0, -1.0]),
            shift=torch.tensor([0.5, 3.0]),
        )
        enc_output = layer(enc_batch)

        assert enc_output.shape == (2, 2)
        assert getattr(enc_output, "_packed_batch", False) is True

        recovered = ctx.decrypt_batch(enc_output)
        expected = [
            torch.tensor([2.5, 1.0]),
            torch.tensor([6.5, -1.0]),
        ]
        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_batchnorm1d_rejects_mismatched_packed_shape(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedBatchNorm1d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])

        with pytest.raises(RuntimeError, match="requires slots_per_sample to match num_features"):
            EncryptedBatchNorm1d(
                num_features=2,
                scale=torch.tensor([1.0, 1.0]),
                shift=torch.tensor([0.0, 0.0]),
            )(enc_batch)

    def test_flatten_keeps_batch_dimension_for_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedFlatten
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]
        enc_batch = ctx.encrypt_batch(samples, slots_per_sample=4)

        flattened = EncryptedFlatten(start_dim=1)(enc_batch)

        assert flattened.shape == (2, 4)
        assert getattr(flattened, "_packed_batch", False) is True
        assert getattr(flattened, "_slots_per_sample", None) == 4

        recovered = ctx.decrypt_batch(flattened, sample_shape=(4,))
        torch.testing.assert_close(recovered[0], torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], torch.tensor([5.0, 6.0, 7.0, 8.0]), rtol=1e-4, atol=1e-4)

    def test_decrypt_batch_uses_packed_metadata_for_1d_shape(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedFlatten
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]
        enc_batch = ctx.encrypt_batch(samples, slots_per_sample=4)

        flattened = EncryptedFlatten(start_dim=0)(enc_batch)

        assert flattened.shape == (8,)
        assert getattr(flattened, "_packed_batch", False) is True
        assert getattr(flattened, "_batch_size", None) == 2
        assert getattr(flattened, "_slots_per_sample", None) == 4

        recovered = ctx.decrypt_batch(flattened, sample_shape=(4,))
        torch.testing.assert_close(recovered[0], torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], torch.tensor([5.0, 6.0, 7.0, 8.0]), rtol=1e-4, atol=1e-4)
