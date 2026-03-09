"""Tests for batched inference functionality."""

import warnings
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


class TestPackingLayout:

    def test_layout_creation(self):
        from cukks.batching import PackingLayout

        layout = PackingLayout(seq_len=4, d_model=16, num_heads=4, block_size=16)

        assert layout.d_head == 4
        assert layout.total_slots_needed == 64

    def test_layout_token_offset(self):
        from cukks.batching import PackingLayout

        layout = PackingLayout(seq_len=4, d_model=16, num_heads=4, block_size=16)

        assert layout.token_offset(0) == 0
        assert layout.token_offset(1) == 16
        assert layout.token_offset(3) == 48

    def test_layout_head_range(self):
        from cukks.batching import PackingLayout

        layout = PackingLayout(seq_len=2, d_model=16, num_heads=4, block_size=16)

        assert layout.head_range(0) == (0, 4)
        assert layout.head_range(3) == (12, 16)

    def test_layout_validation(self):
        from cukks.batching import PackingLayout

        with pytest.raises(ValueError, match="divisible"):
            PackingLayout(seq_len=4, d_model=15, num_heads=4, block_size=16)
        with pytest.raises(ValueError, match="block_size"):
            PackingLayout(seq_len=4, d_model=16, num_heads=4, block_size=8)
        with pytest.raises(ValueError, match="seq_len"):
            PackingLayout(seq_len=0, d_model=16, num_heads=4, block_size=16)

    def test_layout_complex_packing_default(self):
        from cukks.batching import PackingLayout

        layout = PackingLayout(seq_len=4, d_model=16, num_heads=4, block_size=16)

        assert layout.complex_packing == "real_only"


class TestTokenBlockPacker:

    def test_pack_unpack_roundtrip(self):
        from cukks.batching import PackingLayout, TokenBlockPacker

        layout = PackingLayout(seq_len=4, d_model=16, num_heads=4, block_size=16)
        packer = TokenBlockPacker(layout, total_slots=128)
        source = torch.arange(64, dtype=torch.float64).reshape(4, 16)

        packed = packer.pack(source)
        recovered = packer.unpack(packed)

        assert packed.shape == (128,)
        torch.testing.assert_close(recovered, source)

    def test_pack_token_positions(self):
        from cukks.batching import PackingLayout, TokenBlockPacker

        layout = PackingLayout(seq_len=2, d_model=4, num_heads=2, block_size=4)
        packer = TokenBlockPacker(layout, total_slots=16)
        source = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        packed = packer.pack(source)

        torch.testing.assert_close(packed[:8], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))

    def test_pack_block_padding(self):
        from cukks.batching import PackingLayout, TokenBlockPacker

        layout = PackingLayout(seq_len=2, d_model=4, num_heads=2, block_size=8)
        packer = TokenBlockPacker(layout, total_slots=24)
        source = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        packed = packer.pack(source)

        torch.testing.assert_close(
            packed[:16],
            torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_head_masks_cover_active_region(self):
        from cukks.batching import PackingLayout, TokenBlockPacker

        layout = PackingLayout(seq_len=2, d_model=16, num_heads=4, block_size=16)
        packer = TokenBlockPacker(layout, total_slots=64)

        masks = packer.all_head_masks()
        mask_sum = [sum(mask[idx] for mask in masks) for idx in range(layout.total_slots_needed)]

        assert all(value == 1.0 for value in mask_sum)

    def test_non_real_packing_not_implemented(self):
        from cukks.batching import PackingLayout, TokenBlockPacker

        layout = PackingLayout(seq_len=2, d_model=4, num_heads=2, block_size=4, complex_packing="rihp")
        packer = TokenBlockPacker(layout, total_slots=16)

        with pytest.raises(NotImplementedError, match="rihp"):
            packer.pack(torch.ones(2, 4))


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

    def test_context_encrypt_decrypt_sequence(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        source = torch.arange(64, dtype=torch.float64).reshape(4, 16)

        enc = ctx.encrypt_sequence(source, num_heads=4)
        recovered = ctx.decrypt_sequence(enc)

        assert enc.shape == (4, 16)
        assert getattr(enc, "_packing_layout", None) is not None
        torch.testing.assert_close(recovered, source.to(torch.float32), rtol=1e-4, atol=1e-4)

    def test_context_encrypt_sequence_validation(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)

        with pytest.raises(ValueError, match="Expected 2D tensor"):
            ctx.encrypt_sequence(torch.arange(8.0), num_heads=2)

        with pytest.raises(ValueError, match="requires"):
            ctx.encrypt_sequence(torch.ones(2000, 16), num_heads=4)

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

class TestPackedBatchNativeExecution:

    def _patch_mock_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.mocks.mock_backend import MockCKKSConfig, MockCKKSContext
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

    def test_encrypt_batch_no_longer_warns_about_plaintext_fallback(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        self._patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", batch_size=2)
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            enc_batch = ctx.encrypt_batch(samples)

        assert getattr(enc_batch, "_packed_batch", False) is True
        assert not any("decrypt/re-encrypt" in str(w.message) for w in record)

    def test_legacy_packed_batch_fallback_raises(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        self._patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", batch_size=2)
        enc_batch = ctx.encrypt_batch([torch.tensor([1.0]), torch.tensor([2.0])])

        with pytest.raises(RuntimeError, match="fallback has been disabled"):
            ctx._forward_packed_batch(object(), enc_batch)

    def test_packed_batch_mlp_forward_never_decrypts(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedLinear, EncryptedSquare

        self._patch_mock_backend(monkeypatch)

        torch.manual_seed(42)
        batch_size = 4
        in_features = 8
        hidden_features = 8
        out_features = 4

        fc1 = torch.nn.Linear(in_features, hidden_features)
        fc2 = torch.nn.Linear(hidden_features, out_features)
        ctx = CKKSInferenceContext(device="cpu", batch_size=batch_size, max_rotation_dim=in_features, use_bsgs=True)
        enc_fc1 = EncryptedLinear.from_torch(fc1)
        enc_sq = EncryptedSquare()
        enc_fc2 = EncryptedLinear.from_torch(fc2)

        samples = [torch.randn(in_features) for _ in range(batch_size)]
        enc_batch = ctx.encrypt_batch(samples, slots_per_sample=in_features)

        original_decrypt = ctx.decrypt
        original_decrypt_batch = ctx.decrypt_batch

        def _forbid(*args: Any, **kwargs: Any):
            raise AssertionError("packed forward must not decrypt")

        monkeypatch.setattr(ctx, "decrypt", _forbid)
        monkeypatch.setattr(ctx, "decrypt_batch", _forbid)

        enc_out = enc_fc2(enc_sq(enc_fc1(enc_batch)))

        monkeypatch.setattr(ctx, "decrypt", original_decrypt)
        monkeypatch.setattr(ctx, "decrypt_batch", original_decrypt_batch)

        recovered = ctx.decrypt_batch(enc_out, num_samples=batch_size)

        with torch.no_grad():
            expected = [fc2(fc1(sample) ** 2) for sample in samples]

        assert getattr(enc_out, "_packed_batch", False) is True
        for exp, got in zip(expected, recovered):
            torch.testing.assert_close(got[:out_features], exp.to(torch.float32), rtol=1e-4, atol=1e-4)

    def test_packed_batch_cnn_forward_never_decrypts(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedAvgPool2d, EncryptedConv2d, EncryptedSquare

        self._patch_mock_backend(monkeypatch)

        torch.manual_seed(7)
        batch_size = 3
        channels = 1
        height = 4
        width = 4
        out_channels = 2

        conv = torch.nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        conv.eval()
        enc_conv = EncryptedConv2d.from_torch(conv)
        enc_sq = EncryptedSquare()
        enc_pool = EncryptedAvgPool2d(kernel_size=2, stride=2)

        ctx = CKKSInferenceContext(device="cpu", batch_size=batch_size, max_rotation_dim=64, use_bsgs=True)
        images = [torch.randn(channels, height, width) for _ in range(batch_size)]
        conv_params = [{"kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1)}]
        enc_batch = ctx.encrypt_cnn_input_batch(images, conv_params)

        original_decrypt = ctx.decrypt
        original_decrypt_batch = ctx.decrypt_batch

        def _forbid(*args: Any, **kwargs: Any):
            raise AssertionError("packed forward must not decrypt")

        monkeypatch.setattr(ctx, "decrypt", _forbid)
        monkeypatch.setattr(ctx, "decrypt_batch", _forbid)

        enc_out = enc_pool(enc_sq(enc_conv(enc_batch)))

        monkeypatch.setattr(ctx, "decrypt", original_decrypt)
        monkeypatch.setattr(ctx, "decrypt_batch", original_decrypt_batch)

        recovered = ctx.decrypt_batch(enc_out, num_samples=batch_size)

        with torch.no_grad():
            expected = []
            for image in images:
                plain = conv(image.unsqueeze(0))
                plain = plain.squeeze(0) ** 2
                plain = torch.nn.functional.avg_pool2d(plain, kernel_size=2, stride=2)
                expected.append(plain.permute(1, 2, 0).reshape(-1, out_channels).to(torch.float64))

        assert getattr(enc_out, "_packed_batch", False) is True
        assert enc_out._cnn_layout is not None
        assert enc_out._cnn_layout.get("batch_size") == batch_size
        sparse_positions = enc_out._cnn_layout["sparse_positions"]
        slots_per_sample = enc_out._slots_per_sample
        assert slots_per_sample is not None
        values_per_sample = expected[0].numel()
        for sample_idx, (exp, got) in enumerate(zip(expected, recovered)):
            block_start = sample_idx * slots_per_sample
            local_positions = [pos - block_start for pos in sparse_positions if block_start <= pos < block_start + slots_per_sample]
            dense = got[local_positions[:values_per_sample]].reshape(4, out_channels)
            torch.testing.assert_close(dense, exp.to(torch.float32), rtol=1e-4, atol=1e-4)

    def test_layernorm_supports_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedLayerNorm
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        enc_batch = ctx.encrypt_batch(samples)

        layer = EncryptedLayerNorm(normalized_shape=2)
        enc_output = layer(enc_batch)
        recovered = ctx.decrypt_batch(enc_output)
        expected = []
        for sample in samples:
            enc_sample = ctx.encrypt(sample)
            expected.append(ctx.decrypt(layer(enc_sample)))

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_attention_supports_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        enc_batch = ctx.encrypt_batch(samples)

        attn = EncryptedApproxAttention(embed_dim=2, num_heads=1)
        enc_output = attn(enc_batch)
        recovered = ctx.decrypt_batch(enc_output)
        expected = []
        for sample in samples:
            enc_sample = ctx.encrypt(sample)
            expected.append(ctx.decrypt(attn(enc_sample)))

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

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

        with pytest.raises(RuntimeError, match="requires packed sample shape"):
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

    def test_batchnorm2d_supports_packed_batch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedBatchNorm2d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            torch.tensor([[[2.0, 4.0], [6.0, 8.0]], [[1.0, 3.0], [5.0, 7.0]]]),
        ]
        enc_batch = ctx.encrypt_batch(samples)

        layer = EncryptedBatchNorm2d(
            num_features=2,
            scale=torch.tensor([2.0, -1.0]),
            shift=torch.tensor([0.5, 1.5]),
        )
        enc_output = layer(enc_batch)
        recovered = ctx.decrypt_batch(enc_output, sample_shape=(2, 2, 2))

        expected = []
        for sample in samples:
            transformed = sample.clone()
            transformed[0] = transformed[0] * 2.0 + 0.5
            transformed[1] = transformed[1] * -1.0 + 1.5
            expected.append(transformed)

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("normalization_mode", ["power_softmax", "gaussian"])
    def test_attention_supports_packed_multi_token_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        normalization_mode: str,
    ):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.tensor([[0.5, 0.1], [0.2, 0.7]]),
        ]
        enc_batch = ctx.encrypt_batch(samples)

        attn = EncryptedApproxAttention(
            embed_dim=2,
            num_heads=1,
            normalization_mode=normalization_mode,
        )
        enc_output = attn(enc_batch)
        recovered = ctx.decrypt_batch(enc_output, sample_shape=(2, 2))

        expected = []
        for sample in samples:
            tokens = [ctx.encrypt(sample[i]) for i in range(sample.shape[0])]
            enc_tokens = attn.forward_attention(tokens, tokens, tokens)
            assert isinstance(enc_tokens, list)
            expected.append(torch.stack([ctx.decrypt(token) for token in enc_tokens]))

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("seq_len", [2, 4])
    @pytest.mark.parametrize("normalization_mode", ["power_softmax", "gaussian"])
    def test_attention_packed_multi_token_reuses_rotations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seq_len: int,
        normalization_mode: str,
    ):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig, MockCKKSTensor
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        rotate_calls = 0
        original_rotate = MockCKKSTensor.rotate

        def counting_rotate(self: MockCKKSTensor, steps: int) -> MockCKKSTensor:
            nonlocal rotate_calls
            rotate_calls += 1
            return original_rotate(self, steps)

        monkeypatch.setattr(MockCKKSTensor, "rotate", counting_rotate)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.randn(seq_len, 2) * 0.1, torch.randn(seq_len, 2) * 0.1]
        enc_batch = ctx.encrypt_batch(samples)

        attn = EncryptedApproxAttention(
            embed_dim=2,
            num_heads=1,
            normalization_mode=normalization_mode,
        )
        _ = attn(enc_batch)

        if normalization_mode == "power_softmax":
            assert rotate_calls == 0
        else:
            assert rotate_calls == (2 * seq_len * seq_len) + (4 * (seq_len - 1))

    def test_attention_packed_power_softmax_uses_native_fast_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig, MockCKKSTensor
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        native_calls = 0
        original_native = MockCKKSTensor.packed_self_attention_power

        def counting_native(
            self: MockCKKSTensor,
            key: MockCKKSTensor,
            value: MockCKKSTensor,
            batch_size: int,
            seq_len: int,
            embed_dim: int,
            scale: float,
            shift: float,
            reciprocal_coeffs: list[float],
            renorm_reciprocal_coeffs: list[float],
        ) -> MockCKKSTensor:
            nonlocal native_calls
            native_calls += 1
            return original_native(
                self,
                key,
                value,
                batch_size,
                seq_len,
                embed_dim,
                scale,
                shift,
                reciprocal_coeffs,
                renorm_reciprocal_coeffs,
            )

        monkeypatch.setattr(MockCKKSTensor, "packed_self_attention_power", counting_native)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([torch.randn(2, 2) * 0.1, torch.randn(2, 2) * 0.1])
        attn = EncryptedApproxAttention(embed_dim=2, num_heads=1, normalization_mode="power_softmax")

        _ = attn(enc_batch)

        assert native_calls == 1

    def test_attention_packed_gaussian_keeps_python_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig, MockCKKSTensor
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        native_calls = 0
        original_native = MockCKKSTensor.packed_self_attention_power

        def counting_native(
            self: MockCKKSTensor,
            key: MockCKKSTensor,
            value: MockCKKSTensor,
            batch_size: int,
            seq_len: int,
            embed_dim: int,
            scale: float,
            shift: float,
            reciprocal_coeffs: list[float],
            renorm_reciprocal_coeffs: list[float],
        ) -> MockCKKSTensor:
            nonlocal native_calls
            native_calls += 1
            return original_native(
                self,
                key,
                value,
                batch_size,
                seq_len,
                embed_dim,
                scale,
                shift,
                reciprocal_coeffs,
                renorm_reciprocal_coeffs,
            )

        monkeypatch.setattr(MockCKKSTensor, "packed_self_attention_power", counting_native)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([torch.randn(2, 2) * 0.1, torch.randn(2, 2) * 0.1])
        attn = EncryptedApproxAttention(embed_dim=2, num_heads=1, normalization_mode="gaussian")

        _ = attn(enc_batch)

        assert native_calls == 0

    def test_conv2d_supports_packed_batch_of_patches(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedConv2d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([[1.0], [2.0], [3.0], [4.0]]), torch.tensor([[5.0], [6.0], [7.0], [8.0]])]
        enc_batch = ctx.encrypt_batch(samples)

        layer = EncryptedConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            weight=torch.tensor([[[[2.0]]]]),
            bias=torch.tensor([0.5]),
        )
        enc_output = layer(enc_batch)
        recovered = ctx.decrypt_batch(enc_output, sample_shape=(4, 1))

        expected = []
        for sample in samples:
            enc_sample = ctx.encrypt(sample.flatten())
            enc_sample._cnn_layout = {"num_patches": 4, "patch_features": 1, "height": 2, "width": 2}
            enc_sample._shape = (4, 1)
            expected.append(ctx.decrypt(layer(enc_sample), shape=(4, 1)))

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_avgpool2d_supports_packed_batch_of_patches(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedAvgPool2d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([[1.0], [2.0], [3.0], [4.0]]), torch.tensor([[2.0], [4.0], [6.0], [8.0]])]
        enc_batch = ctx.encrypt_batch(samples)

        pool = EncryptedAvgPool2d(kernel_size=2, stride=2)
        enc_output = pool(enc_batch)
        recovered = ctx.decrypt_batch(enc_output, sample_shape=(1, 1))

        expected = [torch.tensor([[2.5]]), torch.tensor([[5.0]])]
        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_maxpool2d_supports_packed_batch_of_patches(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedMaxPool2d
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        ctx = CKKSInferenceContext(device="cpu")
        samples = [torch.tensor([[0.1], [0.7], [0.3], [0.5]]), torch.tensor([[0.2], [0.4], [0.9], [0.6]])]
        enc_batch = ctx.encrypt_batch(samples)

        pool = EncryptedMaxPool2d(kernel_size=2, stride=2)
        enc_output = pool(enc_batch)
        recovered = ctx.decrypt_batch(enc_output, sample_shape=(1, 1))

        expected = []
        for sample in samples:
            enc_sample = ctx.encrypt(sample.flatten())
            enc_sample._cnn_layout = {"num_patches": 4, "patch_features": 1, "height": 2, "width": 2}
            enc_sample._shape = (4, 1)
            expected.append(ctx.decrypt(pool(enc_sample), shape=(1, 1)))

        torch.testing.assert_close(recovered[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], expected[1], rtol=1e-4, atol=1e-4)
