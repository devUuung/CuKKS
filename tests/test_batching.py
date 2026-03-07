"""Tests for batched inference functionality."""

import warnings

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

    def test_encrypt_decrypt_batch_basic(self, mock_enc_context):
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

    def test_encrypt_decrypt_batch_multiple(self, mock_enc_context):
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
        for i, (orig, rec) in enumerate(zip(samples, recovered)):
            torch.testing.assert_close(rec, orig.to(torch.float64), rtol=1e-4, atol=1e-4)


class TestContextBatchMethods:

    def test_context_encrypt_batch(self, monkeypatch):
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

    def test_context_encrypt_batch_forward_with_identity(self, monkeypatch):
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
        recovered = ctx.decrypt_batch(enc_output)

        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1], rtol=1e-4, atol=1e-4)

    def test_standard_layer_accepts_packed_batch(self, monkeypatch):
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
        recovered = ctx.decrypt_batch(enc_output)

        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0], samples[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(recovered[1], samples[1], rtol=1e-4, atol=1e-4)

    def test_context_decrypt_batch(self, monkeypatch):
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

    def test_context_batch_roundtrip_large(self, monkeypatch):
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

    def test_context_batch_with_sample_shape(self, monkeypatch):
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

    def test_batched_linear_inference(self, monkeypatch):
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
        for i, (orig, res) in enumerate(zip(samples, results)):
            expected = orig * scale_factor
            torch.testing.assert_close(res[:input_size], expected, rtol=1e-4, atol=1e-4)

    def test_batched_add_inference(self, monkeypatch):
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


class TestPackedBatchNativeExecution:

    def _patch_mock_backend(self, monkeypatch):
        from tests.mocks.mock_backend import MockCKKSConfig, MockCKKSContext
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

    def test_encrypt_batch_no_longer_warns_about_plaintext_fallback(self, monkeypatch):
        from cukks import CKKSInferenceContext

        self._patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", batch_size=2)
        samples = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            enc_batch = ctx.encrypt_batch(samples)

        assert getattr(enc_batch, "_packed_batch", False) is True
        assert not any("decrypt/re-encrypt" in str(w.message) for w in record)

    def test_legacy_packed_batch_fallback_raises(self, monkeypatch):
        from cukks import CKKSInferenceContext

        self._patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", batch_size=2)
        enc_batch = ctx.encrypt_batch([torch.tensor([1.0]), torch.tensor([2.0])])

        with pytest.raises(RuntimeError, match="fallback has been disabled"):
            ctx._forward_packed_batch(object(), enc_batch)

    def test_packed_batch_mlp_forward_never_decrypts(self, monkeypatch):
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

        def _forbid(*args, **kwargs):
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

    def test_packed_batch_cnn_forward_never_decrypts(self, monkeypatch):
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

        def _forbid(*args, **kwargs):
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
