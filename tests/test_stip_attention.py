from typing import Any

import pytest
import torch


def _patch_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mocks.mock_backend import MockCKKSConfig, MockCKKSContext
    import cukks.context as ctx_module

    monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
    monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
    monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)


class TestSTIPAttention:

    def test_encrypt_sequence_sets_fresh_provenance(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)

        assert getattr(enc, "_packing_layout", None) is not None
        assert getattr(enc, "_stip_layout_fresh", False) is True

    def test_generic_ops_do_not_preserve_fresh_provenance(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        rotated = enc.rotate(0)

        assert getattr(rotated, "_packing_layout", None) is not None
        assert getattr(rotated, "_stip_layout_fresh", False) is False

    def test_clone_preserves_fresh_provenance(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        cloned = enc.clone()

        assert getattr(cloned, "_stip_layout_fresh", False) is True

    def test_forward_routes_stip_layout_to_stip_path(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        called = {"stip": False}

        def fake_stip(x: Any):
            called["stip"] = True
            return x

        monkeypatch.setattr(attn, "_forward_attention_stip", fake_stip)

        out = attn(enc)

        assert called["stip"] is True
        assert out is enc

    def test_default_single_tensor_path_unchanged(self, mock_enc_context: Any):
        from cukks.nn import EncryptedApproxAttention

        attn = EncryptedApproxAttention(embed_dim=8, num_heads=2, softmax_degree=4)
        enc_x = mock_enc_context.encrypt(torch.randn(1, 8) * 0.5)

        out: Any = attn(enc_x)

        dec = mock_enc_context.decrypt(out)
        assert torch.isfinite(dec).all()

    def test_packed_batch_path_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu")
        enc_batch = ctx.encrypt_batch([
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.tensor([[0.5, 0.1], [0.2, 0.7]]),
        ])
        attn = EncryptedApproxAttention(embed_dim=2, num_heads=1)

        out = attn(enc_batch)

        recovered = ctx.decrypt_batch(out, sample_shape=(2, 2))
        assert len(recovered) == 2

    @pytest.mark.parametrize("normalization_mode", ["power_softmax", "gaussian"])
    @pytest.mark.parametrize("seq_len", [2, 4])
    def test_stip_attention_matches_list_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        normalization_mode: str,
        seq_len: int,
    ):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        torch.manual_seed(0)
        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        source = torch.randn(seq_len, 4, dtype=torch.float64) * 0.1
        enc = ctx.encrypt_sequence(source, num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2, normalization_mode=normalization_mode)

        stip_out = attn(enc)
        stip_plain = ctx.decrypt_sequence(stip_out)

        tokens: Any = [ctx.encrypt(source[i]) for i in range(seq_len)]
        list_out = attn.forward_attention(tokens, tokens, tokens)
        assert isinstance(list_out, list)
        list_plain = torch.stack([ctx.decrypt(token) for token in list_out])

        if normalization_mode == "gaussian":
            torch.testing.assert_close(stip_plain, list_plain, rtol=3e-1, atol=2e-2)
        else:
            torch.testing.assert_close(stip_plain, list_plain, rtol=1e-4, atol=1e-4)

    def test_stip_output_preserves_layout_and_freshness(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4, dtype=torch.float64) * 0.1, num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        out = attn(enc)

        assert out.shape == (2, 4)
        assert getattr(out, "_packed_batch", False) is False
        assert getattr(out, "_packing_layout", None) == getattr(enc, "_packing_layout", None)
        assert getattr(out, "_stip_layout_fresh", False) is True

    def test_stip_attention_rejects_missing_fresh_provenance(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        stale = enc.rotate(0)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        with pytest.raises(RuntimeError, match="fresh sequence-layout provenance"):
            attn(stale)

    def test_stip_attention_rejects_d_model_mismatch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=8, num_heads=2)

        with pytest.raises(RuntimeError, match="d_model"):
            attn(enc)

    def test_stip_attention_rejects_num_heads_mismatch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=1)

        with pytest.raises(RuntimeError, match="num_heads"):
            attn(enc)

    def test_stip_attention_rejects_shape_mismatch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        reshaped = enc.view(8)
        reshaped._stip_layout_fresh = True
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        with pytest.raises(RuntimeError, match="tensor shape matching layout"):
            attn(reshaped)

    def test_stip_attention_rejects_block_padding(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2, block_size=8)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        with pytest.raises(NotImplementedError, match="block_size == d_model"):
            attn(enc)

    def test_stip_attention_rejects_seq_len_above_limit(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedApproxAttention

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(9, 4), num_heads=2)
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=2)

        with pytest.raises(NotImplementedError, match="Maximum is 8"):
            attn(enc)

    def test_decrypt_sequence_rejects_missing_fresh_provenance(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        stale = enc.rotate(0)

        with pytest.raises(ValueError, match="fresh STIP sequence-layout provenance"):
            ctx.decrypt_sequence(stale)

    def test_decrypt_sequence_rejects_shape_mismatch(self, monkeypatch: pytest.MonkeyPatch):
        from cukks import CKKSInferenceContext

        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False, architecture="stip")
        enc = ctx.encrypt_sequence(torch.randn(2, 4), num_heads=2)
        reshaped = enc.view(8)
        reshaped._stip_layout_fresh = True

        with pytest.raises(ValueError, match="shape does not match STIP packing layout"):
            ctx.decrypt_sequence(reshaped)
