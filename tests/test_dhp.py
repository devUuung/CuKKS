import pytest
import torch


def _patch_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mocks.mock_backend import MockCKKSConfig, MockCKKSContext
    import cukks.context as ctx_module

    monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
    monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
    monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)


def _mock_ctx(monkeypatch: pytest.MonkeyPatch):
    from cukks import CKKSInferenceContext

    _patch_mock_backend(monkeypatch)
    return CKKSInferenceContext(device="cpu", use_bsgs=False)


def test_precode_weights():
    import cukks.batching as batching_module

    w_h = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    w_h1 = torch.tensor([[0.5, -1.5], [2.5, -3.5]], dtype=torch.float64)

    precoded = batching_module.DHPacker.precode_weights(w_h, w_h1)

    expected = w_h.to(torch.complex128) + 1j * w_h1.to(torch.complex128)
    assert precoded.dtype == torch.complex128
    torch.testing.assert_close(precoded, expected)


def test_unpack_heads_recovers_original(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.DHPacker(num_heads=2)

    head_h_plain = torch.tensor([1.0, -2.0, 3.5, 0.25], dtype=torch.float64)
    head_h1_plain = torch.tensor([-0.5, 4.0, -1.5, 2.25], dtype=torch.float64)

    packed_col = ctx.encrypt(head_h_plain).add(ctx.encrypt(head_h1_plain).mul_by_i())
    head_h, head_h1 = packer.unpack_heads([packed_col])

    recovered_h = ctx.decrypt(head_h[0], shape=(4,)).to(torch.float64)
    recovered_h1 = ctx.decrypt(head_h1[0], shape=(4,)).to(torch.float64)

    torch.testing.assert_close(recovered_h, head_h_plain, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(recovered_h1, head_h1_plain, rtol=1e-5, atol=1e-5)


def test_repack_after_attention(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.DHPacker(num_heads=2)

    att_h_plain = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    att_h1_plain = torch.tensor([1.5, 2.5, -0.5, -4.0], dtype=torch.float64)

    repacked = packer.repack_after_attention(
        [ctx.encrypt(att_h_plain)],
        [ctx.encrypt(att_h1_plain)],
    )

    assert len(repacked) == 1
    expected = att_h_plain.to(torch.complex128) + 1j * att_h1_plain.to(torch.complex128)
    actual = repacked[0]._cipher.data[:4]
    assert actual.is_complex()
    torch.testing.assert_close(actual, expected)


def test_repack_then_unpack_roundtrip(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.DHPacker(num_heads=2)

    torch.manual_seed(7)
    att_h_plain = [torch.randn(6, dtype=torch.float64) for _ in range(3)]
    att_h1_plain = [torch.randn(6, dtype=torch.float64) for _ in range(3)]

    repacked = packer.repack_after_attention(
        [ctx.encrypt(col) for col in att_h_plain],
        [ctx.encrypt(col) for col in att_h1_plain],
    )
    unpacked_h, unpacked_h1 = packer.unpack_heads(repacked)

    for idx in range(3):
        recovered_h = ctx.decrypt(unpacked_h[idx], shape=(6,)).to(torch.float64)
        recovered_h1 = ctx.decrypt(unpacked_h1[idx], shape=(6,)).to(torch.float64)
        torch.testing.assert_close(recovered_h, att_h_plain[idx], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(recovered_h1, att_h1_plain[idx], rtol=1e-5, atol=1e-5)


def test_extract_final(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.DHPacker(num_heads=2)

    real_part = torch.tensor([3.0, -2.0, 1.0, 0.0], dtype=torch.float64)
    imag_part = torch.tensor([10.0, -20.0, 30.0, -40.0], dtype=torch.float64)

    accumulated = ctx.encrypt(real_part).add(ctx.encrypt(imag_part).mul_by_i())
    final = packer.extract_final(accumulated)

    recovered = ctx.decrypt(final, shape=(4,)).to(torch.float64)
    torch.testing.assert_close(recovered, real_part, rtol=1e-5, atol=1e-5)
    assert not final._cipher.data[:4].is_complex()


def test_num_heads_must_be_even():
    import cukks.batching as batching_module

    with pytest.raises(ValueError, match="must be even"):
        batching_module.DHPacker(num_heads=3)


def test_num_heads_must_be_positive():
    import cukks.batching as batching_module

    with pytest.raises(ValueError, match="must be positive"):
        batching_module.DHPacker(num_heads=0)

    with pytest.raises(ValueError, match="must be positive"):
        batching_module.DHPacker(num_heads=-2)
