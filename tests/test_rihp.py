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


def test_pack_hybrid_creates_complex_data(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    m = 4
    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.RIHPacker(seq_len=m)

    keys_plain = [
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        torch.tensor([-1.0, 0.5, 2.0, -3.0], dtype=torch.float64),
    ]
    keys = [ctx.encrypt(col) for col in keys_plain]

    hybrid = packer.pack_hybrid(keys)

    assert len(hybrid) == len(keys_plain)
    for idx, packed_col in enumerate(hybrid):
        expected = keys_plain[idx].to(torch.complex128) + 1j * torch.roll(
            keys_plain[idx], shifts=-(m // 2)
        ).to(torch.complex128)
        actual = packed_col._cipher.data[:m]
        assert actual.is_complex()
        torch.testing.assert_close(actual, expected)


def test_unpack_recovers_original_diagonals(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    m = 6
    half = m // 2
    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.RIHPacker(seq_len=m)

    torch.manual_seed(0)
    plain_diagonals = [torch.randn(m, dtype=torch.float64) for _ in range(m)]

    hybrid_results = []
    for r in range(half):
        real_diag = ctx.encrypt(plain_diagonals[r])
        imag_diag = ctx.encrypt(plain_diagonals[r + half])
        hybrid_results.append(real_diag.add(imag_diag.mul_by_i()))

    unpacked = packer.unpack_diagonals(hybrid_results)

    assert len(unpacked) == m
    for r in range(m):
        recovered = ctx.decrypt(unpacked[r], shape=(m,)).to(torch.float64)
        torch.testing.assert_close(recovered, plain_diagonals[r], rtol=1e-5, atol=1e-5)


def test_halved_ccmm_matches_naive(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    m = 6
    d = 3
    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.RIHPacker(seq_len=m)

    torch.manual_seed(1)
    queries_plain = [torch.randn(m, dtype=torch.float64) for _ in range(d)]
    keys_plain = [torch.randn(m, dtype=torch.float64) for _ in range(d)]

    queries = [ctx.encrypt(col) for col in queries_plain]
    keys = [ctx.encrypt(col) for col in keys_plain]

    keys_hybrid = packer.pack_hybrid(keys)
    hybrid_results = packer.halved_ccmm(queries, keys_hybrid)
    diagonals = packer.unpack_diagonals(hybrid_results)

    assert len(hybrid_results) == m // 2
    assert len(diagonals) == m

    expected_diagonals = []
    for rotation in range(m):
        acc = torch.zeros(m, dtype=torch.float64)
        for col in range(d):
            acc = acc + queries_plain[col] * torch.roll(keys_plain[col], shifts=-rotation)
        expected_diagonals.append(acc)

    for rotation in range(m):
        recovered = ctx.decrypt(diagonals[rotation], shape=(m,)).to(torch.float64)
        torch.testing.assert_close(recovered, expected_diagonals[rotation], rtol=1e-4, atol=1e-4)


def test_seq_len_must_be_even():
    import cukks.batching as batching_module

    with pytest.raises(ValueError, match="must be even"):
        batching_module.RIHPacker(seq_len=3)
