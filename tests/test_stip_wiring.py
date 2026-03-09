import pytest
import torch
from typing import Any


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


def _encrypt_columns(ctx: Any, plain: torch.Tensor) -> list[Any]:
    return [ctx.encrypt(plain[:, col]) for col in range(plain.shape[1])]


def _decrypt_columns(ctx: Any, columns: list[Any]) -> torch.Tensor:
    return torch.stack([ctx.decrypt(col, shape=(col.shape[0],)).to(torch.float64) for col in columns], dim=1)


def _baseline_forward_no_dhp(attn: Any, x_columns: list[Any], seq_len: int, use_rihp: bool) -> list[Any]:
    import cukks.batching as batching_module

    if attn.q_weight is not None:
        q_cols = attn._pcmm(x_columns, attn.q_weight, attn.q_bias)
        k_cols = attn._pcmm(x_columns, attn.k_weight, attn.k_bias)
        v_cols = attn._pcmm(x_columns, attn.v_weight, attn.v_bias)
    else:
        q_cols = list(x_columns)
        k_cols = list(x_columns)
        v_cols = list(x_columns)

    rihp = batching_module.RIHPacker(seq_len) if use_rihp and seq_len % 2 == 0 else None
    out_cols = []
    for head_idx in range(attn.num_heads):
        start = head_idx * attn.head_dim
        end = start + attn.head_dim
        out_cols.extend(attn._head_attention_from_columns(q_cols[start:end], k_cols[start:end], v_cols[start:end], seq_len, rihp))

    if attn.out_weight is not None:
        out_cols = attn._pcmm(out_cols, attn.out_weight, attn.out_bias)
    return out_cols


def test_pcmm_matches_torch_matmul(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(0)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_in, d_out = 6, 4, 8

    x_plain = torch.randn(seq_len, d_in, dtype=torch.float64) * 0.1
    w = torch.randn(d_in, d_out, dtype=torch.float64) * 0.1
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=d_out, num_heads=2, normalization_mode="power_softmax")
    y_cols = attn._pcmm(x_cols, w)
    y_plain = _decrypt_columns(ctx, y_cols)

    expected = x_plain @ w
    torch.testing.assert_close(y_plain, expected, rtol=1e-5, atol=1e-5)


def test_pcmm_with_bias(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(1)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_in, d_out = 6, 4, 8

    x_plain = torch.randn(seq_len, d_in, dtype=torch.float64) * 0.1
    w = torch.randn(d_in, d_out, dtype=torch.float64) * 0.1
    b = torch.randn(d_out, dtype=torch.float64) * 0.05
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=d_out, num_heads=2, normalization_mode="power_softmax")
    y_cols = attn._pcmm(x_cols, w, b)
    y_plain = _decrypt_columns(ctx, y_cols)

    expected = x_plain @ w + b
    torch.testing.assert_close(y_plain, expected, rtol=1e-5, atol=1e-5)


def test_forward_stip_columns_no_weights(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(2)
    ctx = _mock_ctx(monkeypatch)
    x_plain = torch.randn(4, 4, dtype=torch.float64) * 0.1
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=4, num_heads=2, normalization_mode="power_softmax")
    out_cols = attn.forward_stip_columns(x_cols, seq_len=4)
    out_plain = _decrypt_columns(ctx, out_cols)

    assert len(out_cols) == 4
    assert torch.isfinite(out_plain).all()
    assert out_plain.abs().max().item() < 10.0


def test_forward_stip_columns_matches_list_attention(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(3)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_model = 4, 4
    x_plain = torch.randn(seq_len, d_model, dtype=torch.float64) * 0.1

    attn = EncryptedApproxAttention(embed_dim=d_model, num_heads=2, normalization_mode="power_softmax")
    attn.q_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.k_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.v_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.out_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.q_bias = torch.randn(d_model, dtype=torch.float64) * 0.01
    attn.k_bias = torch.randn(d_model, dtype=torch.float64) * 0.01
    attn.v_bias = torch.randn(d_model, dtype=torch.float64) * 0.01
    attn.out_bias = torch.randn(d_model, dtype=torch.float64) * 0.01

    x_cols = _encrypt_columns(ctx, x_plain)
    out_cols = attn.forward_stip_columns(x_cols, seq_len=seq_len)
    col_plain = _decrypt_columns(ctx, out_cols)

    tokens = [ctx.encrypt(x_plain[token]) for token in range(seq_len)]
    list_out = attn.forward_attention(tokens, tokens, tokens)
    assert isinstance(list_out, list)
    list_plain = torch.stack([ctx.decrypt(token).to(torch.float64) for token in list_out])

    torch.testing.assert_close(col_plain, list_plain, rtol=5e-2, atol=5e-3)


def test_rihp_used_when_seq_len_even(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(4)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_model = 4, 4
    x_plain = torch.randn(seq_len, d_model, dtype=torch.float64) * 0.1
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=d_model, num_heads=1, normalization_mode="power_softmax")
    attn.q_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.k_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.v_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.out_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1

    out_cols = attn.forward_stip_columns(x_cols, seq_len=seq_len)
    baseline_cols = _baseline_forward_no_dhp(attn, x_cols, seq_len=seq_len, use_rihp=False)

    out_plain = _decrypt_columns(ctx, out_cols)
    baseline_plain = _decrypt_columns(ctx, baseline_cols)
    torch.testing.assert_close(out_plain, baseline_plain, rtol=1e-4, atol=1e-4)


def test_dhp_projection_used_when_heads_even(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(5)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_model = 4, 4
    x_plain = torch.randn(seq_len, d_model, dtype=torch.float64) * 0.1
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=d_model, num_heads=2, normalization_mode="power_softmax")
    attn.q_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.k_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.v_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.out_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1

    out_cols = attn.forward_stip_columns(x_cols, seq_len=seq_len)
    baseline_cols = _baseline_forward_no_dhp(attn, x_cols, seq_len=seq_len, use_rihp=True)

    out_plain = _decrypt_columns(ctx, out_cols)
    baseline_plain = _decrypt_columns(ctx, baseline_cols)
    torch.testing.assert_close(out_plain, baseline_plain, rtol=1e-4, atol=1e-4)


def test_forward_stip_columns_odd_seq_len(monkeypatch: pytest.MonkeyPatch):
    from cukks.nn import EncryptedApproxAttention

    torch.manual_seed(6)
    ctx = _mock_ctx(monkeypatch)
    seq_len, d_model = 3, 4
    x_plain = torch.randn(seq_len, d_model, dtype=torch.float64) * 0.1
    x_cols = _encrypt_columns(ctx, x_plain)

    attn = EncryptedApproxAttention(embed_dim=d_model, num_heads=1, normalization_mode="power_softmax")
    attn.q_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.k_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.v_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1
    attn.out_weight = torch.randn(d_model, d_model, dtype=torch.float64) * 0.1

    out_cols = attn.forward_stip_columns(x_cols, seq_len=seq_len)
    baseline_cols = _baseline_forward_no_dhp(attn, x_cols, seq_len=seq_len, use_rihp=False)

    out_plain = _decrypt_columns(ctx, out_cols)
    baseline_plain = _decrypt_columns(ctx, baseline_cols)
    torch.testing.assert_close(out_plain, baseline_plain, rtol=1e-4, atol=1e-4)
