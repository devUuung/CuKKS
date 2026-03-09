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


def test_compute_packing_factor_basic():
    import cukks.batching as batching_module

    k = batching_module.AMCPacker.compute_packing_factor(
        num_heads=4,
        seq_len=8,
        batch_size=2,
        total_slots=64,
    )
    assert k == 4


def test_compute_packing_factor_constrained():
    import cukks.batching as batching_module

    k = batching_module.AMCPacker.compute_packing_factor(
        num_heads=8,
        seq_len=16,
        batch_size=4,
        total_slots=128,
    )
    assert k == 2


def test_compute_packing_factor_no_valid():
    import cukks.batching as batching_module

    with pytest.raises(ValueError, match="No valid AMCP packing factor"):
        batching_module.AMCPacker.compute_packing_factor(
            num_heads=4,
            seq_len=8,
            batch_size=2,
            total_slots=7,
        )


def test_build_masks_structure():
    import cukks.batching as batching_module

    packer = batching_module.AMCPacker(
        num_heads=8,
        seq_len=4,
        batch_size=2,
        total_slots=24,
    )
    assert packer.k == 2
    assert packer.stride == 6
    assert packer.width == 4

    u_mask, v_mask = packer.build_masks()
    expected_u_block = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    expected_v_block = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    assert len(u_mask) == packer.total_slots
    assert len(v_mask) == packer.total_slots
    assert u_mask == expected_u_block * packer.seq_len
    assert v_mask == expected_v_block * packer.seq_len


def test_group_wise_rotation_count(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.AMCPacker(
        num_heads=8,
        seq_len=4,
        batch_size=2,
        total_slots=24,
    )
    data = torch.arange(packer.total_slots, dtype=torch.float64)
    ct = ctx.encrypt(data)

    aligned = packer.group_wise_rotation(ct, ctx)

    assert len(aligned) == packer.k


def test_group_wise_rotation_values(monkeypatch: pytest.MonkeyPatch):
    import cukks.batching as batching_module

    ctx = _mock_ctx(monkeypatch)
    packer = batching_module.AMCPacker(
        num_heads=2,
        seq_len=1,
        batch_size=2,
        total_slots=8,
    )
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ct = ctx.encrypt(data)

    aligned = packer.group_wise_rotation(ct, ctx)

    expected_states = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([3.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    ]

    assert len(aligned) == len(expected_states)
    for idx, expected in enumerate(expected_states):
        actual = ctx.decrypt(aligned[idx], shape=(8,)).to(torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_scale_rotation():
    import cukks.batching as batching_module

    packer = batching_module.AMCPacker(
        num_heads=8,
        seq_len=16,
        batch_size=4,
        total_slots=128,
    )
    assert packer.scale_rotation(3) == 24
    assert packer.scale_rotation(-2) == -16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 0, "seq_len": 8, "batch_size": 2, "total_slots": 64},
        {"num_heads": 4, "seq_len": 0, "batch_size": 2, "total_slots": 64},
        {"num_heads": 4, "seq_len": 8, "batch_size": 0, "total_slots": 64},
        {"num_heads": 4, "seq_len": 8, "batch_size": 2, "total_slots": 0},
        {"num_heads": -1, "seq_len": 8, "batch_size": 2, "total_slots": 64},
        {"num_heads": 4, "seq_len": -1, "batch_size": 2, "total_slots": 64},
        {"num_heads": 4, "seq_len": 8, "batch_size": -1, "total_slots": 64},
        {"num_heads": 4, "seq_len": 8, "batch_size": 2, "total_slots": -1},
    ],
)
def test_invalid_params(kwargs: dict[str, int]):
    import cukks.batching as batching_module

    with pytest.raises(ValueError):
        batching_module.AMCPacker(**kwargs)
