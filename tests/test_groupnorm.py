import importlib
from typing import Any

import pytest
import torch
import torch.nn as nn

EncryptedGroupNorm = importlib.import_module("cukks.nn.groupnorm").EncryptedGroupNorm


def _fixed_stats(num_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_channels, dtype=torch.float64),
        torch.ones(num_channels, dtype=torch.float64),
    )


def test_groupnorm_from_torch() -> None:
    module = nn.GroupNorm(2, 6, eps=1e-4)
    encrypted = EncryptedGroupNorm.from_torch(module)

    assert encrypted.num_groups == 2
    assert encrypted.num_channels == 6
    assert encrypted.eps == pytest.approx(1e-4)
    assert torch.allclose(encrypted.group_mean, torch.zeros(6, dtype=torch.float64))
    assert torch.allclose(encrypted.group_inv_std, torch.ones(6, dtype=torch.float64))
    assert torch.allclose(encrypted.weight, module.weight.detach().to(torch.float64))
    assert torch.allclose(encrypted.bias, module.bias.detach().to(torch.float64))


def test_groupnorm_channels_per_group() -> None:
    group_mean, group_inv_std = _fixed_stats(8)
    encrypted = EncryptedGroupNorm(4, 8, group_mean=group_mean, group_inv_std=group_inv_std)
    assert encrypted.channels_per_group == 2


def test_groupnorm_indivisible_channels_raises() -> None:
    group_mean, group_inv_std = _fixed_stats(8)
    with pytest.raises(ValueError, match="divisible"):
        EncryptedGroupNorm(3, 8, group_mean=group_mean, group_inv_std=group_inv_std)


def test_groupnorm_mult_depth() -> None:
    group_mean, group_inv_std = _fixed_stats(6)
    encrypted = EncryptedGroupNorm(2, 6, group_mean=group_mean, group_inv_std=group_inv_std)
    assert encrypted.mult_depth() == 1


def test_groupnorm_weight_bias_shape() -> None:
    group_mean, group_inv_std = _fixed_stats(6)
    encrypted = EncryptedGroupNorm(2, 6, group_mean=group_mean, group_inv_std=group_inv_std)

    assert encrypted.weight.shape == (6,)
    assert encrypted.bias.shape == (6,)
    assert encrypted.weight.dtype == torch.float64
    assert encrypted.bias.dtype == torch.float64
    assert encrypted.weight.device.type == "cpu"
    assert encrypted.bias.device.type == "cpu"


def test_layout_tensors_expand_affine_patterns() -> None:
    group_mean, group_inv_std = _fixed_stats(4)
    encrypted = EncryptedGroupNorm(
        2,
        4,
        group_mean=group_mean,
        group_inv_std=group_inv_std,
        weight=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        bias=torch.tensor([5.0, 6.0, 7.0, 8.0]),
    )

    weight_pattern, bias_pattern = encrypted._layout_tensors((2, 4), 1)

    assert weight_pattern == [1.0, 2.0, 3.0, 4.0] * 2
    assert bias_pattern == [5.0, 6.0, 7.0, 8.0] * 2


def test_forward_applies_fixed_channel_statistics(mock_enc_context: Any) -> None:
    group_mean = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float64)
    group_inv_std = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    weight = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    bias = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float64)
    encrypted = EncryptedGroupNorm(
        2,
        4,
        group_mean=group_mean,
        group_inv_std=group_inv_std,
        weight=weight,
        bias=bias,
    )

    x_plain = torch.tensor([[2.0, 4.0, 12.0, 22.0]], dtype=torch.float64)
    x = mock_enc_context.encrypt(x_plain.reshape(-1)).view(1, 4)
    result = encrypted(x)

    decrypted = mock_enc_context.decrypt(result, shape=(1, 4)).to(torch.float64)
    expected = ((x_plain - group_mean) * group_inv_std) * weight + bias
    torch.testing.assert_close(decrypted, expected)
