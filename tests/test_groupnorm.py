import importlib

import pytest
import torch
import torch.nn as nn

EncryptedGroupNorm = importlib.import_module("cukks.nn.groupnorm").EncryptedGroupNorm


def test_groupnorm_from_torch():
    module = nn.GroupNorm(2, 6, eps=1e-4)
    encrypted = EncryptedGroupNorm.from_torch(module)

    assert encrypted.num_groups == 2
    assert encrypted.num_channels == 6
    assert encrypted.eps == pytest.approx(1e-4)
    assert torch.allclose(encrypted.weight, module.weight.detach().to(torch.float64))
    assert torch.allclose(encrypted.bias, module.bias.detach().to(torch.float64))


def test_groupnorm_channels_per_group():
    encrypted = EncryptedGroupNorm(4, 8)
    assert encrypted.channels_per_group == 2


def test_groupnorm_indivisible_channels_raises():
    with pytest.raises(ValueError, match="divisible"):
        EncryptedGroupNorm(3, 8)


def test_groupnorm_mult_depth():
    assert EncryptedGroupNorm(2, 6).mult_depth() == 18


def test_groupnorm_weight_bias_shape():
    encrypted = EncryptedGroupNorm(2, 6)

    assert encrypted.weight.shape == (6,)
    assert encrypted.bias.shape == (6,)
    assert encrypted.weight.dtype == torch.float64
    assert encrypted.bias.dtype == torch.float64
    assert encrypted.weight.device.type == "cpu"
    assert encrypted.bias.device.type == "cpu"
