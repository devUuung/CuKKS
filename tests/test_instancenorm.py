import importlib

import torch
import torch.nn as nn

_instancenorm = importlib.import_module("cukks.nn.instancenorm")
EncryptedInstanceNorm1d = _instancenorm.EncryptedInstanceNorm1d
EncryptedInstanceNorm2d = _instancenorm.EncryptedInstanceNorm2d


def test_instancenorm_from_torch_1d():
    module = nn.InstanceNorm1d(4, eps=1e-4, affine=False, track_running_stats=False)
    encrypted = EncryptedInstanceNorm1d.from_torch(module)

    assert encrypted.num_features == 4
    assert encrypted.num_groups == 4
    assert encrypted.num_channels == 4
    assert encrypted.eps == 1e-4


def test_instancenorm_from_torch_2d():
    module = nn.InstanceNorm2d(5, eps=1e-3, affine=False, track_running_stats=False)
    encrypted = EncryptedInstanceNorm2d.from_torch(module)

    assert encrypted.num_features == 5
    assert encrypted.num_groups == 5
    assert encrypted.num_channels == 5
    assert encrypted.eps == 1e-3


def test_instancenorm_affine_params():
    module = nn.InstanceNorm2d(3, affine=True)
    encrypted = EncryptedInstanceNorm2d.from_torch(module)

    assert encrypted.affine is True
    assert torch.allclose(encrypted.weight, module.weight.detach().to(torch.float64))
    assert torch.allclose(encrypted.bias, module.bias.detach().to(torch.float64))


def test_instancenorm_no_affine():
    encrypted = EncryptedInstanceNorm1d(3, affine=False)

    assert encrypted.affine is False
    assert torch.allclose(encrypted.weight, torch.ones(3, dtype=torch.float64))
    assert torch.allclose(encrypted.bias, torch.zeros(3, dtype=torch.float64))


def test_instancenorm_mult_depth():
    assert EncryptedInstanceNorm1d(4).mult_depth() == 18
    assert EncryptedInstanceNorm2d(4).mult_depth() == 18
