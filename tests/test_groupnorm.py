import importlib
from typing import Any

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


def test_groupnorm_layout_uses_compact_descriptor_for_contiguous_groups():
    encrypted = EncryptedGroupNorm(4, 2048)

    avg_block, _, _ = encrypted._layout_tensors((2048,), 0)

    assert avg_block.shape == (1, 1)
    assert avg_block.item() == pytest.approx(1.0 / 512.0)


def test_group_matmul_broadcasts_contiguous_group_means(mock_enc_context: Any):
    encrypted = EncryptedGroupNorm(2, 6)
    avg_block, _, _ = encrypted._layout_tensors((6,), 0)

    x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=torch.float64)).view(1, 6)
    result = encrypted._group_matmul(x, avg_block)

    decrypted = mock_enc_context.decrypt(result, shape=(1, 6)).view(-1)
    expected = torch.tensor([2.0, 2.0, 2.0, 20.0, 20.0, 20.0], dtype=torch.float32)
    torch.testing.assert_close(decrypted, expected)


def test_group_matmul_falls_back_for_noncontiguous_channel_last_layout(mock_enc_context: Any):
    encrypted = EncryptedGroupNorm(2, 4)
    avg_block, _, _ = encrypted._layout_tensors((2, 4), 1)

    x = mock_enc_context.encrypt(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=torch.float64)
    ).view(1, 8)
    result = encrypted._group_matmul(x, avg_block)

    decrypted = mock_enc_context.decrypt(result, shape=(1, 8)).view(-1)
    expected = torch.tensor([8.25, 8.25, 19.25, 19.25, 8.25, 8.25, 19.25, 19.25], dtype=torch.float32)
    torch.testing.assert_close(decrypted, expected)
