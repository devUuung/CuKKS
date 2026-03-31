from importlib import import_module
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

EncryptedAdaptiveAvgPool2d = import_module("cukks.nn.adaptive_pooling").EncryptedAdaptiveAvgPool2d


def _make_cnn_tensor(mock_enc_context: Any, image: torch.Tensor):
    flat = image.reshape(-1)
    enc = mock_enc_context.encrypt(flat)
    _, height, width = image.shape
    channels = image.shape[0]
    enc._shape = (height * width, channels)
    enc._cnn_layout = {
        "num_patches": height * width,
        "patch_features": channels,
        "height": height,
        "width": width,
    }
    return enc


def test_from_torch():
    torch_pool = nn.AdaptiveAvgPool2d((1, 1))

    enc_pool = EncryptedAdaptiveAvgPool2d.from_torch(torch_pool)

    assert enc_pool.output_size == (1, 1)


def test_from_torch_int_output():
    torch_pool = nn.AdaptiveAvgPool2d(1)

    enc_pool = EncryptedAdaptiveAvgPool2d.from_torch(torch_pool)

    assert enc_pool.output_size == (1, 1)


def test_output_size_global(mock_enc_context: Any):
    image = torch.arange(49, dtype=torch.float32).view(1, 7, 7)
    enc = _make_cnn_tensor(mock_enc_context, image)
    pool = EncryptedAdaptiveAvgPool2d((1, 1))

    result = pool(enc)
    decrypted = mock_enc_context.decrypt(result, shape=(1, 1))

    assert result._cnn_layout["num_patches"] == 1
    torch.testing.assert_close(decrypted, image.mean().view(1, 1))


def test_output_size_general(mock_enc_context: Any):
    image = torch.arange(49, dtype=torch.float32).view(1, 7, 7)
    enc = _make_cnn_tensor(mock_enc_context, image)
    pool = EncryptedAdaptiveAvgPool2d((3, 3))

    result = pool(enc)
    decrypted = mock_enc_context.decrypt(result, shape=(9, 1)).view(1, 3, 3)
    expected = F.adaptive_avg_pool2d(image, (3, 3))

    assert result._cnn_layout["num_patches"] == 9
    assert result._cnn_layout["patch_features"] == 1
    torch.testing.assert_close(decrypted, expected)


def test_mult_depth():
    assert EncryptedAdaptiveAvgPool2d((3, 3)).mult_depth() == 1


def test_kernel_stride_computation():
    kernel_h, stride_h = EncryptedAdaptiveAvgPool2d._compute_kernel_stride(7, 3)
    kernel_w, stride_w = EncryptedAdaptiveAvgPool2d._compute_kernel_stride(7, 3)

    assert kernel_h == 3
    assert stride_h == 2
    assert kernel_w == 3
    assert stride_w == 2
