from typing import Any

import importlib

import torch
import torch.nn as nn

EncryptedConvTranspose2d = importlib.import_module("cukks.nn.conv_transpose").EncryptedConvTranspose2d


def test_from_torch():
    module = nn.ConvTranspose2d(16, 3, 3)
    module.eval()

    enc_module = EncryptedConvTranspose2d.from_torch(module)

    assert enc_module.in_channels == 16
    assert enc_module.out_channels == 3
    assert enc_module.kernel_size == (3, 3)
    assert enc_module.weight_matrix.dtype == torch.float64
    assert enc_module.weight_matrix.device.type == "cpu"


def test_weight_matrix_shape():
    module = nn.ConvTranspose2d(16, 3, 3)

    enc_module = EncryptedConvTranspose2d.from_torch(module)

    assert enc_module.weight_matrix.shape == (3 * 3 * 3, 16)


def test_output_size():
    module = EncryptedConvTranspose2d(
        in_channels=4,
        out_channels=2,
        kernel_size=3,
        weight=torch.randn(4, 2, 3, 3),
        stride=2,
        padding=1,
        output_padding=1,
    )

    out_h, out_w = module.get_output_size(7, 7)

    assert out_h == 14
    assert out_w == 14


def test_mult_depth():
    module = EncryptedConvTranspose2d(
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        weight=torch.randn(2, 1, 3, 3),
    )

    assert module.mult_depth() == 1


def test_upsample_effect(mock_enc_context: Any):
    weight = torch.ones(1, 1, 3, 3)
    module = EncryptedConvTranspose2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        weight=weight,
        stride=2,
        padding=1,
        output_padding=1,
    )

    x = torch.arange(49, dtype=torch.float32).reshape(49, 1)
    enc_x = mock_enc_context.encrypt(x)
    enc_x._cnn_layout = {
        "num_patches": 49,
        "num_patches_per_image": 49,
        "patch_features": 1,
        "batch_size": 1,
        "height": 7,
        "width": 7,
    }
    enc_x._shape = (49, 1)

    out = module(enc_x)
    dec = mock_enc_context.decrypt(out, shape=out.shape)

    assert out.shape == (14 * 14, 1)
    assert out._cnn_layout["num_patches"] == 14 * 14
    assert out._cnn_layout["patch_features"] == 1
    assert out._cnn_layout["height"] == 14
    assert out._cnn_layout["width"] == 14
    assert dec.reshape(14, 14).abs().sum().item() > 0
