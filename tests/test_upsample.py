import torch
import torch.nn as nn
from importlib import import_module
from typing import Any

EncryptedUpsample = import_module("cukks.nn.upsample").EncryptedUpsample


class TestEncryptedUpsample:
    def test_from_torch_nearest(self):
        module = nn.Upsample(scale_factor=2, mode="nearest")

        encrypted = EncryptedUpsample.from_torch(module)

        assert encrypted.scale_factor == 2.0
        assert encrypted.mode == "nearest"

    def test_from_torch_bilinear(self):
        module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        encrypted = EncryptedUpsample.from_torch(module)

        assert encrypted.scale_factor == 2.0
        assert encrypted.mode == "bilinear"
        assert encrypted.align_corners is False

    def test_from_torch_with_size(self):
        module = nn.Upsample(size=(5, 7), mode="nearest")

        encrypted = EncryptedUpsample.from_torch(module)

        assert encrypted.size == (5, 7)
        assert encrypted.scale_factor is None

    def test_mult_depth(self):
        encrypted = EncryptedUpsample(scale_factor=2, mode="nearest")

        assert encrypted.mult_depth() == 1

    def test_output_shape(self, mock_enc_context: Any):
        encrypted = EncryptedUpsample(scale_factor=2, mode="nearest")
        x = mock_enc_context.encrypt(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        x._cnn_layout = {
            "num_patches": 4,
            "num_patches_per_image": 4,
            "patch_features": 1,
            "height": 2,
            "width": 2,
        }

        result = encrypted(x)

        assert result.shape == (16, 1)
        assert result._cnn_layout is not None
        assert result._cnn_layout["num_patches"] == 16
        assert result._cnn_layout["height"] == 4
        assert result._cnn_layout["width"] == 4

        decrypted = mock_enc_context.decrypt(result)
        expected = torch.tensor(
            [[1.0], [1.0], [2.0], [2.0],
             [1.0], [1.0], [2.0], [2.0],
             [3.0], [3.0], [4.0], [4.0],
             [3.0], [3.0], [4.0], [4.0]]
        )
        assert torch.allclose(decrypted, expected)
