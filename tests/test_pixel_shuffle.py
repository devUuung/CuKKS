import torch
import torch.nn as nn
from importlib import import_module
from typing import Any

pixel_shuffle_module = import_module("cukks.nn.pixel_shuffle")
EncryptedPixelShuffle = pixel_shuffle_module.EncryptedPixelShuffle
EncryptedPixelUnshuffle = pixel_shuffle_module.EncryptedPixelUnshuffle


def _chw_to_patches(x: torch.Tensor) -> torch.Tensor:
    channels, height, width = x.shape
    return x.permute(1, 2, 0).reshape(height * width, channels)


def _patches_to_chw(x: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    return x.view(height, width, channels).permute(2, 0, 1)


class TestEncryptedPixelShuffle:
    def test_from_torch_shuffle(self):
        module = nn.PixelShuffle(2)

        encrypted = EncryptedPixelShuffle.from_torch(module)

        assert encrypted.upscale_factor == 2

    def test_from_torch_unshuffle(self):
        module = nn.PixelUnshuffle(2)

        encrypted = EncryptedPixelUnshuffle.from_torch(module)

        assert encrypted.downscale_factor == 2

    def test_output_channels(self, mock_enc_context: Any):
        module = EncryptedPixelShuffle(2)
        plain = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
        x = mock_enc_context.encrypt(_chw_to_patches(plain))
        x._cnn_layout = {
            "num_patches": 4,
            "num_patches_per_image": 4,
            "patch_features": 4,
            "height": 2,
            "width": 2,
        }

        result = module(x)

        assert result.shape == (16, 1)
        assert result._cnn_layout["patch_features"] == 1

    def test_output_spatial(self, mock_enc_context: Any):
        module = EncryptedPixelShuffle(2)
        plain = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
        x = mock_enc_context.encrypt(_chw_to_patches(plain))
        x._cnn_layout = {
            "num_patches": 4,
            "num_patches_per_image": 4,
            "patch_features": 4,
            "height": 2,
            "width": 2,
        }

        result = module(x)

        assert result._cnn_layout["height"] == 4
        assert result._cnn_layout["width"] == 4
        decrypted = mock_enc_context.decrypt(result)
        actual = _patches_to_chw(decrypted, channels=1, height=4, width=4)
        expected = nn.PixelShuffle(2)(plain)
        assert torch.allclose(actual, expected)

    def test_inverse_relationship(self, mock_enc_context: Any):
        shuffle = EncryptedPixelShuffle(2)
        unshuffle = EncryptedPixelUnshuffle(2)
        plain = torch.arange(32, dtype=torch.float32).view(8, 2, 2)
        x = mock_enc_context.encrypt(_chw_to_patches(plain))
        x._cnn_layout = {
            "num_patches": 4,
            "num_patches_per_image": 4,
            "patch_features": 8,
            "height": 2,
            "width": 2,
        }

        shuffled = shuffle(x)
        restored = unshuffle(shuffled)

        decrypted = mock_enc_context.decrypt(restored)
        actual = _patches_to_chw(decrypted, channels=8, height=2, width=2)
        assert torch.allclose(actual, plain)
