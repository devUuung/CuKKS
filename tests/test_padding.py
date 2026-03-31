import torch
import torch.nn as nn
from importlib import import_module
from typing import Any

padding_module = import_module("cukks.nn.padding")
EncryptedZeroPad2d = padding_module.EncryptedZeroPad2d
EncryptedConstantPad2d = padding_module.EncryptedConstantPad2d
EncryptedReflectionPad2d = padding_module.EncryptedReflectionPad2d
EncryptedReplicationPad2d = padding_module.EncryptedReplicationPad2d


def _chw_to_patches(x: torch.Tensor) -> torch.Tensor:
    channels, height, width = x.shape
    return x.permute(1, 2, 0).reshape(height * width, channels)


def _patches_to_chw(x: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    return x.view(height, width, channels).permute(2, 0, 1)


def _encrypt_chw(mock_enc_context: Any, x: torch.Tensor):
    enc = mock_enc_context.encrypt(_chw_to_patches(x))
    _, height, width = x.shape
    enc._cnn_layout = {
        "num_patches": height * width,
        "num_patches_per_image": height * width,
        "patch_features": x.shape[0],
        "height": height,
        "width": width,
    }
    return enc


class TestEncryptedPadding:
    def test_zeropad_from_torch(self):
        module = nn.ZeroPad2d((1, 2, 3, 4))

        encrypted = EncryptedZeroPad2d.from_torch(module)

        assert encrypted.padding == (1, 2, 3, 4)

    def test_zeropad_output_size(self, mock_enc_context: Any):
        plain = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
        encrypted = _encrypt_chw(mock_enc_context, plain)

        result = EncryptedZeroPad2d(1)(encrypted)

        assert result.shape == (16, 1)
        assert result._cnn_layout["height"] == 4
        assert result._cnn_layout["width"] == 4
        actual = _patches_to_chw(mock_enc_context.decrypt(result), 1, 4, 4)
        expected = nn.ZeroPad2d(1)(plain)
        assert torch.allclose(actual, expected)

    def test_constantpad_from_torch(self, mock_enc_context: Any):
        module = nn.ConstantPad2d((1, 1, 0, 2), 3.5)
        encrypted_module = EncryptedConstantPad2d.from_torch(module)
        plain = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        encrypted = _encrypt_chw(mock_enc_context, plain)

        result = encrypted_module(encrypted)

        actual = _patches_to_chw(mock_enc_context.decrypt(result), 1, 4, 4)
        expected = module(plain)
        assert encrypted_module.padding == (1, 1, 0, 2)
        assert encrypted_module.value == 3.5
        assert torch.allclose(actual, expected)

    def test_reflectionpad_from_torch(self, mock_enc_context: Any):
        module = nn.ReflectionPad2d(1)
        encrypted_module = EncryptedReflectionPad2d.from_torch(module)
        plain = torch.arange(1, 10, dtype=torch.float32).view(1, 3, 3)
        encrypted = _encrypt_chw(mock_enc_context, plain)

        result = encrypted_module(encrypted)

        actual = _patches_to_chw(mock_enc_context.decrypt(result), 1, 5, 5)
        expected = module(plain)
        assert encrypted_module.padding == (1, 1, 1, 1)
        assert torch.allclose(actual, expected)

    def test_replicationpad_from_torch(self, mock_enc_context: Any):
        module = nn.ReplicationPad2d((1, 0, 2, 1))
        encrypted_module = EncryptedReplicationPad2d.from_torch(module)
        plain = torch.arange(1, 10, dtype=torch.float32).view(1, 3, 3)
        encrypted = _encrypt_chw(mock_enc_context, plain)

        result = encrypted_module(encrypted)

        actual = _patches_to_chw(mock_enc_context.decrypt(result), 1, 6, 4)
        expected = module(plain)
        assert encrypted_module.padding == (1, 0, 2, 1)
        assert torch.allclose(actual, expected)

    def test_asymmetric_padding(self, mock_enc_context: Any):
        plain = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        )
        encrypted = _encrypt_chw(mock_enc_context, plain)

        result = EncryptedZeroPad2d((2, 1, 1, 0))(encrypted)

        actual = _patches_to_chw(mock_enc_context.decrypt(result), 1, 3, 6)
        expected = nn.ZeroPad2d((2, 1, 1, 0))(plain)
        assert torch.allclose(actual, expected)
