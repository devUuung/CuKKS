"""Tests for residual connection support."""

import pytest
import torch

from cukks.nn import (
    EncryptedLinear,
    EncryptedResidualBlock,
    EncryptedSequential,
    EncryptedSquare,
)


class TestTensorAddition:
    """Test EncryptedTensor cipher-cipher addition."""

    def test_tensor_addition(self, mock_enc_context):
        """Test enc_tensor + enc_tensor2 with same shape."""
        x1 = torch.randn(1, 64)
        x2 = torch.randn(1, 64)

        enc_x1 = mock_enc_context.encrypt(x1)
        enc_x2 = mock_enc_context.encrypt(x2)

        # Cipher-cipher addition
        enc_sum = enc_x1.add(enc_x2)

        # Decrypt and verify
        result = mock_enc_context.decrypt(enc_sum)
        expected = x1 + x2

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-5)

    def test_tensor_addition_different_values(self, mock_enc_context):
        """Test addition with known values."""
        x1 = torch.tensor([[1.0, 2.0, 3.0]])
        x2 = torch.tensor([[4.0, 5.0, 6.0]])

        enc_x1 = mock_enc_context.encrypt(x1)
        enc_x2 = mock_enc_context.encrypt(x2)

        enc_sum = enc_x1.add(enc_x2)
        result = mock_enc_context.decrypt(enc_sum)

        expected = torch.tensor([[5.0, 7.0, 9.0]])
        assert torch.allclose(result, expected, atol=1e-5)


class TestResidualBlockSimple:
    """Test EncryptedResidualBlock without downsample."""

    def test_residual_block_simple(self, mock_enc_context):
        """Test basic residual: out = block(x) + x."""
        in_features = 64

        # Create a simple linear block (in=out for identity skip)
        weight = torch.randn(in_features, in_features)
        linear = EncryptedLinear(in_features, in_features, weight)

        # Wrap in residual block
        residual = EncryptedResidualBlock(linear)

        # Input
        x = torch.randn(1, in_features)
        enc_x = mock_enc_context.encrypt(x)

        # Forward
        enc_out = residual(enc_x)
        result = mock_enc_context.decrypt(enc_out)

        # Expected: linear(x) + x
        linear_out = x @ weight.T
        expected = linear_out + x

        assert result.numel() == expected.numel()
        assert torch.allclose(result.flatten(), expected.flatten(), atol=1e-5)

    def test_residual_pattern_enc_linear_plus_input(self, mock_enc_context):
        """Test enc_output = enc_linear(enc_input) + enc_input pattern."""
        in_features = 32

        weight = torch.randn(in_features, in_features)
        enc_linear = EncryptedLinear(in_features, in_features, weight)

        x = torch.randn(1, in_features)
        enc_input = mock_enc_context.encrypt(x)

        # The pattern: enc_output = enc_linear(enc_input) + enc_input
        enc_linear_out = enc_linear(enc_input)
        enc_output = enc_linear_out.add(enc_input)

        result = mock_enc_context.decrypt(enc_output)
        expected = (x @ weight.T) + x

        assert torch.allclose(result, expected, atol=1e-5)

    def test_residual_block_mult_depth(self):
        """Test mult_depth calculation for residual block."""
        weight = torch.randn(64, 64)
        linear = EncryptedLinear(64, 64, weight)  # depth 1

        residual = EncryptedResidualBlock(linear)

        # Addition is free, so depth = block depth
        assert residual.mult_depth() == 1

    def test_residual_block_repr(self):
        """Test string representation."""
        weight = torch.randn(64, 64)
        linear = EncryptedLinear(64, 64, weight)
        residual = EncryptedResidualBlock(linear)

        repr_str = repr(residual)
        assert "EncryptedResidualBlock" in repr_str
        assert "has_downsample=False" in repr_str


class TestResidualBlockWithDownsample:
    """Test EncryptedResidualBlock with downsample projection."""

    def test_residual_block_with_downsample(self, mock_enc_context):
        """Test residual with dimension change: out = block(x) + downsample(x)."""
        in_features = 64
        out_features = 128

        # Main block: 64 -> 128
        block_weight = torch.randn(out_features, in_features)
        block = EncryptedLinear(in_features, out_features, block_weight)

        # Downsample: 64 -> 128 (projection for skip connection)
        downsample_weight = torch.randn(out_features, in_features)
        downsample = EncryptedLinear(in_features, out_features, downsample_weight)

        # Residual block
        residual = EncryptedResidualBlock(block, downsample)

        # Input
        x = torch.randn(1, in_features)
        enc_x = mock_enc_context.encrypt(x)

        # Forward
        enc_out = residual(enc_x)
        result = mock_enc_context.decrypt(enc_out)

        # Expected: block(x) + downsample(x)
        block_out = x @ block_weight.T
        downsample_out = x @ downsample_weight.T
        expected = block_out + downsample_out

        assert result.numel() == expected.numel()
        assert result.numel() == out_features
        assert torch.allclose(result.flatten(), expected.flatten(), atol=1e-5)

    def test_residual_with_downsample_mult_depth(self):
        """Test mult_depth with downsample (takes max of both paths)."""
        # Block with depth 2 (linear + square)
        weight1 = torch.randn(64, 64)
        block = EncryptedSequential(
            EncryptedLinear(64, 64, weight1),  # depth 1
            EncryptedSquare(),  # depth 1
        )

        # Downsample with depth 1
        downsample_weight = torch.randn(64, 64)
        downsample = EncryptedLinear(64, 64, downsample_weight)

        residual = EncryptedResidualBlock(block, downsample)

        # Should be max(2, 1) = 2
        assert residual.mult_depth() == 2

    def test_residual_with_downsample_repr(self):
        """Test repr shows downsample present."""
        weight = torch.randn(128, 64)
        block = EncryptedLinear(64, 128, weight)

        downsample_weight = torch.randn(128, 64)
        downsample = EncryptedLinear(64, 128, downsample_weight)

        residual = EncryptedResidualBlock(block, downsample)

        repr_str = repr(residual)
        assert "has_downsample=True" in repr_str

    def test_residual_block_children(self):
        """Test that block and downsample are registered as children."""
        weight = torch.randn(64, 64)
        block = EncryptedLinear(64, 64, weight)

        downsample_weight = torch.randn(64, 64)
        downsample = EncryptedLinear(64, 64, downsample_weight)

        residual = EncryptedResidualBlock(block, downsample)

        children = list(residual.children())
        assert len(children) == 2
        assert block in children
        assert downsample in children
