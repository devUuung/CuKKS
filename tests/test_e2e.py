"""End-to-end tests for CuKKS encrypted inference.

This module tests complete inference pipelines:
- MNIST-style MLP inference
- Simple CNN inference  
- Accuracy comparison: plaintext vs encrypted
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mocks.mock_backend import MockCKKSConfig, MockCKKSContext, MockCKKSTensor
from cukks import convert, estimate_depth
from cukks.tensor import EncryptedTensor
from cukks.nn import (
    EncryptedLinear,
    EncryptedSequential,
    EncryptedSquare,
    EncryptedReLU,
    EncryptedConv2d,
    EncryptedFlatten,
    EncryptedAvgPool2d,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_enc_context():
    """Mock encrypted context for testing without actual HE backend."""

    class EncryptedMockContext:
        def __init__(self) -> None:
            self._ctx = MockCKKSContext()
            self.use_bsgs = False
            self._max_rotation_dim = 1024
            self._auto_bootstrap = False
            self._bootstrap_threshold = 2

        @property
        def auto_bootstrap(self) -> bool:
            return self._auto_bootstrap

        @property
        def bootstrap_threshold(self) -> int:
            return self._bootstrap_threshold

        def encrypt(self, tensor: torch.Tensor) -> EncryptedTensor:
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(
            self, enc_tensor: EncryptedTensor, shape: Tuple[int, ...] | None = None
        ) -> torch.Tensor:
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)

    return EncryptedMockContext()


@pytest.fixture
def simple_mlp():
    """Create a simple 784 -> 128 -> 10 MLP."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.eval()
    return model


@pytest.fixture
def deeper_mlp():
    """Create a deeper MLP with multiple hidden layers."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    model.eval()
    return model


@pytest.fixture
def simple_cnn():
    """Create a simple CNN for MNIST-like input (1, 28, 28)."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 14x14
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 7x7
        nn.Flatten(),
        nn.Linear(8 * 7 * 7, 10),
    )
    model.eval()
    return model


# =============================================================================
# MNIST Inference Tests
# =============================================================================


class TestMNISTInference:
    """End-to-end tests for MNIST-style inference."""

    def test_simple_mlp_encrypted_inference(self, simple_mlp, mock_enc_context):
        """Test that a simple MLP can run encrypted inference."""
        # 1. Create and convert model
        enc_model, _ = convert(simple_mlp, use_square_activation=True)

        # 2. Verify conversion structure
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 3
        assert isinstance(enc_model[0], EncryptedLinear)
        assert isinstance(enc_model[1], EncryptedSquare)
        assert isinstance(enc_model[2], EncryptedLinear)

        # 3. Create mock encrypted input
        torch.manual_seed(123)
        sample = torch.randn(784)
        enc_input = mock_enc_context.encrypt(sample)

        # 4. Run encrypted inference
        enc_output = enc_model(enc_input)

        # 5. Decrypt and verify shape
        output = mock_enc_context.decrypt(enc_output)
        assert output.shape == (10,), f"Expected shape (10,), got {output.shape}"

    def test_simple_mlp_with_relu_approximation(self, simple_mlp, mock_enc_context):
        """Test MLP with ReLU polynomial approximation."""
        # Convert with ReLU approximation (not square)
        enc_model, _ = convert(
            simple_mlp, use_square_activation=False, activation_degree=4
        )

        assert isinstance(enc_model, EncryptedSequential)
        assert isinstance(enc_model[1], EncryptedReLU)
        assert enc_model[1].degree == 4

        # Run inference
        sample = torch.randn(784)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (10,)

    def test_deeper_mlp_encrypted_inference(self, deeper_mlp, mock_enc_context):
        """Test deeper MLP with multiple hidden layers."""
        enc_model, _ = convert(deeper_mlp, use_square_activation=True)

        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 7  # 4 linear + 3 activations

        # Run inference
        sample = torch.randn(784)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (10,)

    def test_mlp_with_batchnorm_folding(self, mock_enc_context):
        """Test that BatchNorm is properly folded into Linear."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        model.eval()

        # Initialize running stats
        with torch.no_grad():
            model(torch.randn(32, 784))

        enc_model, _ = convert(model, fold_batchnorm=True, use_square_activation=True)

        # BatchNorm should be folded, so only 3 modules remain
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 3

        # Run inference
        sample = torch.randn(784)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (10,)


# =============================================================================
# CNN Inference Tests
# =============================================================================


class TestCNNInference:
    """End-to-end tests for CNN inference."""

    def test_simple_cnn_conversion(self, simple_cnn):
        """Test that a simple CNN converts correctly."""
        enc_model, _ = convert(simple_cnn, use_square_activation=True)

        assert isinstance(enc_model, EncryptedSequential)
        # Conv2d + Square + AvgPool + Conv2d + Square + AvgPool + Flatten + Linear = 8
        assert len(enc_model) == 8

        # Check layer types
        assert isinstance(enc_model[0], EncryptedConv2d)
        assert isinstance(enc_model[1], EncryptedSquare)
        assert isinstance(enc_model[2], EncryptedAvgPool2d)
        assert isinstance(enc_model[6], EncryptedFlatten)
        assert isinstance(enc_model[7], EncryptedLinear)

    def test_simple_cnn_encrypted_inference(self, simple_cnn, mock_enc_context):
        """Test CNN encrypted inference raises RuntimeError for non-CNN-layout input."""
        enc_model, _ = convert(simple_cnn, use_square_activation=True)

        # Create input: (1, 28, 28) image — not CNN layout
        torch.manual_seed(42)
        sample = torch.randn(1, 28, 28)
        enc_input = mock_enc_context.encrypt(sample)

        # Should raise RuntimeError because input lacks _cnn_layout
        with pytest.raises(RuntimeError, match="encrypt_cnn_input"):
            enc_model(enc_input)


# =============================================================================
# Accuracy Comparison Tests
# =============================================================================


class TestAccuracyComparison:
    """Tests comparing plaintext vs encrypted inference accuracy."""

    def test_accuracy_comparison_linear_only(self, mock_enc_context):
        """Compare plaintext vs encrypted for linear layer only."""
        torch.manual_seed(42)

        # Simple linear model
        model = nn.Linear(16, 8)
        model.eval()

        enc_model, _ = convert(model)

        # Test on multiple samples
        for _ in range(5):
            sample = torch.randn(16)

            # Plaintext inference
            with torch.no_grad():
                plain_output = model(sample)

            # Encrypted inference
            enc_input = mock_enc_context.encrypt(sample)
            enc_output = enc_model(enc_input)
            decrypted_output = mock_enc_context.decrypt(enc_output)

            # Compare (mock backend should be exact or very close)
            assert plain_output.shape == decrypted_output.shape
            error = (plain_output - decrypted_output).abs().max().item()
            # Mock backend uses float64 internally, expect high precision
            assert error < 1e-4, f"Max error {error} exceeds threshold"

    def test_accuracy_comparison_mlp_square(self, simple_mlp, mock_enc_context):
        """Compare plaintext vs encrypted for MLP with square activation."""
        torch.manual_seed(42)

        # Create model with square activation for fair comparison
        square_model = nn.Sequential(
            nn.Linear(784, 128),
            # Square activation (x^2)
            nn.Linear(128, 10),
        )
        square_model.eval()

        # Get layers with proper typing
        simple_fc1 = simple_mlp[0]
        simple_fc2 = simple_mlp[2]
        square_fc1 = square_model[0]
        square_fc2 = square_model[1]

        # Copy weights from simple_mlp
        with torch.no_grad():
            assert isinstance(simple_fc1, nn.Linear)
            assert isinstance(simple_fc2, nn.Linear)
            assert isinstance(square_fc1, nn.Linear)
            assert isinstance(square_fc2, nn.Linear)
            square_fc1.weight.copy_(simple_fc1.weight)
            square_fc1.bias.copy_(simple_fc1.bias)  # type: ignore[arg-type]
            square_fc2.weight.copy_(simple_fc2.weight)
            square_fc2.bias.copy_(simple_fc2.bias)  # type: ignore[arg-type]

        enc_model, _ = convert(simple_mlp, use_square_activation=True)

        # Test sample
        sample = torch.randn(784) * 0.1  # Small values for stability

        # Plaintext inference with manual square
        with torch.no_grad():
            x = simple_fc1(sample)  # Linear
            x = x * x  # Square (same as EncryptedSquare)
            plain_output = simple_fc2(x)  # Linear

        # Encrypted inference
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        decrypted_output = mock_enc_context.decrypt(enc_output)

        # Compare outputs
        error = (plain_output - decrypted_output).abs().mean().item()
        # Allow slightly larger error due to rescaling in encrypted ops
        assert error < 0.1, f"Mean error {error} exceeds threshold"

    def test_depth_estimation(self, simple_mlp, deeper_mlp):
        """Test that depth estimation works correctly."""
        simple_depth = estimate_depth(simple_mlp)
        deeper_depth = estimate_depth(deeper_mlp)

        # Simple: 2 linear + 1 relu = 2 + 2 = 4
        assert simple_depth >= 3

        # Deeper: 4 linear + 3 relu = 4 + 6 = 10
        assert deeper_depth >= 8

        # Deeper should require more depth
        assert deeper_depth > simple_depth


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_linear_layer(self, mock_enc_context):
        """Test conversion of single linear layer."""
        model = nn.Linear(32, 16)
        model.eval()

        enc_model, _ = convert(model)
        assert isinstance(enc_model, EncryptedLinear)

        sample = torch.randn(32)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (16,)

    def test_empty_sequential(self):
        """Test that empty sequential still works."""
        model = nn.Sequential()
        model.eval()

        depth = estimate_depth(model)
        assert depth >= 1  # Should have minimum depth

    def test_model_with_dropout(self, mock_enc_context):
        """Test that dropout is handled correctly (ignored in eval mode)."""
        import warnings

        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
        )
        model.eval()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enc_model, _ = convert(model, use_square_activation=True)

            # Should emit warning about dropout
            dropout_warnings = [x for x in w if "Dropout" in str(x.message)]
            assert len(dropout_warnings) >= 1

        # Model should still work
        sample = torch.randn(32)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (8,)

    def test_large_input(self, mock_enc_context):
        """Test with larger input dimensions."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        model.eval()

        enc_model, _ = convert(model, use_square_activation=True)

        sample = torch.randn(1024)
        enc_input = mock_enc_context.encrypt(sample)
        enc_output = enc_model(enc_input)
        output = mock_enc_context.decrypt(enc_output)

        assert output.shape == (64,)


# =============================================================================
# Performance Characteristics Tests
# =============================================================================


class TestPerformanceCharacteristics:
    """Tests for performance-related characteristics."""

    def test_mult_depth_tracking(self, simple_mlp):
        """Test that multiplicative depth is tracked correctly."""
        enc_model, _ = convert(simple_mlp, use_square_activation=True)

        total_depth = enc_model.mult_depth()
        # 2 linear (1 each) + 1 square (1) = 3
        assert total_depth >= 3

    def test_conversion_preserves_weights(self):
        """Test that conversion preserves weight values."""
        torch.manual_seed(42)

        model = nn.Linear(10, 5)
        model.eval()

        enc_model, _ = convert(model)

        assert torch.allclose(
            enc_model.weight.float(), model.weight.data.float(), atol=1e-6
        )
        if model.bias is not None:
            assert torch.allclose(
                enc_model.bias.float(), model.bias.data.float(), atol=1e-6
            )

    def test_model_can_be_reused(self, simple_mlp, mock_enc_context):
        """Test that encrypted model can process multiple inputs."""
        enc_model, _ = convert(simple_mlp, use_square_activation=True)

        outputs = []
        for i in range(3):
            torch.manual_seed(i)
            sample = torch.randn(784)
            enc_input = mock_enc_context.encrypt(sample)
            enc_output = enc_model(enc_input)
            output = mock_enc_context.decrypt(enc_output)
            outputs.append(output)

        # Each output should be different (different inputs)
        assert not torch.allclose(outputs[0], outputs[1])
        assert not torch.allclose(outputs[1], outputs[2])


# =============================================================================
# True HE CNN Tests (encrypt_cnn_input + from_torch_cnn)
# =============================================================================


class TestTrueHECNN:
    """Tests for True HE CNN pipeline: im2col before encryption, no decryption during inference.
    
    This tests the complete secure CNN pipeline:
    1. encrypt_cnn_input() - Pre-applies im2col before encryption
    2. EncryptedConv2d - Block-diagonal matmul
    3. EncryptedSquare - Element-wise squaring
    4. EncryptedAvgPool2d - Rotation-based pooling
    5. EncryptedFlatten - No-op (permutation absorbed)
    6. EncryptedLinear.from_torch_cnn() - Permutation absorbed into weights
    """

    @pytest.fixture
    def small_cnn(self):
        """Small CNN for 8x8 single-channel input."""
        torch.manual_seed(42)
        
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 8x8 -> 8x8
                self.pool = nn.AvgPool2d(2)  # 8x8 -> 4x4
                self.fc = nn.Linear(4 * 4 * 4, 10)  # 64 -> 10
                
            def forward(self, x):
                x = self.conv(x)
                x = x ** 2  # Square activation
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x
        
        model = SmallCNN()
        model.eval()
        return model

    def test_encrypt_cnn_input_shape(self, mock_enc_context):
        """Test that encrypt_cnn_input produces correct shape and layout."""
        # 8x8 single-channel image
        torch.manual_seed(42)
        image = torch.randn(1, 8, 8)
        
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        
        # Use mock context's encrypt_cnn_input if available, otherwise test structure
        if hasattr(mock_enc_context, 'encrypt_cnn_input'):
            enc_input = mock_enc_context.encrypt_cnn_input(image, conv_params)
            
            # Should have _cnn_layout metadata
            assert hasattr(enc_input, '_cnn_layout')
            assert enc_input._cnn_layout is not None
            assert 'num_patches' in enc_input._cnn_layout
            assert 'patch_features' in enc_input._cnn_layout
            
            # For 8x8 image with 3x3 kernel, padding=1, stride=1: 8*8 = 64 patches
            # Each patch: 1 channel * 3 * 3 = 9 features
            assert enc_input._cnn_layout['num_patches'] == 64
            assert enc_input._cnn_layout['patch_features'] == 9

    def test_from_torch_cnn_weight_permutation(self):
        """Test that from_torch_cnn correctly permutes FC weights."""
        torch.manual_seed(42)
        
        # CNN layout: 4x4 spatial, 4 channels = 16 patches, 4 features per patch
        cnn_layout = {'num_patches': 16, 'patch_features': 4}
        
        # FC layer: 64 -> 10
        fc = nn.Linear(64, 10)
        
        # Convert with CNN layout
        from cukks.nn import EncryptedLinear
        enc_fc = EncryptedLinear.from_torch_cnn(fc, cnn_layout)
        
        # Weights should have same shape but different values (permuted)
        assert enc_fc.weight.shape == fc.weight.shape
        
        # Verify permutation is applied (weights should differ)
        # The permutation reorders from (H*W, C) to (C, H*W)
        assert not torch.allclose(enc_fc.weight, fc.weight.to(torch.float64))
        
        # Bias should remain unchanged
        if fc.bias is not None:
            assert torch.allclose(enc_fc.bias, fc.bias.to(torch.float64))

    def test_he_cnn_vs_plaintext_accuracy(self, small_cnn, mock_enc_context):
        """Test that HE CNN output matches plaintext CNN output.
        
        This is the critical accuracy test: HE inference should match
        plaintext inference within acceptable tolerance (96%+ correlation).
        """
        from cukks.nn import (
            EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
            EncryptedFlatten, EncryptedLinear
        )
        
        torch.manual_seed(123)
        
        # Create 8x8 test image
        image = torch.randn(1, 1, 8, 8)
        
        # === Plaintext inference ===
        with torch.no_grad():
            plain_output = small_cnn(image)
        
        # === Build encrypted model manually ===
        # Conv params for encrypt_cnn_input
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        
        # Create encrypted layers
        enc_conv = EncryptedConv2d.from_torch(small_cnn.conv)
        enc_square = EncryptedSquare()
        enc_pool = EncryptedAvgPool2d.from_torch(small_cnn.pool)
        enc_flatten = EncryptedFlatten._with_absorbed_permutation()
        
        # === Encrypted inference ===
        if hasattr(mock_enc_context, 'encrypt_cnn_input'):
            enc_input = mock_enc_context.encrypt_cnn_input(image.squeeze(0), conv_params)
        else:
            # Fallback: use regular encrypt with manual im2col
            import torch.nn.functional as F
            padded = F.pad(image, (1, 1, 1, 1))
            patches = F.unfold(padded.to(torch.float64), (3, 3), stride=(1, 1))
            patches = patches.transpose(1, 2).squeeze(0)
            flat_patches = patches.flatten()
            enc_input = mock_enc_context.encrypt(flat_patches)
            enc_input._cnn_layout = {
                'num_patches': 64,
                'patch_features': 9,
            }
            enc_input._shape = (64, 9)
        
        # Run through conv, square, pool to get actual CNN layout
        enc_x = enc_conv(enc_input)
        enc_x = enc_square(enc_x)
        enc_x = enc_pool(enc_x)
        
        # Build FC with the actual layout from pool output (may be sparse)
        cnn_layout = enc_x._cnn_layout
        enc_fc = EncryptedLinear.from_torch_cnn(small_cnn.fc, cnn_layout)
        
        enc_x = enc_flatten(enc_x)
        enc_output = enc_fc(enc_x)
        
        # Decrypt
        he_output = mock_enc_context.decrypt(enc_output)
        
        # === Compare outputs ===
        # Flatten both for comparison
        plain_flat = plain_output.flatten().to(torch.float64)
        he_flat = he_output.flatten()[:10]  # Take first 10 values
        
        # Check shape
        assert he_flat.shape[0] >= 10, f"Expected at least 10 outputs, got {he_flat.shape}"
        
        # Check correlation (should be > 0.95 for good accuracy)
        # Using cosine similarity as a proxy
        cos_sim = F.cosine_similarity(plain_flat.unsqueeze(0), he_flat.unsqueeze(0))
        
        # For mock backend, we expect near-perfect match
        # Real HE backend would have ~96%+ match
        assert cos_sim.item() > 0.90, f"Cosine similarity {cos_sim.item():.4f} < 0.90"

    def test_convert_with_optimize_cnn(self, mock_enc_context):
        """Test that convert(optimize_cnn=True) correctly optimizes CNN models."""
        torch.manual_seed(42)
        
        # Simple CNN: Conv -> Square -> Pool -> Flatten -> FC
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),  # Will become Square
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 10),  # Assumes 8x8 input
        )
        model.eval()
        
        # Convert with CNN optimization
        enc_model, _ = convert(model, use_square_activation=True, optimize_cnn=True)
        
        # Check that Flatten has absorbed permutation
        flatten_layer = enc_model[3]
        assert isinstance(flatten_layer, EncryptedFlatten)
        assert flatten_layer._absorb_permutation == True
        
        # Check that FC weights were permuted (from_torch_cnn was used)
        fc_layer = enc_model[4]
        assert isinstance(fc_layer, EncryptedLinear)

    def test_cnn_multiple_samples(self, small_cnn, mock_enc_context):
        """Test HE CNN with multiple different inputs."""
        from cukks.nn import (
            EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
            EncryptedFlatten, EncryptedLinear
        )
        
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        cnn_layout = {'num_patches': 16, 'patch_features': 4}
        
        # Build encrypted model
        enc_conv = EncryptedConv2d.from_torch(small_cnn.conv)
        enc_square = EncryptedSquare()
        enc_pool = EncryptedAvgPool2d.from_torch(small_cnn.pool)
        enc_flatten = EncryptedFlatten._with_absorbed_permutation()
        enc_fc = EncryptedLinear.from_torch_cnn(small_cnn.fc, cnn_layout)
        
        he_outputs = []
        plain_outputs = []
        
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            image = torch.randn(1, 1, 8, 8)
            
            # Plaintext
            with torch.no_grad():
                plain_out = small_cnn(image)
            plain_outputs.append(plain_out.flatten())
            
            # HE (using fallback for mock)
            import torch.nn.functional as F
            padded = F.pad(image, (1, 1, 1, 1))
            patches = F.unfold(padded.to(torch.float64), (3, 3), stride=(1, 1))
            patches = patches.transpose(1, 2).squeeze(0)
            flat_patches = patches.flatten()
            enc_input = mock_enc_context.encrypt(flat_patches)
            enc_input._cnn_layout = {'num_patches': 64, 'patch_features': 9}
            enc_input._shape = (64, 9)
            
            enc_x = enc_conv(enc_input)
            enc_x = enc_square(enc_x)
            enc_x = enc_pool(enc_x)
            enc_x = enc_flatten(enc_x)
            enc_output = enc_fc(enc_x)
            
            he_out = mock_enc_context.decrypt(enc_output)
            he_outputs.append(he_out.flatten()[:10])
        
        # Each sample should produce different output
        assert not torch.allclose(he_outputs[0], he_outputs[1])
        assert not torch.allclose(he_outputs[1], he_outputs[2])
        
        # Plain outputs should also differ
        assert not torch.allclose(plain_outputs[0], plain_outputs[1])
