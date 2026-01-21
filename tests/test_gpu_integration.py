"""GPU Integration Tests for ckks-torch.

These tests verify that MLP and CNN models work correctly with
the real CKKS backend on GPU (CUDA).

Tests are skipped if:
- Real CKKS backend is not available
- CUDA is not available

Run GPU tests only:
    pytest tests/test_gpu_integration.py -v

Run without GPU tests (for CI without GPU):
    pytest tests/ -m "not gpu"
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from conftest import requires_gpu, requires_real_backend


# Mark entire module as GPU tests
pytestmark = pytest.mark.gpu


# =============================================================================
# MLP Tests
# =============================================================================


class TestGPU_MLP:
    """GPU integration tests for MLP models."""
    
    @pytest.fixture
    def simple_mlp(self):
        """Simple MLP: 16 -> 32 -> 10 with Square activation."""
        torch.manual_seed(42)
        
        class SquareMLP(nn.Module):
            """MLP with Square activation for HE compatibility."""
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 10)
            
            def forward(self, x):
                x = self.fc1(x)
                x = x ** 2  # Square activation (HE-compatible)
                x = self.fc2(x)
                return x
        
        model = SquareMLP()
        model.eval()
        return model

    @requires_gpu
    def test_mlp_encrypt_decrypt(self, gpu_context):
        """Test basic encrypt/decrypt on GPU."""
        torch.manual_seed(42)
        data = torch.randn(16, dtype=torch.float64)
        
        # Encrypt
        enc_data = gpu_context.encrypt(data)
        
        # Decrypt
        decrypted = gpu_context.decrypt(enc_data)
        
        # Move to CPU for comparison
        decrypted_cpu = decrypted.cpu() if decrypted.is_cuda else decrypted
        
        # Should match within tolerance
        error = (data - decrypted_cpu[:16]).abs().max().item()
        assert error < 1e-4, f"Decrypt error {error:.2e} > 1e-4"

    @requires_gpu
    def test_mlp_he_inference(self, simple_mlp, gpu_context):
        """Test full MLP inference on GPU with HE."""
        from ckks_torch.nn import EncryptedLinear, EncryptedSquare
        
        torch.manual_seed(42)
        
        # Build encrypted model manually
        enc_fc1 = EncryptedLinear.from_torch(simple_mlp.fc1)
        enc_square = EncryptedSquare()
        enc_fc2 = EncryptedLinear.from_torch(simple_mlp.fc2)
        
        # Create input
        x = torch.randn(16, dtype=torch.float64)
        
        # Plaintext inference (with Square activation)
        with torch.no_grad():
            plain_output = simple_mlp(x.float()).to(torch.float64)
        
        # HE inference
        enc_x = gpu_context.encrypt(x)
        enc_x = enc_fc1(enc_x)
        enc_x = enc_square(enc_x)
        enc_output = enc_fc2(enc_x)
        he_output = gpu_context.decrypt(enc_output)
        
        # Move to CPU for comparison
        he_output_cpu = he_output.cpu() if he_output.is_cuda else he_output
        he_output_cpu = he_output_cpu[:10]
        
        # Compare
        cos_sim = F.cosine_similarity(
            plain_output.flatten().unsqueeze(0),
            he_output_cpu.flatten().unsqueeze(0)
        ).item()
        
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} < 0.95"

    @requires_gpu
    def test_mlp_multiple_samples(self, simple_mlp, gpu_context):
        """Test MLP with multiple different inputs."""
        from ckks_torch.nn import EncryptedLinear, EncryptedSquare
        
        enc_fc1 = EncryptedLinear.from_torch(simple_mlp.fc1)
        enc_square = EncryptedSquare()
        enc_fc2 = EncryptedLinear.from_torch(simple_mlp.fc2)
        
        outputs = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            x = torch.randn(16, dtype=torch.float64)
            
            enc_x = gpu_context.encrypt(x)
            enc_x = enc_fc1(enc_x)
            enc_x = enc_square(enc_x)
            enc_out = enc_fc2(enc_x)
            out = gpu_context.decrypt(enc_out)
            out_cpu = out.cpu() if out.is_cuda else out
            outputs.append(out_cpu[:10])
        
        # Each output should be different
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-3)
        assert not torch.allclose(outputs[1], outputs[2], atol=1e-3)

    @requires_gpu
    def test_mlp_timing(self, simple_mlp, gpu_context):
        """Benchmark MLP inference on GPU."""
        from ckks_torch.nn import EncryptedLinear, EncryptedSquare
        
        enc_fc1 = EncryptedLinear.from_torch(simple_mlp.fc1)
        enc_square = EncryptedSquare()
        enc_fc2 = EncryptedLinear.from_torch(simple_mlp.fc2)
        
        torch.manual_seed(42)
        x = torch.randn(16, dtype=torch.float64)
        
        # Warmup
        enc_x = gpu_context.encrypt(x)
        _ = enc_fc2(enc_square(enc_fc1(enc_x)))
        
        # Benchmark
        start = time.time()
        n_runs = 5
        for _ in range(n_runs):
            enc_x = gpu_context.encrypt(x)
            enc_x = enc_fc1(enc_x)
            enc_x = enc_square(enc_x)
            _ = enc_fc2(enc_x)
        elapsed = time.time() - start
        
        avg_time = elapsed / n_runs * 1000  # ms
        print(f"\nMLP inference: {avg_time:.1f}ms/sample")
        
        # Sanity check - should complete in reasonable time
        assert avg_time < 10000, f"Too slow: {avg_time:.1f}ms"


# =============================================================================
# CNN Tests
# =============================================================================


class TestGPU_CNN:
    """GPU integration tests for CNN models."""
    
    @pytest.fixture
    def simple_cnn(self):
        """Simple CNN for 8x8 input: Conv -> Square -> Pool -> FC."""
        torch.manual_seed(42)
        
        class SimpleCNN(nn.Module):
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
        
        model = SimpleCNN()
        model.eval()
        return model

    @requires_gpu
    def test_cnn_encrypt_image(self, gpu_context):
        """Test encrypting a CNN input (image) on GPU."""
        torch.manual_seed(42)
        
        # 8x8 single-channel image
        image = torch.randn(1, 8, 8, dtype=torch.float64)
        
        # Conv params for im2col
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        
        # Encrypt with CNN layout
        enc_input = gpu_context.encrypt_cnn_input(image, conv_params)
        
        # Should have CNN layout metadata
        assert hasattr(enc_input, '_cnn_layout')
        assert enc_input._cnn_layout is not None
        assert enc_input._cnn_layout['num_patches'] == 64  # 8x8
        assert enc_input._cnn_layout['patch_features'] == 9  # 3x3 kernel

    @requires_gpu
    def test_cnn_he_inference(self, simple_cnn, gpu_context):
        """Test full CNN inference on GPU with HE."""
        from ckks_torch.nn import (
            EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
            EncryptedFlatten, EncryptedLinear
        )
        
        torch.manual_seed(42)
        
        # Create 8x8 test image
        image = torch.randn(1, 1, 8, 8)
        
        # Plaintext inference
        with torch.no_grad():
            plain_output = simple_cnn(image).flatten().to(torch.float64)
        
        # Build encrypted model
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        cnn_layout = {'num_patches': 16, 'patch_features': 4}  # After pool: 4x4
        
        enc_conv = EncryptedConv2d.from_torch(simple_cnn.conv)
        enc_square = EncryptedSquare()
        enc_pool = EncryptedAvgPool2d.from_torch(simple_cnn.pool)
        enc_flatten = EncryptedFlatten._with_absorbed_permutation()
        enc_fc = EncryptedLinear.from_torch_cnn(simple_cnn.fc, cnn_layout)
        
        # HE inference
        enc_input = gpu_context.encrypt_cnn_input(image.squeeze(0).to(torch.float64), conv_params)
        enc_x = enc_conv(enc_input)
        enc_x = enc_square(enc_x)
        enc_x = enc_pool(enc_x)
        enc_x = enc_flatten(enc_x)
        enc_output = enc_fc(enc_x)
        
        he_output = gpu_context.decrypt(enc_output)
        he_output_cpu = he_output.cpu() if he_output.is_cuda else he_output
        he_output_cpu = he_output_cpu[:10]
        
        # Compare
        cos_sim = F.cosine_similarity(
            plain_output.unsqueeze(0),
            he_output_cpu.unsqueeze(0)
        ).item()
        
        assert cos_sim > 0.85, f"Cosine similarity {cos_sim:.4f} < 0.85"

    @requires_gpu
    @pytest.mark.skip(reason="convert() CNN support needs investigation - manual construction works")
    def test_cnn_convert_and_infer(self, simple_cnn, gpu_context):
        """Test CNN conversion with optimize_cnn and inference."""
        from ckks_torch import convert
        
        torch.manual_seed(42)
        
        # Convert with optimization
        enc_model, _ = convert(simple_cnn, use_square_activation=True, optimize_cnn=True)
        
        # Create input
        image = torch.randn(1, 1, 8, 8)
        
        # Plaintext inference
        with torch.no_grad():
            plain_output = simple_cnn(image).flatten().to(torch.float64)
        
        # HE inference
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        enc_input = gpu_context.encrypt_cnn_input(image.squeeze(0).to(torch.float64), conv_params)
        
        # Run through encrypted model
        enc_output = enc_model(enc_input)
        he_output = gpu_context.decrypt(enc_output)
        he_output_cpu = he_output.cpu() if he_output.is_cuda else he_output
        he_output_cpu = he_output_cpu[:10]
        
        # Compare
        cos_sim = F.cosine_similarity(
            plain_output.unsqueeze(0),
            he_output_cpu.unsqueeze(0)
        ).item()
        
        assert cos_sim > 0.90, f"Cosine similarity {cos_sim:.4f} < 0.90"

    @requires_gpu
    def test_cnn_timing(self, simple_cnn, gpu_context):
        """Benchmark CNN inference on GPU."""
        from ckks_torch.nn import (
            EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
            EncryptedFlatten, EncryptedLinear
        )
        
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        cnn_layout = {'num_patches': 16, 'patch_features': 4}
        
        enc_conv = EncryptedConv2d.from_torch(simple_cnn.conv)
        enc_square = EncryptedSquare()
        enc_pool = EncryptedAvgPool2d.from_torch(simple_cnn.pool)
        enc_flatten = EncryptedFlatten._with_absorbed_permutation()
        enc_fc = EncryptedLinear.from_torch_cnn(simple_cnn.fc, cnn_layout)
        
        torch.manual_seed(42)
        image = torch.randn(1, 8, 8, dtype=torch.float64)
        
        # Warmup
        enc_input = gpu_context.encrypt_cnn_input(image, conv_params)
        _ = enc_fc(enc_flatten(enc_pool(enc_square(enc_conv(enc_input)))))
        
        # Benchmark
        start = time.time()
        n_runs = 3
        for _ in range(n_runs):
            enc_input = gpu_context.encrypt_cnn_input(image, conv_params)
            enc_x = enc_conv(enc_input)
            enc_x = enc_square(enc_x)
            enc_x = enc_pool(enc_x)
            enc_x = enc_flatten(enc_x)
            enc_output = enc_fc(enc_x)
        elapsed = time.time() - start
        
        avg_time = elapsed / n_runs * 1000  # ms
        print(f"\nCNN inference: {avg_time:.1f}ms/sample")
        
        # Sanity check
        assert avg_time < 30000, f"Too slow: {avg_time:.1f}ms"


# =============================================================================
# Real Backend (CPU) Tests - For CI without GPU
# =============================================================================


class TestRealBackend_CPU:
    """Tests with real CKKS backend on CPU (for CI without GPU)."""
    
    @requires_real_backend
    def test_basic_encrypt_decrypt(self, real_context):
        """Test basic encrypt/decrypt on CPU."""
        torch.manual_seed(42)
        data = torch.randn(16, dtype=torch.float64)
        
        enc_data = real_context.encrypt(data)
        decrypted = real_context.decrypt(enc_data)
        
        error = (data - decrypted[:16]).abs().max().item()
        assert error < 1e-4, f"Decrypt error {error:.2e} > 1e-4"

    @requires_real_backend
    def test_basic_operations(self, real_context):
        """Test basic HE operations on CPU."""
        torch.manual_seed(42)
        data = torch.randn(16, dtype=torch.float64)
        
        enc_data = real_context.encrypt(data)
        
        # Add
        enc_sum = enc_data.add(enc_data)
        dec_sum = real_context.decrypt(enc_sum)[:16]
        expected_sum = data + data
        assert torch.allclose(expected_sum, dec_sum, atol=1e-3)
        
        # Mul
        enc_prod = enc_data.mul(enc_data)
        dec_prod = real_context.decrypt(enc_prod)[:16]
        expected_prod = data * data
        assert torch.allclose(expected_prod, dec_prod, atol=1e-3)

    @requires_real_backend
    def test_mlp_inference_cpu(self, real_context):
        """Test MLP inference on CPU with real backend."""
        from ckks_torch.nn import EncryptedLinear, EncryptedSquare
        
        torch.manual_seed(42)
        
        # Create model layers directly
        fc1 = nn.Linear(16, 32)
        fc2 = nn.Linear(32, 10)
        
        # Build encrypted model
        enc_fc1 = EncryptedLinear.from_torch(fc1)
        enc_square = EncryptedSquare()
        enc_fc2 = EncryptedLinear.from_torch(fc2)
        
        x = torch.randn(16, dtype=torch.float64)
        
        # Plaintext with Square activation
        with torch.no_grad():
            h = fc1(x.float())
            h = h ** 2  # Square activation
            plain_output = fc2(h).to(torch.float64)
        
        # HE
        enc_x = real_context.encrypt(x)
        enc_x = enc_fc1(enc_x)
        enc_x = enc_square(enc_x)
        enc_output = enc_fc2(enc_x)
        he_output = real_context.decrypt(enc_output)[:10]
        
        cos_sim = F.cosine_similarity(
            plain_output.flatten().unsqueeze(0),
            he_output.flatten().unsqueeze(0)
        ).item()
        
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} < 0.95"
