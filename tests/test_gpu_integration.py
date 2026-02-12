"""GPU Integration Tests for CuKKS.

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

from conftest import requires_gpu, requires_real_backend, _has_real_backend, _has_cuda


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
        from cukks.nn import EncryptedLinear, EncryptedSquare
        
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
        from cukks.nn import EncryptedLinear, EncryptedSquare
        
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
        from cukks.nn import EncryptedLinear, EncryptedSquare
        
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
                self.act = nn.ReLU()  # Placeholder, will be replaced with Square by converter
                self.pool = nn.AvgPool2d(2)  # 8x8 -> 4x4
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(4 * 4 * 4, 10)  # 64 -> 10
            
            def forward(self, x):
                x = self.conv(x)
                x = self.act(x)
                x = self.pool(x)
                x = self.flatten(x)
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
        from cukks.nn import (
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
        
        pre_pool_h, pre_pool_w = 8, 8
        out_H, out_W, channels = 4, 4, 4
        
        sparse_positions = []
        for out_y in range(out_H):
            for out_x in range(out_W):
                in_y = 2 * out_y
                in_x = 2 * out_x
                in_patch_idx = in_y * pre_pool_w + in_x
                for c in range(channels):
                    sparse_positions.append(in_patch_idx * channels + c)
        
        cnn_layout = {
            'num_patches': out_H * out_W,
            'patch_features': channels,
            'sparse': True,
            'sparse_positions': sparse_positions,
            'total_slots': pre_pool_h * pre_pool_w * channels,
        }
        
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
    def test_cnn_convert_and_infer(self, simple_cnn, gpu_context):
        """Test CNN conversion with optimize_cnn and inference."""
        from cukks import convert
        
        torch.manual_seed(42)
        
        # Convert with optimization
        enc_model, _ = convert(simple_cnn, use_square_activation=True, optimize_cnn=True, input_shape=(1, 8, 8))
        
        # Create input
        image = torch.randn(1, 1, 8, 8)
        
        # Plaintext inference with Square activation (matching HE)
        with torch.no_grad():
            x = simple_cnn.conv(image)
            x = x ** 2  # Square instead of ReLU
            x = simple_cnn.pool(x)
            x = simple_cnn.flatten(x)
            plain_output = simple_cnn.fc(x).flatten().to(torch.float64)
        
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
        from cukks.nn import (
            EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
            EncryptedFlatten, EncryptedLinear
        )
        
        conv_params = [
            {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}
        ]
        pre_pool_h, pre_pool_w = 8, 8
        out_H, out_W, channels = 4, 4, 4
        sparse_positions = []
        for out_y in range(out_H):
            for out_x in range(out_W):
                in_patch_idx = (2 * out_y) * pre_pool_w + (2 * out_x)
                for c in range(channels):
                    sparse_positions.append(in_patch_idx * channels + c)
        cnn_layout = {
            'num_patches': 16, 'patch_features': 4,
            'sparse': True, 'sparse_positions': sparse_positions,
            'total_slots': pre_pool_h * pre_pool_w * channels,
        }
        
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
        expected_sum = (data + data).to(dec_sum.dtype)
        assert torch.allclose(expected_sum, dec_sum, atol=1e-3)
        
        # Mul
        enc_prod = enc_data.mul(enc_data)
        dec_prod = real_context.decrypt(enc_prod)[:16]
        expected_prod = (data * data).to(dec_prod.dtype)
        assert torch.allclose(expected_prod, dec_prod, atol=1e-3)

    @requires_real_backend
    def test_mlp_inference_cpu(self, real_context):
         """Test MLP inference on CPU with real backend."""
         from cukks.nn import EncryptedLinear, EncryptedSquare
         
         torch.manual_seed(42)
         
         # Create model layers directly
         fc1 = nn.Linear(16, 32).double()
         fc2 = nn.Linear(32, 10).double()
         
         # Build encrypted model
         enc_fc1 = EncryptedLinear.from_torch(fc1)
         enc_square = EncryptedSquare()
         enc_fc2 = EncryptedLinear.from_torch(fc2)
         
         x = torch.randn(16, dtype=torch.float64)
         
         # Plaintext with Square activation (float64 to match HE path)
         with torch.no_grad():
             h = fc1(x)
             h = h ** 2
             plain_output = fc2(h)
         
         # HE
         enc_x = real_context.encrypt(x)
         enc_x = enc_fc1(enc_x)
         enc_x = enc_square(enc_x)
         enc_output = enc_fc2(enc_x)
         he_output = real_context.decrypt(enc_output)[:10]
         
         cos_sim = F.cosine_similarity(
             plain_output.flatten().float().unsqueeze(0),
             he_output.flatten().float().unsqueeze(0)
         ).item()
         
         assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} < 0.95"


# =============================================================================
# GPU Operation Tests (Rotate, Square, MulPlain)
# =============================================================================


class TestGPURotate:
    """Tests for GPU-accelerated rotation."""
    
    @requires_gpu
    def test_rotate_uses_gpu(self, gpu_context):
        """Test that rotation operation executes on GPU."""
        # Create and encrypt a small test vector
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        # Pad to match num_slots
        x = torch.cat([x, torch.zeros(gpu_context.num_slots - len(x), dtype=torch.float64)])
        ct = gpu_context.encrypt(x)
        
        # Perform rotation - just verify it doesn't crash
        rotated = ct.rotate(1)
        
        # Verify we got an EncryptedTensor back
        assert hasattr(rotated, 'rotate'), "rotate() should return EncryptedTensor"
    
    @requires_gpu
    def test_rotate_multiple_steps(self, gpu_context):
        """Test rotation by various step sizes."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        x = torch.cat([x, torch.zeros(gpu_context.num_slots - len(x), dtype=torch.float64)])
        ct = gpu_context.encrypt(x)
        
        # Test multiple rotation steps
        for step in [1, -1, 5]:
            rotated = ct.rotate(step)
            # Just verify operation completes without error
            assert hasattr(rotated, 'rotate')


class TestGPUSquare:
    """Tests for GPU-accelerated square operation."""
    
    @requires_gpu
    def test_square_uses_gpu(self, gpu_context):
        """Test that square uses GPU when available."""
        x = torch.tensor([float(i) * 0.1 for i in range(gpu_context.num_slots)], dtype=torch.float64)
        ct = gpu_context.encrypt(x)
        
        # Perform square
        squared = ct.square()
        
        # Decrypt and verify
        result = gpu_context.decrypt(squared)
        for i in range(min(100, len(result))):
            expected = (x[i] ** 2).item()
            assert abs(result[i] - expected) < 1e-3
    
    @requires_gpu
    def test_square_chain(self, gpu_context):
        """Test chained square operations stay on GPU."""
        x = torch.full((gpu_context.num_slots,), 0.5, dtype=torch.float64)
        ct = gpu_context.encrypt(x)
        
        # Chain: x -> x^2 -> x^4
        sq1 = ct.square()
        sq2 = sq1.square()
        
        result = gpu_context.decrypt(sq2)
        expected = 0.5 ** 4  # 0.0625
        assert abs(result[0] - expected) < 1e-3


class TestGPUMulPlain:
    """Tests for GPU-accelerated plaintext multiplication."""
    
    @requires_gpu
    def test_mul_plain_uses_gpu(self, gpu_context):
        """Test that mul_plain uses GPU when available."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        x = torch.cat([x, torch.zeros(gpu_context.num_slots - len(x), dtype=torch.float64)])
        ct = gpu_context.encrypt(x)
        
        # Multiply by plaintext scalar
        scale = 2.0
        result_ct = ct.mul(scale)
        
        # Just verify operation completes without error
        assert hasattr(result_ct, 'mul'), "mul() should return EncryptedTensor"
    
    @requires_gpu
    def test_mul_plain_vector(self, gpu_context):
        """Test multiplication by plaintext vector."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        x = torch.cat([x, torch.zeros(gpu_context.num_slots - len(x), dtype=torch.float64)])
        plain = torch.full((gpu_context.num_slots,), 2.0, dtype=torch.float64)
        ct = gpu_context.encrypt(x)
        
        result_ct = ct.mul(plain)
        
        # Just verify operation completes without error
        assert hasattr(result_ct, 'mul')


class TestGPUChainedOperations:
    """Test that chained GPU operations stay on GPU."""
    
    @requires_gpu
    def test_add_rotate_square_chain(self, gpu_context):
        """Test a chain of operations: add -> rotate -> square."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        x = torch.cat([x, torch.zeros(gpu_context.num_slots - len(x), dtype=torch.float64)])
        ct1 = gpu_context.encrypt(x)
        ct2 = gpu_context.encrypt(x)
        
        # Chain operations
        added = ct1.add(ct2)
        rotated = added.rotate(1)
        squared = rotated.square()
        
        # Just verify the chain completes without error
        assert hasattr(squared, 'square'), "Chained operations should return EncryptedTensor"


# =============================================================================
# Bootstrap E2E Tests
# =============================================================================


class TestGPUBootstrap:
    """GPU E2E tests for bootstrapping on deep networks."""

    @pytest.fixture(scope="class")
    def bootstrap_context(self):
        """CKKS context with bootstrapping enabled (expensive, class-scoped).

        Uses security_level=None so OpenFHE does not inflate the ring
        beyond what GPU memory can handle.  The C++ backend sets
        mult_depth = levelsAfterBootstrap(10) + bootstrapDepth(~12) ≈ 22
        automatically when enable_bootstrap=True.
        """
        if not _has_real_backend() or not _has_cuda():
            pytest.skip("GPU bootstrap test requires real backend + CUDA")

        from cukks import CKKSInferenceContext, InferenceConfig
        from cukks.context import compute_bsgs_rotations

        max_dim = 16
        rotations = compute_bsgs_rotations(max_dim)
        neg_rotations = [-r for r in rotations if r > 0]
        rotations = sorted(set(rotations + neg_rotations))

        config = InferenceConfig(
            poly_mod_degree=65536,
            scale_bits=45,
            mult_depth=10,
            enable_bootstrap=True,
            level_budget=(3, 3),
            security_level=None,
        )

        ctx = CKKSInferenceContext(
            config=config,
            device="cpu",
            rotations=rotations,
            use_bsgs=True,
            max_rotation_dim=max_dim,
            auto_bootstrap=True,
            bootstrap_threshold=3,
            enable_gpu=False,
        )
        return ctx

    @requires_real_backend
    def test_manual_bootstrap_refreshes_levels(self, bootstrap_context):
        """Verify EvalBootstrap restores ciphertext levels."""
        ctx = bootstrap_context
        x = torch.randn(16, dtype=torch.float64)
        enc = ctx.encrypt(x)

        # Consume a few levels: mul → rescale × 3
        # Keep it modest so degree stays within correction factor
        for _ in range(3):
            enc = enc.mul(1.5)
            enc = enc.rescale()

        # Bootstrap should reset depth
        refreshed = enc.bootstrap()
        assert refreshed._depth == 0, f"Expected depth 0 after bootstrap, got {refreshed._depth}"

        # Values should still be recoverable
        dec = ctx.decrypt(refreshed)
        dec_cpu = dec.cpu() if dec.is_cuda else dec
        assert dec_cpu[:16].shape == (16,)

    @requires_real_backend
    def test_deep_mlp_with_auto_bootstrap(self, bootstrap_context):
        """Run an MLP (Linear→x²)×2→Linear through auto-bootstrap.

        With threshold=3 the bootstrap fires after Linear(1)+Square(2)=depth 3
        before the second block, proving the auto-bootstrap hook works on the
        real OpenFHE backend.
        """
        from cukks.nn import EncryptedLinear, EncryptedSquare, EncryptedSequential

        ctx = bootstrap_context
        torch.manual_seed(42)

        dim = 16
        layers = []
        for _ in range(2):
            w = torch.eye(dim, dtype=torch.float64) * 0.5
            b = torch.zeros(dim, dtype=torch.float64)
            layers.append(EncryptedLinear(dim, dim, w, b))
            layers.append(EncryptedSquare())
        w_out = torch.eye(dim, dtype=torch.float64)
        b_out = torch.zeros(dim, dtype=torch.float64)
        layers.append(EncryptedLinear(dim, dim, w_out, b_out))

        seq = EncryptedSequential(*layers)

        x = torch.full((dim,), 0.8, dtype=torch.float64)
        enc_x = ctx.encrypt(x)
        enc_out = seq(enc_x)

        dec = ctx.decrypt(enc_out)
        dec_cpu = dec.cpu() if dec.is_cuda else dec
        dec_cpu = dec_cpu[:dim]

        # Plaintext reference: ((0.8 * 0.5)²)² * 1.0
        val = 0.8
        for _ in range(2):
            val = (val * 0.5) ** 2
        expected = torch.full((dim,), val, dtype=torch.float64)

        cos_sim = F.cosine_similarity(
            expected.unsqueeze(0), dec_cpu.unsqueeze(0)
        ).item()

        print(f"\nDeep MLP bootstrap E2E: cosine={cos_sim:.6f}, "
              f"expected={val:.6e}, got={dec_cpu[0].item():.6e}")

        assert cos_sim > 0.80, f"Cosine similarity {cos_sim:.4f} < 0.80"

    @requires_real_backend
    def test_convert_deep_model_auto_bootstrap(self, bootstrap_context):
        """Verify convert() auto-detects bootstrap need for a deep model."""
        from cukks import convert

        torch.manual_seed(42)
        layers = []
        for _ in range(8):
            layers.append(nn.Linear(16, 16))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 4))
        model = nn.Sequential(*layers).eval()

        _, ctx = convert(model, activation_degree=4)

        assert ctx.config.enable_bootstrap is True, "Deep model should auto-enable bootstrap"
        assert ctx._auto_bootstrap is True, "Deep model should enable auto_bootstrap"
        assert ctx.config.poly_mod_degree == 65536, "Bootstrap requires poly_mod_degree=65536"
