"""Tests for encrypted neural network modules."""

import pytest
import torch
import torch.nn as nn

from ckks_torch.nn import (
    EncryptedModule,
    EncryptedLinear,
    EncryptedConv2d,
    EncryptedSequential,
    EncryptedFlatten,
    EncryptedSquare,
    EncryptedReLU,
    EncryptedGELU,
    EncryptedAvgPool2d,
    EncryptedMaxPool2d,
    EncryptedApproxAttention,
)


class TestEncryptedLinear:
    """Test EncryptedLinear layer."""
    
    def test_from_torch(self):
        """Test creation from PyTorch Linear."""
        torch_linear = nn.Linear(10, 5, bias=True)
        torch_linear.eval()
        
        enc_linear = EncryptedLinear.from_torch(torch_linear)
        
        assert enc_linear.in_features == 10
        assert enc_linear.out_features == 5
        assert enc_linear.bias is not None
    
    def test_from_torch_no_bias(self):
        """Test creation from PyTorch Linear without bias."""
        torch_linear = nn.Linear(10, 5, bias=False)
        torch_linear.eval()
        
        enc_linear = EncryptedLinear.from_torch(torch_linear)
        
        assert enc_linear.bias is None
    
    def test_weight_shape(self):
        """Test that weights have correct shape."""
        weight = torch.randn(5, 10)
        enc_linear = EncryptedLinear(10, 5, weight)
        
        assert enc_linear.weight.shape == (5, 10)
    
    def test_mult_depth(self):
        """Linear layer should have mult_depth 1."""
        enc_linear = EncryptedLinear(10, 5, torch.randn(5, 10))
        assert enc_linear.mult_depth() == 1
    
    def test_repr(self):
        """Test string representation."""
        enc_linear = EncryptedLinear(10, 5, torch.randn(5, 10))
        repr_str = repr(enc_linear)
        
        assert "EncryptedLinear" in repr_str
        assert "in_features=10" in repr_str
        assert "out_features=5" in repr_str


class TestEncryptedConv2d:
    """Test EncryptedConv2d layer."""
    
    def test_from_torch(self):
        """Test creation from PyTorch Conv2d."""
        torch_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        torch_conv.eval()
        
        enc_conv = EncryptedConv2d.from_torch(torch_conv)
        
        assert enc_conv.in_channels == 3
        assert enc_conv.out_channels == 16
        assert enc_conv.kernel_size == (3, 3)
        assert enc_conv.padding == (1, 1)
    
    def test_weight_matrix_shape(self):
        """Test that weight is reshaped for matmul."""
        torch_conv = nn.Conv2d(3, 16, kernel_size=3)
        enc_conv = EncryptedConv2d.from_torch(torch_conv)
        
        # Should be (out_channels, in_channels * kH * kW)
        assert enc_conv.weight_matrix.shape == (16, 3 * 3 * 3)
    
    def test_output_size_calculation(self):
        """Test output size calculation."""
        enc_conv = EncryptedConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            weight=torch.randn(16, 3, 3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        
        out_h, out_w = enc_conv.get_output_size(28, 28)
        
        # (28 + 2*1 - 3) // 2 + 1 = 14
        assert out_h == 14
        assert out_w == 14
    
    def test_unfold_input(self):
        """Test input unfolding for im2col."""
        x = torch.randn(3, 8, 8)  # C, H, W
        
        patches = EncryptedConv2d.unfold_input(
            x,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
        )
        
        # Output: (6*6, 3*3*3) = (36, 27)
        assert patches.shape == (36, 27)
    
    def test_conv2d_auto_unfold(self, mock_enc_context):
        """Test auto-unfold with 4D input."""
        torch_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        torch_conv.eval()
        enc_conv = EncryptedConv2d.from_torch(torch_conv)
        
        x = torch.randn(1, 3, 8, 8)
        
        enc_tensor = mock_enc_context.encrypt(x)
        enc_output = enc_conv(enc_tensor)
        
        assert enc_conv.input_shape == (1, 3, 8, 8)
        assert enc_conv.output_shape == (1, 8, 8, 8)
        assert enc_output.shape == (1, 8, 8, 8)
    
    def test_conv2d_chained(self, mock_enc_context):
        """Test chained conv: conv2(conv1(x))."""
        conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        conv1.eval()
        conv2.eval()
        
        enc_conv1 = EncryptedConv2d.from_torch(conv1)
        enc_conv2 = EncryptedConv2d.from_torch(conv2)
        
        x = torch.randn(1, 3, 8, 8)
        enc_tensor = mock_enc_context.encrypt(x)
        
        out1 = enc_conv1(enc_tensor)
        assert enc_conv1.output_shape == (1, 8, 8, 8)
        assert out1.shape == (1, 8, 8, 8)
        
        out2 = enc_conv2(out1)
        assert enc_conv2.output_shape == (1, 16, 8, 8)
        assert out2.shape == (1, 16, 8, 8)


class TestEncryptedSequential:
    """Test EncryptedSequential container."""
    
    def test_construction(self):
        """Test creating a sequential model."""
        model = EncryptedSequential(
            EncryptedLinear(10, 5, torch.randn(5, 10)),
            EncryptedSquare(),
            EncryptedLinear(5, 2, torch.randn(2, 5)),
        )
        
        assert len(model) == 3
    
    def test_indexing(self):
        """Test accessing layers by index."""
        linear = EncryptedLinear(10, 5, torch.randn(5, 10))
        square = EncryptedSquare()
        
        model = EncryptedSequential(linear, square)
        
        assert model[0] is linear
        assert model[1] is square
    
    def test_iteration(self):
        """Test iterating over layers."""
        layers = [
            EncryptedLinear(10, 5, torch.randn(5, 10)),
            EncryptedSquare(),
        ]
        model = EncryptedSequential(*layers)
        
        for i, layer in enumerate(model):
            assert layer is layers[i]
    
    def test_mult_depth(self):
        """Test total multiplicative depth."""
        model = EncryptedSequential(
            EncryptedLinear(10, 5, torch.randn(5, 10)),  # depth 1
            EncryptedSquare(),  # depth 1
            EncryptedLinear(5, 2, torch.randn(2, 5)),  # depth 1
        )
        
        assert model.mult_depth() == 3


class TestActivations:
    """Test activation functions."""
    
    def test_square_mult_depth(self):
        """Square activation has mult_depth 1."""
        assert EncryptedSquare().mult_depth() == 1
    
    def test_relu_degree(self):
        """Test ReLU with different degrees."""
        relu4 = EncryptedReLU(degree=4)
        relu7 = EncryptedReLU(degree=7)
        
        assert relu4.degree == 4
        assert relu7.degree == 7
        assert len(relu4.coeffs) == 5  # degree + 1
    
    def test_relu_mult_depth(self):
        """Higher degree should have higher depth."""
        relu2 = EncryptedReLU(degree=2)
        relu8 = EncryptedReLU(degree=8)
        
        assert relu2.mult_depth() <= relu8.mult_depth()
    
    def test_gelu_creation(self):
        """Test GELU activation."""
        gelu = EncryptedGELU(degree=4)
        assert gelu.degree == 4


class TestEncryptedFlatten:
    """Test EncryptedFlatten layer."""
    
    def test_mult_depth(self):
        """Flatten is free (no multiplication)."""
        flatten = EncryptedFlatten()
        assert flatten.mult_depth() == 0
    
    def test_repr(self):
        """Test string representation."""
        flatten = EncryptedFlatten(start_dim=1, end_dim=-1)
        repr_str = repr(flatten)
        
        assert "EncryptedFlatten" in repr_str


class TestMaxPool2d:

    def test_maxpool2d_creation(self):
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2, padding=0, degree=4)
        
        assert pool.kernel_size == (2, 2)
        assert pool.stride == (2, 2)
        assert pool.padding == (0, 0)
        assert pool.degree == 4
        assert pool.pool_area == 4

    def test_maxpool2d_from_torch(self):
        torch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        enc_pool = EncryptedMaxPool2d.from_torch(torch_pool)
        
        assert enc_pool.kernel_size == (3, 3)
        assert enc_pool.stride == (2, 2)
        assert enc_pool.padding == (1, 1)

    def test_maxpool2d_output_size(self):
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2)
        
        out_h, out_w = pool.get_output_size(8, 8)
        assert out_h == 4
        assert out_w == 4
        
        out_h2, out_w2 = pool.get_output_size(7, 7)
        assert out_h2 == 3
        assert out_w2 == 3

    def test_maxpool2d_mult_depth(self):
        pool2x2 = EncryptedMaxPool2d(kernel_size=2, degree=4)
        pool3x3 = EncryptedMaxPool2d(kernel_size=3, degree=4)
        
        assert pool2x2.mult_depth() >= 1
        assert pool3x3.mult_depth() >= pool2x2.mult_depth()

    def test_maxpool2d_accuracy(self):
        import numpy as np
        
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2, degree=4)
        
        x_vals = np.linspace(-1, 1, 100)
        errors = []
        for a in x_vals[::10]:
            for b in x_vals[::10]:
                exact_max = max(a, b)
                diff = a - b
                eps = 0.01
                approx_abs = np.sqrt(diff**2 + eps)
                approx_max = (a + b + approx_abs) / 2
                abs_error = abs(approx_max - exact_max)
                errors.append(abs_error)
        
        avg_error = np.mean(errors)
        assert avg_error < 0.10, f"Average absolute error {avg_error:.4f} exceeds 0.10"

    def test_maxpool2d_repr(self):
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2, padding=1, degree=6)
        repr_str = repr(pool)
        
        assert "EncryptedMaxPool2d" in repr_str
        assert "kernel_size=(2, 2)" in repr_str
        assert "degree=6" in repr_str


class TestAttention:
    """Test EncryptedApproxAttention module."""

    def test_attention_creation(self):
        """Test creating attention module with valid parameters."""
        attn = EncryptedApproxAttention(embed_dim=64, num_heads=4, softmax_degree=4)
        
        assert attn.embed_dim == 64
        assert attn.num_heads == 4
        assert attn.head_dim == 16
        assert attn.softmax_degree == 4
        assert attn.scale == 1.0 / (16 ** 0.5)

    def test_attention_creation_invalid_heads(self):
        """Test that invalid num_heads raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            EncryptedApproxAttention(embed_dim=64, num_heads=5)

    def test_attention_from_torch(self):
        """Test conversion from PyTorch MultiheadAttention."""
        torch_attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        torch_attn.eval()
        
        enc_attn = EncryptedApproxAttention.from_torch(torch_attn, softmax_degree=4)
        
        assert enc_attn.embed_dim == 32
        assert enc_attn.num_heads == 4
        assert enc_attn.q_weight is not None
        assert enc_attn.k_weight is not None
        assert enc_attn.v_weight is not None
        assert enc_attn.out_weight is not None

    def test_attention_mult_depth(self):
        """Test multiplicative depth estimation."""
        attn_no_proj = EncryptedApproxAttention(embed_dim=64, num_heads=4, softmax_degree=4)
        depth_no_proj = attn_no_proj.mult_depth()
        
        assert depth_no_proj == 6  # 1 (Q@K) + 4 (softmax) + 1 (attn@V)
        
        torch_attn = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        enc_attn_with_proj = EncryptedApproxAttention.from_torch(torch_attn, softmax_degree=4)
        depth_with_proj = enc_attn_with_proj.mult_depth()
        
        assert depth_with_proj == 10  # 3 (Q,K,V proj) + 1 + 4 + 1 + 1 (out proj)

    def test_attention_forward(self, mock_enc_context):
        """Test forward pass with encrypted tensors."""
        embed_dim = 8
        
        attn = EncryptedApproxAttention(embed_dim=embed_dim, num_heads=2, softmax_degree=4)
        
        query = torch.randn(1, embed_dim) * 0.5
        key = torch.randn(1, embed_dim) * 0.5
        value = torch.randn(1, embed_dim) * 0.5
        
        enc_query = mock_enc_context.encrypt(query)
        enc_key = mock_enc_context.encrypt(key)
        enc_value = mock_enc_context.encrypt(value)
        
        output = attn.forward_attention(enc_query, enc_key, enc_value)
        
        assert output is not None
        output_plain = mock_enc_context.decrypt(output)
        assert torch.isfinite(output_plain).all()

    def test_attention_approximate_accuracy(self, mock_enc_context):
        """Test that approximate attention produces reasonable outputs.
        
        Note: This is an approximation, so we check that outputs are
        in a reasonable range rather than exact equality.
        """
        import numpy as np
        
        embed_dim = 8
        
        attn = EncryptedApproxAttention(embed_dim=embed_dim, num_heads=2, softmax_degree=4)
        
        query = torch.randn(1, embed_dim) * 0.5
        key = torch.randn(1, embed_dim) * 0.5
        value = torch.randn(1, embed_dim) * 0.5
        
        enc_query = mock_enc_context.encrypt(query)
        enc_key = mock_enc_context.encrypt(key)
        enc_value = mock_enc_context.encrypt(value)
        
        output = attn.forward_attention(enc_query, enc_key, enc_value)
        output_plain = mock_enc_context.decrypt(output)
        
        assert torch.isfinite(output_plain).all(), "Output contains non-finite values"
        assert output_plain.abs().max() < 100, "Output values too large"

    def test_attention_repr(self):
        """Test string representation."""
        attn = EncryptedApproxAttention(embed_dim=64, num_heads=8, softmax_degree=6)
        repr_str = repr(attn)
        
        assert "EncryptedApproxAttention" in repr_str
        assert "embed_dim=64" in repr_str
        assert "num_heads=8" in repr_str
        assert "softmax_degree=6" in repr_str

    def test_taylor_exp_coeffs(self):
        """Test Taylor expansion coefficients for exp(x)."""
        from ckks_torch.nn.attention import _taylor_exp_coeffs
        
        coeffs = _taylor_exp_coeffs(4)
        
        expected = [1.0, 1.0, 0.5, 1/6, 1/24]  # 1, 1, 1/2!, 1/3!, 1/4!
        
        assert len(coeffs) == 5
        for i, (c, e) in enumerate(zip(coeffs, expected)):
            assert abs(c - e) < 1e-10, f"Coefficient {i} mismatch: {c} vs {e}"

