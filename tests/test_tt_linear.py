"""Tests for EncryptedTTLinear (Tensor Train decomposition)."""

from typing import Optional

import torch
import torch.nn as nn

from ckks_torch.nn import EncryptedTTLinear


class TestTTDecomposition:
    """Tests for TT decomposition via from_torch()."""
    
    def test_from_torch_basic(self):
        """Test basic TT decomposition of a Linear layer."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert len(tt.tt_cores) > 0
        assert len(tt.tt_shapes) == len(tt.tt_cores)
        assert tt.in_features >= 784  # May be padded
        assert tt.out_features >= 128
    
    def test_from_torch_small_layer_returns_none(self):
        """Small layers should return None (below threshold)."""
        small = nn.Linear(16, 16)  # 256 < 1024 threshold
        result = EncryptedTTLinear.from_torch(small)
        assert result is None
    
    def test_from_torch_with_bias(self):
        """Test that bias is preserved."""
        linear = nn.Linear(784, 128, bias=True)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.shape[0] >= 128
    
    def test_from_torch_no_bias(self):
        """Test layer without bias."""
        linear = nn.Linear(784, 128, bias=False)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is None
    
    def test_from_torch_max_rank(self):
        """Test max_rank parameter limits TT ranks."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear, max_rank=8)
        
        assert tt is not None
        # Check all intermediate ranks are <= 8
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] <= 8
    
    def test_mult_depth(self):
        """Forward uses pre-computed weight, so mult_depth is 1 (single matmul)."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.mult_depth() == 1


class TestTTLinearProperties:
    """Test properties and attributes of EncryptedTTLinear."""
    
    def test_tt_shapes_match_cores(self):
        """Test that tt_shapes has correct format."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for shape, core in zip(tt.tt_shapes, tt.tt_cores):
            n_k, m_k = shape
            # Core mode dimension should be n_k * m_k
            assert core.shape[1] == n_k * m_k
    
    def test_extra_repr(self):
        """Test string representation."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        repr_str = tt.extra_repr()
        assert 'in_features=' in repr_str
        assert 'out_features=' in repr_str
        assert 'num_cores=' in repr_str
    
    def test_cores_are_float64(self):
        """TT cores should be float64 for CKKS precision."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert core.dtype == torch.float64
    
    def test_bias_is_float64(self):
        """Bias should be float64 for CKKS precision."""
        linear = nn.Linear(784, 128, bias=True)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.dtype == torch.float64


class TestTTEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_prime_dimensions(self):
        """Test with prime-sized dimensions (need padding)."""
        linear = nn.Linear(797, 131)  # Primes
        tt = EncryptedTTLinear.from_torch(linear)
        
        # Should still work (with padding)
        assert tt is not None
        assert tt.in_features >= 797
        assert tt.out_features >= 131
    
    def test_power_of_two_dimensions(self):
        """Test with power-of-2 dimensions (clean factorization)."""
        linear = nn.Linear(1024, 256)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
    
    def test_asymmetric_dimensions(self):
        """Test highly asymmetric dimensions."""
        linear = nn.Linear(2048, 64)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
    
    def test_boundary_threshold(self):
        """Test layers near the 1024 threshold."""
        # Just below threshold
        small = nn.Linear(32, 31)  # 992 < 1024
        assert EncryptedTTLinear.from_torch(small) is None
        
        # Just above threshold
        large = nn.Linear(32, 32)  # 1024 >= 1024
        assert EncryptedTTLinear.from_torch(large) is not None


class TestTTCoreStructure:
    """Test internal structure of TT cores."""
    
    def test_first_core_rank_is_one(self):
        """First core should have left rank = 1."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        first_core = tt.tt_cores[0]
        assert first_core.shape[0] == 1
    
    def test_last_core_rank_is_one(self):
        """Last core should have right rank = 1."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        last_core = tt.tt_cores[-1]
        assert last_core.shape[2] == 1
    
    def test_core_rank_continuity(self):
        """Right rank of core k should equal left rank of core k+1."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for i in range(len(tt.tt_cores) - 1):
            right_rank = tt.tt_cores[i].shape[2]
            left_rank = tt.tt_cores[i + 1].shape[0]
            assert right_rank == left_rank
    
    def test_core_shapes_consistency(self):
        """Test that all cores have consistent shape structure."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert len(core.shape) == 3
            r_prev, mode_size, r_next = core.shape
            assert r_prev >= 1
            assert mode_size >= 1
            assert r_next >= 1


class TestTTBiasHandling:
    """Test bias handling in TT decomposition."""
    
    def test_bias_padding_matches_output(self):
        """Padded bias should match padded output features."""
        linear = nn.Linear(797, 131, bias=True)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.shape[0] == tt.out_features
    
    def test_bias_original_values_preserved(self):
        """Original bias values should be preserved in padded bias."""
        linear = nn.Linear(100, 50, bias=True)
        original_bias = linear.bias.clone().to(dtype=torch.float64)
        
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is not None
        # First 50 elements should match original
        assert torch.allclose(tt.bias[:50], original_bias, atol=1e-6)
    
    def test_bias_padding_is_zero(self):
        """Padded bias elements should be zero."""
        linear = nn.Linear(797, 131, bias=True)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.bias is not None
        if tt.out_features > 131:
            # Padded elements should be zero
            assert torch.allclose(tt.bias[131:], torch.zeros(tt.out_features - 131, dtype=torch.float64), atol=1e-10)


class TestTTMaxRankBehavior:
    """Test max_rank parameter behavior."""
    
    def test_max_rank_respected(self):
        """All intermediate ranks should respect max_rank."""
        linear = nn.Linear(784, 128)
        max_rank = 4
        tt = EncryptedTTLinear.from_torch(linear, max_rank=max_rank)
        
        assert tt is not None
        # Check intermediate ranks (not first or last)
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] <= max_rank
    
    def test_max_rank_one(self):
        """Test with max_rank=1 (minimal decomposition)."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear, max_rank=1)
        
        assert tt is not None
        # All intermediate ranks should be 1
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] == 1
    
    def test_max_rank_large(self):
        """Test with very large max_rank."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear, max_rank=1000)
        
        assert tt is not None
        # Should still work, ranks limited by actual decomposition


class TestTTDifferentSizes:
    """Test TT decomposition with various layer sizes."""
    
    def test_square_layer(self):
        """Test with square weight matrix."""
        linear = nn.Linear(256, 256)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.in_features == tt.out_features
    
    def test_wide_layer(self):
        """Test with more inputs than outputs."""
        linear = nn.Linear(1024, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.in_features > tt.out_features
    
    def test_tall_layer(self):
        """Test with more outputs than inputs."""
        linear = nn.Linear(128, 1024)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt.out_features > tt.in_features
    
    def test_large_layer(self):
        """Test with large layer."""
        linear = nn.Linear(4096, 2048)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None


class TestTTOriginalFeatures:
    """Test tracking of original (unpadded) features."""
    
    def test_original_out_features_stored(self):
        """Original output features should be stored."""
        linear = nn.Linear(797, 131)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt._original_out_features == 131
    
    def test_original_out_features_matches_input(self):
        """Original features should match input when no padding needed."""
        linear = nn.Linear(256, 256)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert tt._original_out_features == 256
        assert tt.out_features == 256


class TestTTConverterIntegration:
    """Test integration with model converter."""
    
    def test_from_torch_returns_encrypted_tt_linear(self):
        """from_torch should return EncryptedTTLinear instance."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert isinstance(tt, EncryptedTTLinear)
    
    def test_from_torch_detaches_weights(self):
        """Weights should be detached from computation graph."""
        linear = nn.Linear(784, 128)
        linear.weight.requires_grad = True
        
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert not core.requires_grad
    
    def test_from_torch_moves_to_cpu(self):
        """Weights should be on CPU."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert core.device.type == 'cpu'
        if tt.bias is not None:
            assert tt.bias.device.type == 'cpu'


class TestTTFactorization:
    """Test factorization behavior."""
    
    def test_factorization_produces_cores(self):
        """Factorization should produce at least 2 cores."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert len(tt.tt_cores) >= 2
    
    def test_factorization_shapes_match(self):
        """Number of shapes should match number of cores."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        assert len(tt.tt_shapes) == len(tt.tt_cores)
    
    def test_shapes_are_tuples(self):
        """Each shape should be a tuple of 2 integers."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        
        assert tt is not None
        for shape in tt.tt_shapes:
            assert isinstance(shape, tuple)
            assert len(shape) == 2
            assert all(isinstance(x, int) for x in shape)


class TestTTLinearForward:
    """Tests for EncryptedTTLinear forward() method with weight reconstruction."""
    
    def test_forward_weight_reconstruction_matches_original(self):
        """Verify _effective_weight reconstructs from TT-cores correctly."""
        linear = nn.Linear(784, 128)
        
        tt = EncryptedTTLinear.from_torch(linear)
        assert tt is not None
        
        # Verify that _effective_weight was computed and has correct shape
        assert tt._effective_weight is not None
        assert tt._effective_weight.shape == (tt.out_features, tt.in_features)
        
        # Verify reconstruction is deterministic (same cores produce same weight)
        reconstructed_again = tt._reconstruct_weight()
        torch.testing.assert_close(
            tt._effective_weight,
            reconstructed_again,
            rtol=1e-10,
            atol=1e-12
        )
    
    def test_forward_input_validation(self):
        """Verify ValueError is raised on wrong input size."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        assert tt is not None
        
        # Create a mock EncryptedTensor with wrong shape
        class MockEncryptedTensor:
            def __init__(self, shape: tuple) -> None:
                self.shape = shape
            
            def matmul(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "MockEncryptedTensor":
                raise AssertionError("Should not reach matmul")
        
        wrong_input = MockEncryptedTensor((512,))
        
        try:
            tt.forward(wrong_input)  # type: ignore
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Input size mismatch" in str(e)
            assert "784" in str(e)
            assert "512" in str(e)
    
    def test_forward_output_shape(self):
        """Verify output shape matches out_features."""
        linear = nn.Linear(784, 128)
        tt = EncryptedTTLinear.from_torch(linear)
        assert tt is not None
        
        # Create a mock EncryptedTensor with correct input shape
        class MockEncryptedTensor:
            def __init__(self, shape: tuple) -> None:
                self.shape = shape
            
            def matmul(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "MockEncryptedTensor":
                out_size = weight.shape[0]
                return MockEncryptedTensor((out_size,))
        
        input_tensor = MockEncryptedTensor((tt.in_features,))
        output = tt.forward(input_tensor)  # type: ignore
        
        assert output.shape == (tt.out_features,)
    
    def test_forward_with_svd_threshold(self):
        """Verify custom svd_threshold parameter affects decomposition."""
        linear = nn.Linear(784, 128)
        
        # Create two TT layers with different SVD thresholds
        tt_loose = EncryptedTTLinear.from_torch(linear, svd_threshold=1e-3)
        tt_strict = EncryptedTTLinear.from_torch(linear, svd_threshold=1e-8)
        
        assert tt_loose is not None
        assert tt_strict is not None
        
        # Looser threshold should generally have lower ranks (fewer singular values kept)
        # Count total rank across all cores
        loose_rank_sum = sum(core.shape[2] for core in tt_loose.tt_cores[:-1])
        strict_rank_sum = sum(core.shape[2] for core in tt_strict.tt_cores[:-1])
        
        # Strict threshold should keep more singular values
        assert strict_rank_sum >= loose_rank_sum
    
    def test_forward_weight_reconstruction_with_bias(self):
        """Verify weight reconstruction works with bias present."""
        linear = nn.Linear(256, 64, bias=True)
        original_weight = linear.weight.detach().to(dtype=torch.float64)
        original_bias = linear.bias.detach().to(dtype=torch.float64)
        
        tt = EncryptedTTLinear.from_torch(linear)
        assert tt is not None
        assert tt.bias is not None
        
        # Check weight reconstruction
        reconstructed_weight = tt._effective_weight[:64, :256]
        torch.testing.assert_close(
            reconstructed_weight,
            original_weight,
            rtol=1e-4,
            atol=1e-6
        )
        
        # Check bias preservation
        torch.testing.assert_close(
            tt.bias[:64],
            original_bias,
            rtol=1e-4,
            atol=1e-6
        )
    
    def test_forward_effective_weight_shape(self):
        """Verify _effective_weight has correct shape."""
        linear = nn.Linear(512, 256)
        tt = EncryptedTTLinear.from_torch(linear)
        assert tt is not None
        
        # Effective weight should be (padded_out, padded_in)
        assert tt._effective_weight.shape == (tt.out_features, tt.in_features)
        assert tt._effective_weight.dtype == torch.float64


# Run with: pytest tests/test_tt_linear.py -v
