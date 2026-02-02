"""Tests for EncryptedTTConv2d (Tensor Train decomposition for Conv2d)."""

from typing import Optional

import pytest
import torch
import torch.nn as nn

from ckks_torch.nn import EncryptedTTConv2d


class TestTTConvDecomposition:
    """Tests for TT decomposition via from_torch()."""
    
    def test_from_torch_basic(self):
        """Test basic TT decomposition of a Conv2d layer."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert len(tt.tt_cores) > 0
        assert len(tt.tt_shapes) == len(tt.tt_cores)
        assert tt.in_channels == 16
        assert tt._original_out_channels == 32
    
    def test_from_torch_small_layer_returns_none(self):
        """Small layers should return None (below 1024 param threshold)."""
        # 4 * 4 * 3 * 3 = 144 < 1024
        small = nn.Conv2d(4, 4, kernel_size=3)
        result = EncryptedTTConv2d.from_torch(small)
        assert result is None
    
    def test_from_torch_with_bias(self):
        """Test that bias is preserved."""
        conv = nn.Conv2d(16, 32, kernel_size=3, bias=True)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.shape[0] >= 32
    
    def test_from_torch_no_bias(self):
        """Test layer without bias."""
        conv = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is None
    
    def test_from_torch_max_rank(self):
        """Test max_rank parameter limits TT ranks."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, max_rank=8)
        
        assert tt is not None
        # Check all intermediate ranks are <= 8
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] <= 8
    
    def test_mult_depth(self):
        """Test that mult_depth returns 1 (single matmul)."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.mult_depth() == 1  # Single matmul, not number of cores


class TestTTConvProperties:
    """Test properties and attributes of EncryptedTTConv2d."""
    
    def test_tt_shapes_match_cores(self):
        """Test that tt_shapes has correct format."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for shape, core in zip(tt.tt_shapes, tt.tt_cores):
            n_k, m_k = shape
            # Core mode dimension should be n_k * m_k
            assert core.shape[1] == n_k * m_k
    
    def test_extra_repr(self):
        """Test string representation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        repr_str = tt.extra_repr()
        assert '16' in repr_str  # in_channels
        assert '32' in repr_str  # out_channels
        assert 'kernel_size=' in repr_str
        assert 'num_cores=' in repr_str
    
    def test_cores_are_float64(self):
        """TT cores should be float64 for CKKS precision."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert core.dtype == torch.float64
    
    def test_bias_is_float64(self):
        """Bias should be float64 for CKKS precision."""
        conv = nn.Conv2d(16, 32, kernel_size=3, bias=True)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.dtype == torch.float64
    
    def test_kernel_size_preserved(self):
        """Kernel size should be preserved."""
        conv = nn.Conv2d(16, 32, kernel_size=(5, 3))
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.kernel_size == (5, 3)
    
    def test_stride_preserved(self):
        """Stride should be preserved."""
        conv = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.stride == (2, 2)
    
    def test_padding_preserved(self):
        """Padding should be preserved."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.padding == (2, 2)


class TestTTConvEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_kernel_size_1x1(self):
        """Test with 1x1 kernel."""
        # 32 * 64 * 1 * 1 = 2048 > 1024 threshold
        conv = nn.Conv2d(32, 64, kernel_size=1)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.kernel_size == (1, 1)
    
    def test_asymmetric_kernel(self):
        """Test with asymmetric kernel size."""
        conv = nn.Conv2d(16, 32, kernel_size=(3, 5))
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.kernel_size == (3, 5)
    
    def test_large_kernel(self):
        """Test with large kernel."""
        conv = nn.Conv2d(8, 16, kernel_size=7)  # 8*16*7*7 = 6272 > 1024
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
    
    def test_asymmetric_channels(self):
        """Test with asymmetric channel counts."""
        conv = nn.Conv2d(3, 64, kernel_size=3)  # 3*64*9 = 1728 > 1024
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
    
    def test_boundary_threshold(self):
        """Test layers near the 1024 threshold."""
        # Just below threshold: 8 * 8 * 4 * 4 = 1024, but 8*8*3*3=576 < 1024
        small = nn.Conv2d(8, 8, kernel_size=3)  # 576 < 1024
        assert EncryptedTTConv2d.from_torch(small) is None
        
        # Just above threshold
        large = nn.Conv2d(8, 16, kernel_size=3)  # 8*16*9 = 1152 > 1024
        assert EncryptedTTConv2d.from_torch(large) is not None
    
    def test_same_padding(self):
        """Test 'same' padding string handling."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.padding == (1, 1)  # 3 // 2 = 1


class TestTTConvGroups:
    """Test grouped convolution support."""
    
    def test_groups_basic(self):
        """Test basic grouped convolution."""
        # groups=2: each group has (16, 8, 3, 3) = 1152 params > 1024
        conv = nn.Conv2d(16, 32, kernel_size=3, groups=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.groups == 2
        assert len(tt.tt_cores_per_group) == 2
    
    def test_groups_weight_reconstruction(self):
        """Test that grouped weight reconstruction is block-diagonal."""
        conv = nn.Conv2d(16, 32, kernel_size=3, groups=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        weight = tt._effective_weight
        
        # Weight should have block-diagonal structure
        out_per_group = tt._original_out_channels // tt.groups
        in_per_group = tt._original_in_size_per_group
        
        # Off-diagonal blocks should be zero
        # Block (0, 1) - top-right
        block_01 = weight[:out_per_group, in_per_group:2*in_per_group]
        assert torch.allclose(block_01, torch.zeros_like(block_01), atol=1e-10)
        
        # Block (1, 0) - bottom-left  
        block_10 = weight[out_per_group:2*out_per_group, :in_per_group]
        assert torch.allclose(block_10, torch.zeros_like(block_10), atol=1e-10)
    
    def test_groups_4(self):
        """Test with groups=4."""
        # groups=4: each group has (8, 4, 3, 3) = 288 params < 1024
        # Need larger layer
        conv = nn.Conv2d(32, 64, kernel_size=3, groups=4)  # each: (16, 8, 3, 3) = 1152
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.groups == 4
        assert len(tt.tt_cores_per_group) == 4
    
    def test_groups_depthwise(self):
        """Test depthwise convolution (groups=in_channels)."""
        # Depthwise: groups=in_channels, each filter is (1, 1, 3, 3) = 9 params
        # Too small for TT, should return None
        conv = nn.Conv2d(16, 16, kernel_size=3, groups=16)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is None  # 9 params < 1024 threshold
    
    def test_groups_with_bias(self):
        """Test grouped conv with bias."""
        conv = nn.Conv2d(16, 32, kernel_size=3, groups=2, bias=True)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.shape[0] >= 32
    
    def test_groups_small_per_group_returns_none(self):
        """If per-group params < 1024, return None."""
        # groups=4: each group has (4, 4, 3, 3) = 144 params < 1024
        conv = nn.Conv2d(16, 16, kernel_size=3, groups=4)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is None
    
    def test_groups_invalid_divisibility(self):
        """Test error when channels not divisible by groups."""
        # This should be caught by PyTorch, but verify our code handles it
        with pytest.raises((ValueError, RuntimeError)):
            conv = nn.Conv2d(15, 32, kernel_size=3, groups=2)  # 15 not divisible by 2
            EncryptedTTConv2d.from_torch(conv)


class TestTTConvDilation:
    """Test dilation support."""
    
    def test_dilation_basic(self):
        """Test basic dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, dilation=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.dilation == (2, 2)
    
    def test_dilation_asymmetric(self):
        """Test asymmetric dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, dilation=(2, 3))
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.dilation == (2, 3)
    
    def test_dilation_output_size(self):
        """Test output size calculation with dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, dilation=2, padding=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        # Effective kernel size = dilation * (kernel - 1) + 1 = 2 * 2 + 1 = 5
        # Output = (28 + 2*2 - 5) // 1 + 1 = 28
        out_h, out_w = tt.get_output_size(28, 28)
        assert out_h == 28
        assert out_w == 28
    
    def test_dilation_large(self):
        """Test with large dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, dilation=4)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.dilation == (4, 4)
    
    def test_dilation_with_stride(self):
        """Test dilation combined with stride."""
        conv = nn.Conv2d(16, 32, kernel_size=3, dilation=2, stride=2, padding=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.dilation == (2, 2)
        assert tt.stride == (2, 2)
        
        # Effective kernel = 5, output = (28 + 4 - 5) // 2 + 1 = 14
        out_h, out_w = tt.get_output_size(28, 28)
        assert out_h == 14
        assert out_w == 14


class TestTTConvGroupsAndDilation:
    """Test combined groups and dilation."""
    
    def test_groups_and_dilation(self):
        """Test groups with dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, groups=2, dilation=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.groups == 2
        assert tt.dilation == (2, 2)
    
    def test_groups_dilation_extra_repr(self):
        """Test extra_repr includes groups and dilation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, groups=2, dilation=2)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        repr_str = tt.extra_repr()
        assert 'groups=2' in repr_str
        assert 'dilation=(2, 2)' in repr_str


class TestTTConvCoreStructure:
    """Test internal structure of TT cores."""
    
    def test_first_core_rank_is_one(self):
        """First core should have left rank = 1."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        first_core = tt.tt_cores[0]
        assert first_core.shape[0] == 1
    
    def test_last_core_rank_is_one(self):
        """Last core should have right rank = 1."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        last_core = tt.tt_cores[-1]
        assert last_core.shape[2] == 1
    
    def test_core_rank_continuity(self):
        """Right rank of core k should equal left rank of core k+1."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for i in range(len(tt.tt_cores) - 1):
            right_rank = tt.tt_cores[i].shape[2]
            left_rank = tt.tt_cores[i + 1].shape[0]
            assert right_rank == left_rank
    
    def test_core_shapes_consistency(self):
        """Test that all cores have consistent shape structure."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert len(core.shape) == 3
            r_prev, mode_size, r_next = core.shape
            assert r_prev >= 1
            assert mode_size >= 1
            assert r_next >= 1


class TestTTConvBiasHandling:
    """Test bias handling in TT decomposition."""
    
    def test_bias_padding_matches_output(self):
        """Padded bias should match padded output channels."""
        conv = nn.Conv2d(16, 31, kernel_size=3, bias=True)  # Prime out_channels
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        assert tt.bias.shape[0] == tt.out_channels
    
    def test_bias_original_values_preserved(self):
        """Original bias values should be preserved in padded bias."""
        conv = nn.Conv2d(16, 32, kernel_size=3, bias=True)
        original_bias = conv.bias.clone().to(dtype=torch.float64)
        
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        # First 32 elements should match original
        torch.testing.assert_close(tt.bias[:32], original_bias, atol=1e-6, rtol=1e-6)
    
    def test_bias_padding_is_zero(self):
        """Padded bias elements should be zero."""
        conv = nn.Conv2d(16, 31, kernel_size=3, bias=True)  # 31 is prime
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt.bias is not None
        if tt.out_channels > 31:
            # Padded elements should be zero
            torch.testing.assert_close(
                tt.bias[31:], 
                torch.zeros(tt.out_channels - 31, dtype=torch.float64), 
                atol=1e-10, 
                rtol=1e-10
            )


class TestTTConvMaxRankBehavior:
    """Test max_rank parameter behavior."""
    
    def test_max_rank_respected(self):
        """All intermediate ranks should respect max_rank."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        max_rank = 4
        tt = EncryptedTTConv2d.from_torch(conv, max_rank=max_rank)
        
        assert tt is not None
        # Check intermediate ranks (not first or last)
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] <= max_rank
    
    def test_max_rank_one(self):
        """Test with max_rank=1 (minimal decomposition)."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, max_rank=1)
        
        assert tt is not None
        # All intermediate ranks should be 1
        for core in tt.tt_cores[:-1]:
            assert core.shape[2] == 1
    
    def test_max_rank_large(self):
        """Test with very large max_rank."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, max_rank=1000)
        
        assert tt is not None


class TestTTConvForward:
    """Tests for EncryptedTTConv2d forward() method with weight reconstruction."""
    
    def test_forward_weight_reconstruction_matches_shape(self):
        """Verify _effective_weight has correct shape."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        assert tt._effective_weight is not None
        # Shape should be (padded_out, padded_in)
        assert tt._effective_weight.shape[0] == tt.out_channels
        assert tt._effective_weight.shape[1] >= 16 * 3 * 3  # in_ch * kH * kW
    
    def test_forward_weight_reconstruction_is_deterministic(self):
        """Verify reconstruction is deterministic."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        reconstructed_again = tt._reconstruct_weight()
        torch.testing.assert_close(
            tt._effective_weight,
            reconstructed_again,
            rtol=1e-10,
            atol=1e-12
        )
    
    def test_forward_input_validation_1d(self):
        """Verify 1D forward works with correct matmul."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        assert tt is not None
        
        # Create a mock EncryptedTensor
        class MockEncryptedTensor:
            def __init__(self, shape: tuple) -> None:
                self.shape = shape
                self._cnn_layout = None
            
            def matmul(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "MockEncryptedTensor":
                out_size = weight.shape[0]
                result = MockEncryptedTensor((out_size,))
                result._cnn_layout = None
                return result
        
        # 1D input: single flattened patch (in_ch * kH * kW = 16 * 3 * 3 = 144)
        input_tensor = MockEncryptedTensor((tt._flat_in_size,))
        output = tt.forward(input_tensor)  # type: ignore
        
        assert output.shape == (tt.out_channels,)
    
    def test_forward_raises_on_4d_input(self):
        """Verify RuntimeError is raised on 4D input."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        assert tt is not None
        
        class MockEncryptedTensor:
            def __init__(self, shape: tuple) -> None:
                self.shape = shape
        
        input_4d = MockEncryptedTensor((1, 16, 28, 28))
        
        with pytest.raises(RuntimeError, match="4D"):
            tt.forward(input_4d)  # type: ignore
    
    def test_forward_raises_on_2d_without_cnn_layout(self):
        """Verify RuntimeError is raised on 2D input without CNN layout."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        assert tt is not None
        
        class MockEncryptedTensor:
            def __init__(self, shape: tuple) -> None:
                self.shape = shape
                self._cnn_layout = None
        
        input_2d = MockEncryptedTensor((49, 144))  # (num_patches, patch_features)
        
        with pytest.raises(RuntimeError, match="CNN layout"):
            tt.forward(input_2d)  # type: ignore


class TestTTConvOutputSize:
    """Test output size calculation."""
    
    def test_get_output_size_basic(self):
        """Test basic output size calculation."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        out_h, out_w = tt.get_output_size(28, 28)
        assert out_h == 28  # Same padding
        assert out_w == 28
    
    def test_get_output_size_no_padding(self):
        """Test output size without padding."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        out_h, out_w = tt.get_output_size(28, 28)
        assert out_h == 26  # 28 - 3 + 1 = 26
        assert out_w == 26
    
    def test_get_output_size_with_stride(self):
        """Test output size with stride."""
        conv = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        out_h, out_w = tt.get_output_size(28, 28)
        assert out_h == 14  # (28 + 2*1 - 3) // 2 + 1 = 14
        assert out_w == 14


class TestTTConvConverterIntegration:
    """Test integration with model converter."""
    
    def test_from_torch_returns_encrypted_tt_conv2d(self):
        """from_torch should return EncryptedTTConv2d instance."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert isinstance(tt, EncryptedTTConv2d)
    
    def test_from_torch_detaches_weights(self):
        """Weights should be detached from computation graph."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        conv.weight.requires_grad = True
        
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert not core.requires_grad
    
    def test_from_torch_moves_to_cpu(self):
        """Weights should be on CPU."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
        for core in tt.tt_cores:
            assert core.device.type == 'cpu'
        if tt.bias is not None:
            assert tt.bias.device.type == 'cpu'


class TestTTConvSVDThreshold:
    """Test SVD threshold parameter."""
    
    def test_svd_threshold_affects_ranks(self):
        """Verify custom svd_threshold parameter affects decomposition."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        
        # Create two TT layers with different SVD thresholds
        tt_loose = EncryptedTTConv2d.from_torch(conv, svd_threshold=1e-3)
        tt_strict = EncryptedTTConv2d.from_torch(conv, svd_threshold=1e-8)
        
        assert tt_loose is not None
        assert tt_strict is not None
        
        # Count total rank across all cores
        loose_rank_sum = sum(core.shape[2] for core in tt_loose.tt_cores[:-1])
        strict_rank_sum = sum(core.shape[2] for core in tt_strict.tt_cores[:-1])
        
        # Strict threshold should keep more singular values (higher ranks)
        assert strict_rank_sum >= loose_rank_sum


class TestTTConvDifferentSizes:
    """Test TT decomposition with various layer sizes."""
    
    def test_wide_layer(self):
        """Test with more input channels than output."""
        conv = nn.Conv2d(64, 16, kernel_size=3)  # 64*16*9 = 9216 > 1024
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
    
    def test_deep_layer(self):
        """Test with many output channels."""
        conv = nn.Conv2d(8, 128, kernel_size=3)  # 8*128*9 = 9216 > 1024
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None
    
    def test_large_layer(self):
        """Test with large layer."""
        conv = nn.Conv2d(64, 128, kernel_size=3)  # 64*128*9 = 73728 > 1024
        tt = EncryptedTTConv2d.from_torch(conv)
        
        assert tt is not None


class TestPermutationOptimization:
    """Test permutation optimization feature (Gabor & Zdunek 2022 style)."""
    
    def test_tt_parameter_accepted(self):
        """Verify TT parameter is accepted."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        assert tt is not None
    
    def test_default_behavior_unchanged(self):
        """Verify default behavior (TT=False) is backward compatible."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv)
        assert tt is not None
        assert tt._kernel_permutation is None
        assert tt._inverse_permutation is None
    
    def test_permutation_stored_correctly(self):
        """Verify permutation is stored when TT enabled."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt is not None
        assert tt._kernel_permutation is not None
        assert tt._inverse_permutation is not None
        assert len(tt._kernel_permutation) == 4
        assert len(tt._inverse_permutation) == 4
    
    def test_inverse_permutation_is_correct(self):
        """Verify inverse permutation correctly inverts the permutation."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt is not None
        perm = tt._kernel_permutation
        inv = tt._inverse_permutation
        assert perm is not None
        assert inv is not None
        
        composed = tuple(perm[inv[i]] for i in range(4))
        assert composed == (0, 1, 2, 3)
    
    def test_optimization_produces_valid_results(self):
        """Verify optimization produces valid reconstruction."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        
        tt_baseline = EncryptedTTConv2d.from_torch(conv, TT=False)
        tt_optimized = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt_baseline is not None
        assert tt_optimized is not None
        
        orig = conv.weight.reshape(32, -1).to(torch.float64)
        orig_in_size = 16 * 3 * 3
        
        baseline_err = torch.norm(orig - tt_baseline._effective_weight[:32, :orig_in_size]) / torch.norm(orig)
        optimized_err = torch.norm(orig - tt_optimized._effective_weight[:32, :orig_in_size]) / torch.norm(orig)
        
        # Both should have reasonable error (< 50%)
        assert baseline_err < 0.5
        assert optimized_err < 0.5
    
    def test_reconstruction_shape_correct(self):
        """Verify reconstructed weight has correct shape."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt is not None
        orig_in_size = 16 * 3 * 3
        
        assert tt._effective_weight.shape[0] >= 32
        assert tt._effective_weight.shape[1] >= orig_in_size
    
    def test_optimization_with_grouped_conv(self):
        """Verify optimization works with grouped convolutions."""
        conv = nn.Conv2d(32, 64, kernel_size=3, groups=2)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt is not None
        assert tt.groups == 2
    
    def test_optimization_with_different_kernel_sizes(self):
        """Verify optimization works with various kernel sizes."""
        for kernel_size in [3, 5, 7]:
            conv = nn.Conv2d(16, 32, kernel_size=kernel_size)
            tt = EncryptedTTConv2d.from_torch(conv, TT=True)
            assert tt is not None
    
    def test_permutation_starts_with_output_dimension(self):
        """Verify permutation always keeps output dimension first."""
        conv = nn.Conv2d(16, 32, kernel_size=3)
        tt = EncryptedTTConv2d.from_torch(conv, TT=True)
        
        assert tt is not None
        assert tt._kernel_permutation is not None
        assert tt._kernel_permutation[0] == 0


# Run with: pytest tests/test_tt_conv.py -v
