"""Tests for specific bug fixes.

Each test class targets a specific bug that was identified and fixed.
Tests are designed to fail if the bug regresses.
"""

import copy
import threading
import pytest
import torch

from collections import OrderedDict


class TestScaleMismatchInAddSub:
    """Bug: tensor.py add/sub failed when operands had different _needs_rescale states.
    
    When one operand has _needs_rescale=True (scale=Δ²) and the other has 
    _needs_rescale=False (scale=Δ), CKKS addition fails due to scale mismatch.
    Fix: Auto-rescale the operand with _needs_rescale=True before operation.
    """

    def test_add_with_different_rescale_states(self, mock_enc_context):
        """Add tensor after mul (needs_rescale=True) with fresh tensor (needs_rescale=False)."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))
        
        a_mul = a.mul(2.0)
        assert a_mul._needs_rescale is True
        assert b._needs_rescale is False
        
        result = a_mul.add(b)
        decrypted = mock_enc_context.decrypt(result)
        
        expected = torch.tensor([2.0, 4.0, 6.0]) + torch.tensor([4.0, 5.0, 6.0])
        torch.testing.assert_close(decrypted, expected, rtol=1e-4, atol=1e-4)

    def test_add_fresh_to_multiplied(self, mock_enc_context):
        """Add fresh tensor to tensor after mul (reversed operand order)."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        b = mock_enc_context.encrypt(torch.tensor([4.0, 5.0, 6.0]))
        
        b_mul = b.mul(3.0)
        assert b_mul._needs_rescale is True
        assert a._needs_rescale is False
        
        result = a.add(b_mul)
        decrypted = mock_enc_context.decrypt(result)
        
        expected = torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([12.0, 15.0, 18.0])
        torch.testing.assert_close(decrypted, expected, rtol=1e-4, atol=1e-4)

    def test_sub_with_different_rescale_states(self, mock_enc_context):
        """Sub tensor after mul from fresh tensor."""
        a = mock_enc_context.encrypt(torch.tensor([10.0, 20.0, 30.0]))
        b = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0]))
        
        b_mul = b.mul(2.0)
        assert b_mul._needs_rescale is True
        
        result = a.sub(b_mul)
        decrypted = mock_enc_context.decrypt(result)
        
        expected = torch.tensor([10.0, 20.0, 30.0]) - torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(decrypted, expected, rtol=1e-4, atol=1e-4)

    def test_chained_operations_with_mixed_rescale(self, mock_enc_context):
        """Complex chain: (a * 2) + (b * 3) - c."""
        a = mock_enc_context.encrypt(torch.tensor([1.0, 2.0]))
        b = mock_enc_context.encrypt(torch.tensor([3.0, 4.0]))
        c = mock_enc_context.encrypt(torch.tensor([0.5, 0.5]))
        
        a_scaled = a.mul(2.0)
        b_scaled = b.mul(3.0)
        
        sum_ab = a_scaled.add(b_scaled)
        result = sum_ab.sub(c)
        
        decrypted = mock_enc_context.decrypt(result)
        expected = (torch.tensor([1.0, 2.0]) * 2 + torch.tensor([3.0, 4.0]) * 3 
                    - torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(decrypted, expected, rtol=1e-3, atol=1e-3)


class TestMaxPoolLevelDrift:
    """Bug: pooling.py MaxPool accumulated level drift in iterative loop.
    
    In _approx_max, sum_ab could have _needs_rescale=True while abs_diff 
    had _needs_rescale=False, causing scale mismatch when adding them.
    Fix: Rescale sum_ab if it needs rescaling before adding abs_diff.
    """

    def test_maxpool_2x2_basic(self, mock_enc_context):
        """Basic 2x2 maxpool should not crash due to level mismatch."""
        from cukks.nn import EncryptedMaxPool2d
        
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2)
        
        x = mock_enc_context.encrypt(torch.randn(16))
        x._shape = (4, 4)
        x._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
            'height': 2,
            'width': 2,
        }
        
        result = pool(x)
        assert result is not None
        assert result._cnn_layout is not None

    def test_approx_max_single_pair(self, mock_enc_context):
        """Test _approx_max with two tensors at different states."""
        from cukks.nn.pooling import EncryptedMaxPool2d
        
        pool = EncryptedMaxPool2d(kernel_size=2, stride=2)
        
        a = mock_enc_context.encrypt(torch.tensor([3.0, 1.0, 4.0, 1.0]))
        b = mock_enc_context.encrypt(torch.tensor([2.0, 7.0, 1.0, 8.0]))
        
        result = pool._approx_max(a, b)
        decrypted = mock_enc_context.decrypt(result)
        
        assert decrypted.shape[0] == 4


class TestSoftmaxSeqLen1:
    """Bug: attention.py seq_len=1 softmax returned exp(score) instead of 1.0.
    
    For softmax of a single element, the result should always be 1.0
    regardless of the input value: softmax([x]) = [1.0].
    Fix: Special case seq_len=1 to return tensor of 1.0s.
    """

    def test_softmax_single_element_returns_one(self, mock_enc_context):
        """softmax([x]) should return [1.0] for any x."""
        from cukks.nn import EncryptedApproxAttention
        
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=1)
        
        scores = mock_enc_context.encrypt(torch.tensor([5.0, 5.0, 5.0, 5.0]))
        
        result = attn._approx_softmax_row(scores, seq_len=1)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(
            decrypted, 
            torch.ones(4), 
            rtol=1e-4, atol=1e-4
        )

    def test_softmax_single_element_negative_input(self, mock_enc_context):
        """softmax([x]) = [1.0] even for negative x."""
        from cukks.nn import EncryptedApproxAttention
        
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=1)
        
        scores = mock_enc_context.encrypt(torch.tensor([-10.0, -10.0, -10.0, -10.0]))
        
        result = attn._approx_softmax_row(scores, seq_len=1)
        decrypted = mock_enc_context.decrypt(result)
        
        torch.testing.assert_close(
            decrypted, 
            torch.ones(4), 
            rtol=1e-4, atol=1e-4
        )

    def test_softmax_seq_len_zero_raises(self, mock_enc_context):
        """seq_len=0 should raise ValueError."""
        from cukks.nn import EncryptedApproxAttention
        
        attn = EncryptedApproxAttention(embed_dim=4, num_heads=1)
        scores = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        with pytest.raises(ValueError, match="seq_len must be positive"):
            attn._approx_softmax_row(scores, seq_len=0)


class TestConvMemoryExplosion:
    """Bug: conv.py created dense block-diagonal matrix causing memory explosion.
    
    For large images (e.g., 224x224), the block-diagonal matrix could require
    ~92TB of memory, causing OOM crashes.
    Fix: Guard with element count limit and clear error message.
    """

    def test_small_conv_works(self, mock_enc_context):
        """Small convolution should work normally."""
        from cukks.nn import EncryptedConv2d
        
        weight = torch.randn(4, 1, 3, 3)
        conv = EncryptedConv2d(
            in_channels=1, out_channels=4,
            kernel_size=3, weight=weight, stride=1, padding=1
        )
        
        x = mock_enc_context.encrypt(torch.randn(16))
        x._shape = (16,)
        x._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
        }
        
        result = conv(x)
        assert result is not None

    def test_compact_conv_matches_dense(self, mock_enc_context):
        """Compact diagonal conv path should produce same results as dense path."""
        from cukks.nn import EncryptedConv2d

        in_ch, out_ch, ksize = 2, 4, 3
        weight = torch.randn(out_ch, in_ch, ksize, ksize)
        conv = EncryptedConv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=ksize, weight=weight, stride=1, padding=1,
        )

        num_patches = 4
        patch_features = in_ch * ksize * ksize
        total_in = num_patches * patch_features

        input_data = torch.randn(total_in)
        total_out = num_patches * out_ch

        x_dense = mock_enc_context.encrypt(input_data)
        x_dense._shape = (num_patches, patch_features)
        x_dense._cnn_layout = {'num_patches': num_patches, 'patch_features': patch_features}

        x_compact = mock_enc_context.encrypt(input_data)
        x_compact._shape = (num_patches, patch_features)
        x_compact._cnn_layout = {'num_patches': num_patches, 'patch_features': patch_features}

        result_dense = conv._forward_he_packed_dense(
            x_dense, num_patches, patch_features, total_out, total_in
        )
        result_compact = conv._forward_he_packed_compact(
            x_compact, num_patches, patch_features
        )

        dec_dense = mock_enc_context.decrypt(result_dense, shape=(total_out,))
        dec_compact = mock_enc_context.decrypt(result_compact, shape=(total_out,))
        torch.testing.assert_close(dec_dense, dec_compact, atol=1e-6, rtol=1e-5)


class TestGlobalStateMutation:
    """Bug: converter.py DEFAULT_ACTIVATION_MAP was shared, causing mutation.
    
    When ConversionOptions used `activation_map or DEFAULT_ACTIVATION_MAP`,
    modifying the map would affect all future conversions.
    Fix: Use .copy() when using the default map.
    """

    def test_modifying_options_does_not_affect_default(self):
        """Modifying one ConversionOptions should not affect others."""
        from cukks.converter import ConversionOptions, DEFAULT_ACTIVATION_MAP
        from cukks.nn import EncryptedSquare
        import torch.nn as nn
        
        original_relu_converter = DEFAULT_ACTIVATION_MAP.get(nn.ReLU)
        
        opts1 = ConversionOptions()
        opts1.activation_map[nn.ReLU] = EncryptedSquare
        
        opts2 = ConversionOptions()
        
        assert opts2.activation_map.get(nn.ReLU) == original_relu_converter
        assert DEFAULT_ACTIVATION_MAP.get(nn.ReLU) == original_relu_converter

    def test_two_converters_independent(self):
        """Two converters should have independent activation maps."""
        from cukks.converter import ConversionOptions
        from cukks.nn import EncryptedSquare, EncryptedSigmoid
        import torch.nn as nn
        
        opts1 = ConversionOptions()
        opts2 = ConversionOptions()
        
        opts1.activation_map[nn.ReLU] = EncryptedSquare
        opts2.activation_map[nn.ReLU] = EncryptedSigmoid
        
        assert opts1.activation_map[nn.ReLU] != opts2.activation_map[nn.ReLU]


class TestThreadSafeInitialization:
    """Bug: context.py _ensure_initialized had race condition.
    
    Two threads could both pass the _initialized check and both create
    CKKSContext instances.
    Fix: Double-checked locking with threading.Lock.
    """

    def test_context_has_init_lock(self):
        """CKKSInferenceContext should have _init_lock attribute."""
        from cukks.context import CKKSInferenceContext
        
        ctx = CKKSInferenceContext()
        
        assert hasattr(ctx, '_init_lock')
        assert isinstance(ctx._init_lock, type(threading.Lock()))

    def test_concurrent_initialization_safe(self):
        """Multiple threads calling _ensure_initialized should be safe."""
        from cukks.context import CKKSInferenceContext
        
        ctx = CKKSInferenceContext()
        init_count = [0]
        original_initialized = ctx._initialized
        
        def init_thread():
            ctx._ensure_initialized()
            init_count[0] += 1
        
        threads = [threading.Thread(target=init_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert init_count[0] == 10
        assert ctx._initialized is True


class TestSequentialTypeValidation:
    """Bug: sequential.py didn't validate OrderedDict values were EncryptedModule.
    
    Passing an OrderedDict with non-EncryptedModule values would silently add them.
    Fix: Validate each value is EncryptedModule.
    """

    def test_valid_ordered_dict_works(self):
        """Valid OrderedDict should work."""
        from cukks.nn import EncryptedSequential, EncryptedLinear
        
        weight = torch.randn(10, 10)
        modules = OrderedDict([
            ('fc1', EncryptedLinear(10, 10, weight)),
            ('fc2', EncryptedLinear(10, 10, weight)),
        ])
        
        seq = EncryptedSequential(modules)
        assert len(seq) == 2

    def test_invalid_ordered_dict_raises(self):
        """OrderedDict with non-EncryptedModule should raise TypeError."""
        from cukks.nn import EncryptedSequential, EncryptedLinear
        
        weight = torch.randn(10, 10)
        modules = OrderedDict([
            ('fc1', EncryptedLinear(10, 10, weight)),
            ('invalid', "not a module"),
        ])
        
        with pytest.raises(TypeError, match="Expected EncryptedModule"):
            EncryptedSequential(modules)

    def test_invalid_ordered_dict_shows_key_name(self):
        """Error message should include the problematic key name."""
        from cukks.nn import EncryptedSequential
        
        modules = OrderedDict([
            ('bad_layer', 12345),
        ])
        
        with pytest.raises(TypeError, match="bad_layer"):
            EncryptedSequential(modules)


class TestBatchNorm2dCNNLayout:
    """Bug: batchnorm.py didn't handle CNN packed layout correctly.
    
    EncryptedBatchNorm2d.forward() with CNN layout would produce wrong results
    because parameters weren't replicated across patches.
    Fix: Raise clear error suggesting to use fold_batchnorm=True.
    """

    def test_batchnorm2d_with_cnn_layout_raises(self, mock_enc_context):
        """BatchNorm2d with CNN layout should raise RuntimeError."""
        from cukks.nn import EncryptedBatchNorm2d
        
        bn = EncryptedBatchNorm2d(
            num_features=4,
            scale=torch.ones(4),
            shift=torch.zeros(4),
        )
        
        x = mock_enc_context.encrypt(torch.randn(16))
        x._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
        }
        
        with pytest.raises(RuntimeError, match="CNN packed layout"):
            bn(x)

    def test_batchnorm2d_without_cnn_layout_works(self, mock_enc_context):
        """BatchNorm2d without CNN layout should work normally."""
        from cukks.nn import EncryptedBatchNorm2d
        
        bn = EncryptedBatchNorm2d(
            num_features=4,
            scale=torch.tensor([2.0, 2.0, 2.0, 2.0]),
            shift=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
        
        x = mock_enc_context.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        
        result = bn(x)
        decrypted = mock_enc_context.decrypt(result)
        
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2.0 + 1.0
        torch.testing.assert_close(decrypted, expected, rtol=1e-4, atol=1e-4)


class TestSizeValidationInOperations:
    """Bug: tensor.py didn't validate operand sizes in add/sub/mul.
    
    Adding tensors with different numbers of elements would produce
    incorrect results without any warning.
    Fix: Validate element counts match (allowing different shapes with same size).
    """

    def test_add_same_size_different_shape_works(self, mock_enc_context):
        """Adding (64,) and (1, 64) should work (same element count)."""
        a = mock_enc_context.encrypt(torch.randn(64))
        b = mock_enc_context.encrypt(torch.randn(1, 64))
        
        result = a.add(b)
        assert result.size == 64

    def test_add_different_sizes_raises(self, mock_enc_context):
        """Adding tensors with different element counts should raise."""
        a = mock_enc_context.encrypt(torch.randn(64))
        b = mock_enc_context.encrypt(torch.randn(32))
        
        with pytest.raises(ValueError, match="Size mismatch"):
            a.add(b)

    def test_sub_different_sizes_raises(self, mock_enc_context):
        """Subtracting tensors with different element counts should raise."""
        a = mock_enc_context.encrypt(torch.randn(100))
        b = mock_enc_context.encrypt(torch.randn(50))
        
        with pytest.raises(ValueError, match="Size mismatch"):
            a.sub(b)

    def test_mul_different_sizes_raises(self, mock_enc_context):
        """Multiplying tensors with different element counts should raise."""
        a = mock_enc_context.encrypt(torch.randn(16))
        b = mock_enc_context.encrypt(torch.randn(32))
        
        with pytest.raises(ValueError, match="Size mismatch"):
            a.mul(b)

    def test_error_message_includes_sizes(self, mock_enc_context):
        """Error message should include both sizes for debugging."""
        a = mock_enc_context.encrypt(torch.randn(100))
        b = mock_enc_context.encrypt(torch.randn(50))
        
        with pytest.raises(ValueError, match="100 elements.*50 elements"):
            a.add(b)


class TestDeepCopyCNNLayout:
    """Bug: tensor.py used shallow copy for _cnn_layout, causing shared state.
    
    Operations like add/mul/rotate returned new tensors but with shared
    _cnn_layout dict, so modifying one affected others.
    Fix: Use copy.deepcopy for _cnn_layout propagation.
    """

    def test_add_preserves_independent_layout(self, mock_enc_context):
        """After add, modifying result's layout shouldn't affect original."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
            'nested': {'key': 'original'},
        }
        
        b = mock_enc_context.encrypt(torch.randn(16))
        result = a.add(b)
        
        result._cnn_layout['nested']['key'] = 'modified'
        
        assert a._cnn_layout['nested']['key'] == 'original'

    def test_mul_preserves_independent_layout(self, mock_enc_context):
        """After mul, layouts should be independent."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
            'metadata': [1, 2, 3],
        }
        
        result = a.mul(2.0)
        
        result._cnn_layout['metadata'].append(4)
        
        assert len(a._cnn_layout['metadata']) == 3

    def test_rotate_preserves_independent_layout(self, mock_enc_context):
        """After rotate, layouts should be independent."""
        a = mock_enc_context.encrypt(torch.randn(16))
        a._cnn_layout = {
            'num_patches': 4,
            'patch_features': 4,
            'info': {'value': 100},
        }
        
        result = a.rotate(1)
        
        result._cnn_layout['info']['value'] = 999
        
        assert a._cnn_layout['info']['value'] == 100


class TestPoolingInputValidation:
    """Bug: pooling.py allowed kernel_size=0 and stride=0.
    
    Zero kernel_size or stride would cause division by zero or other errors.
    Fix: Validate in __init__ that kernel_size and stride are positive.
    """

    def test_avgpool_zero_kernel_raises(self):
        """AvgPool2d with kernel_size=0 should raise ValueError."""
        from cukks.nn import EncryptedAvgPool2d
        
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            EncryptedAvgPool2d(kernel_size=0)

    def test_avgpool_zero_stride_raises(self):
        """AvgPool2d with stride=0 should raise ValueError."""
        from cukks.nn import EncryptedAvgPool2d
        
        with pytest.raises(ValueError, match="stride must be positive"):
            EncryptedAvgPool2d(kernel_size=2, stride=0)

    def test_maxpool_zero_kernel_raises(self):
        """MaxPool2d with kernel_size=0 should raise ValueError."""
        from cukks.nn import EncryptedMaxPool2d
        
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            EncryptedMaxPool2d(kernel_size=0)

    def test_maxpool_negative_kernel_raises(self):
        """MaxPool2d with negative kernel_size should raise ValueError."""
        from cukks.nn import EncryptedMaxPool2d
        
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            EncryptedMaxPool2d(kernel_size=-2)


class TestPoolingRectangularInput:
    """Bug: pooling.py assumed square inputs (sqrt of num_patches).
    
    For rectangular images, int(sqrt(num_patches)) would be wrong.
    Fix: Check for height/width in layout, raise clear error if missing.
    """

    def test_avgpool_with_height_width_works(self, mock_enc_context):
        """AvgPool with explicit height/width in layout should work."""
        from cukks.nn import EncryptedAvgPool2d
        
        pool = EncryptedAvgPool2d(kernel_size=2, stride=2)
        
        x = mock_enc_context.encrypt(torch.randn(32))
        x._shape = (8, 4)
        x._cnn_layout = {
            'num_patches': 8,
            'patch_features': 4,
            'height': 4,
            'width': 2,
        }
        
        result = pool._forward_he_packed(x)
        assert result is not None

    def test_avgpool_rectangular_without_dims_raises(self, mock_enc_context):
        """AvgPool with rectangular input but no height/width should raise."""
        from cukks.nn import EncryptedAvgPool2d
        
        pool = EncryptedAvgPool2d(kernel_size=2, stride=2)
        
        x = mock_enc_context.encrypt(torch.randn(24))
        x._shape = (6, 4)
        x._cnn_layout = {
            'num_patches': 6,
            'patch_features': 4,
        }
        
        with pytest.raises(ValueError, match="not a perfect square"):
            pool._forward_he_packed(x)

