"""Tests for model conversion."""

import pytest
import warnings
import torch
import torch.nn as nn

from ckks_torch.converter import (
    ModelConverter,
    ConversionOptions,
    convert,
    estimate_depth,
)
from ckks_torch.nn import (
    EncryptedLinear,
    EncryptedSquare,
    EncryptedReLU,
    EncryptedSequential,
    EncryptedDropout,
    EncryptedMaxPool2d,
)


class TestModelConverter:
    """Test ModelConverter class."""
    
    def test_convert_simple_linear(self):
        """Test converting a single Linear layer."""
        model = nn.Linear(10, 5)
        model.eval()
        
        converter = ModelConverter()
        enc_model = converter.convert(model)
        
        assert isinstance(enc_model, EncryptedLinear)
        assert enc_model.in_features == 10
        assert enc_model.out_features == 5
    
    def test_convert_sequential_mlp(self):
        """Test converting a Sequential MLP."""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        model.eval()
        
        converter = ModelConverter(ConversionOptions(use_square_activation=True))
        enc_model = converter.convert(model)
        
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 3
        assert isinstance(enc_model[0], EncryptedLinear)
        assert isinstance(enc_model[1], EncryptedSquare)
        assert isinstance(enc_model[2], EncryptedLinear)
    
    def test_convert_with_relu_approximation(self):
        """Test converting with ReLU polynomial approximation."""
        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        model.eval()
        
        options = ConversionOptions(
            use_square_activation=False,
            activation_degree=4,
        )
        converter = ModelConverter(options)
        enc_model = converter.convert(model)
        
        assert isinstance(enc_model, EncryptedSequential)
        assert isinstance(enc_model[1], EncryptedReLU)
        assert enc_model[1].degree == 4
    
    def test_batchnorm_folding(self):
        """Test that BatchNorm is folded into Linear."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        model.eval()
        
        # Run a forward pass to populate running stats
        with torch.no_grad():
            model(torch.randn(5, 10))
        
        converter = ModelConverter(ConversionOptions(
            fold_batchnorm=True,
            use_square_activation=True,
        ))
        enc_model = converter.convert(model)
        
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 2
        assert isinstance(enc_model[0], EncryptedLinear)
        assert isinstance(enc_model[1], EncryptedSquare)
    
    def test_weights_preserved(self):
        """Test that weights are correctly copied."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        model.eval()
        
        converter = ModelConverter()
        enc_model = converter.convert(model)
        
        # Check weight values match
        assert torch.allclose(
            enc_model.weight.float(),
            model.weight.data.float(),
            atol=1e-6,
        )
        
        if model.bias is not None:
            assert torch.allclose(
                enc_model.bias.float(),
                model.bias.data.float(),
                atol=1e-6,
            )


class TestConvertFunction:
    """Test the convert() convenience function."""
    
    def test_convert_returns_tuple(self):
        """Test that convert() returns (model, context)."""
        model = nn.Linear(10, 5)
        model.eval()
        
        result = convert(model)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        enc_model, ctx = result
        assert isinstance(enc_model, EncryptedLinear)
    
    def test_convert_with_options(self):
        """Test convert() with custom options."""
        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        model.eval()
        
        enc_model, ctx = convert(
            model,
            use_square_activation=True,
            fold_batchnorm=True,
        )
        
        assert isinstance(enc_model, EncryptedSequential)
        assert isinstance(enc_model[1], EncryptedSquare)


class TestEstimateDepth:
    """Test depth estimation."""
    
    def test_single_linear(self):
        """Single linear layer has depth 1."""
        model = nn.Linear(10, 5)
        assert estimate_depth(model) == 1
    
    def test_mlp_with_activations(self):
        """MLP with activations should count both layers and activations."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        # 2 linear (1 each) + 2 relu (2 each for polynomial approx) = 2 + 4 = 6
        assert estimate_depth(model) == 6
    
    def test_empty_model(self):
        """Empty model should return at least 1."""
        model = nn.Sequential()
        assert estimate_depth(model) >= 1


class TestDropoutConversion:
    """Test Dropout layer conversion."""
    
    def test_dropout_warning(self):
        """Test that converting Dropout emits a warning."""
        model = nn.Dropout(p=0.5)
        model.eval()
        
        converter = ModelConverter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enc_model = converter.convert(model)
            
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Dropout layers are ignored during inference (no-op)" in str(w[0].message)
        
        assert isinstance(enc_model, EncryptedDropout)
    
    def test_model_with_dropout_converts(self):
        """Test that a model with Dropout layers converts successfully."""
        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 4),
        )
        model.eval()
        
        converter = ModelConverter(ConversionOptions(use_square_activation=True))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enc_model = converter.convert(model)
            
            dropout_warnings = [x for x in w if "Dropout" in str(x.message)]
            assert len(dropout_warnings) == 1
        
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 4
        assert isinstance(enc_model[0], EncryptedLinear)
        assert isinstance(enc_model[1], EncryptedSquare)
        assert isinstance(enc_model[2], EncryptedDropout)
        assert isinstance(enc_model[3], EncryptedLinear)


class TestMaxPool2dConversion:

    def test_model_with_maxpool_converts(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        model.eval()
        
        converter = ModelConverter(ConversionOptions(use_square_activation=True))
        enc_model = converter.convert(model)
        
        assert isinstance(enc_model, EncryptedSequential)
        assert len(enc_model) == 4
        assert isinstance(enc_model[2], EncryptedMaxPool2d)
        assert enc_model[2].kernel_size == (2, 2)
        assert enc_model[2].stride == (2, 2)
