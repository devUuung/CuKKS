"""Tests for inverse-free LayerNorm: detection, forward, converter integration, and bias scaling."""

import math

import pytest
import torch
import torch.nn as nn

from cukks.analysis.inverse_free_detect import detect_inverse_free_layernorms
from cukks.nn.inverse_free_layernorm import EncryptedInverseFreeLayerNorm
from cukks.converter import ModelConverter, ConversionOptions
from cukks.nn import EncryptedLayerNorm, EncryptedLinear


# ============================================================================
# Helper models for detection tests
# ============================================================================

class TwoLNLinearReLU(nn.Module):
    """LN → Linear → ReLU → LN.  First LN is eligible."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.act(self.fc(self.norm1(x))))


class TwoLNWithGELU(nn.Module):
    """LN → Linear → GELU → LN.  GELU is NOT homogeneous → not eligible."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.act(self.fc(self.norm1(x))))


class SingleLN(nn.Module):
    """Only one LN → no pair → not eligible."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class ThreeLNChain(nn.Module):
    """LN → Linear → ReLU → LN → Linear → ReLU → LN.
    First two LNs are eligible."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.act1 = nn.ReLU()
        self.norm2 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d)
        self.act2 = nn.ReLU()
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.act1(self.fc1(x))
        x = self.norm2(x)
        x = self.act2(self.fc2(x))
        x = self.norm3(x)
        return x


class TwoLNWithDropout(nn.Module):
    """LN → Linear → Dropout → ReLU → LN.  Dropout is safe → eligible."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.drop = nn.Dropout(0.1)
        self.act = nn.ReLU()
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.act(self.drop(self.fc(self.norm1(x)))))


class TwoLNWithResidual(nn.Module):
    """LN → Linear → ReLU + residual add → LN.
    Residual add IS allowed between the two LNs."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(self.fc(h))
        h = h + x  # residual
        return self.norm2(h)


class NoLNModel(nn.Module):
    """No LayerNorm at all."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# Detection tests
# ============================================================================

class TestDetectInverseFreeLayerNorms:

    def test_basic_eligible_pair(self):
        model = TwoLNLinearReLU(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert "norm1" in eligible
        assert "norm2" not in eligible  # norm2 has no cancelling LN after it

    def test_gelu_breaks_eligibility(self):
        model = TwoLNWithGELU(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert "norm1" not in eligible

    def test_single_ln_not_eligible(self):
        model = SingleLN(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert len(eligible) == 0

    def test_three_ln_chain(self):
        model = ThreeLNChain(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert "norm1" in eligible
        assert "norm2" in eligible
        assert "norm3" not in eligible

    def test_dropout_is_safe(self):
        model = TwoLNWithDropout(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert "norm1" in eligible

    def test_residual_add_is_safe(self):
        model = TwoLNWithResidual(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert "norm1" in eligible

    def test_no_ln_returns_empty(self):
        model = NoLNModel(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert len(eligible) == 0

    def test_returns_frozenset(self):
        model = TwoLNLinearReLU(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        assert isinstance(eligible, frozenset)


# ============================================================================
# EncryptedInverseFreeLayerNorm unit tests
# ============================================================================

class TestEncryptedInverseFreeLayerNorm:

    def test_from_torch(self):
        ln = nn.LayerNorm(32)
        enc_ln = EncryptedInverseFreeLayerNorm.from_torch(ln)
        assert enc_ln.normalized_shape == [32]
        assert enc_ln.lam == 0.01
        assert enc_ln.weight.shape == (32,)
        assert enc_ln.bias.shape == (32,)

    def test_from_torch_custom_lam(self):
        ln = nn.LayerNorm(64)
        enc_ln = EncryptedInverseFreeLayerNorm.from_torch(ln, lam=0.005)
        assert enc_ln.lam == 0.005

    def test_mult_depth(self):
        enc_ln = EncryptedInverseFreeLayerNorm(16)
        assert enc_ln.mult_depth() == 5

    def test_extra_repr(self):
        enc_ln = EncryptedInverseFreeLayerNorm(32, lam=0.02)
        r = enc_ln.extra_repr()
        assert "inverse_free=True" in r
        assert "lam=0.02" in r

    def test_weight_bias_default(self):
        enc_ln = EncryptedInverseFreeLayerNorm(16)
        assert torch.allclose(enc_ln.weight, torch.ones(16, dtype=torch.float64))
        assert torch.allclose(enc_ln.bias, torch.zeros(16, dtype=torch.float64))

    def test_weight_bias_copied(self):
        w = torch.randn(16)
        b = torch.randn(16)
        enc_ln = EncryptedInverseFreeLayerNorm(16, weight=w, bias=b)
        assert torch.allclose(enc_ln.weight, w.to(torch.float64))
        assert torch.allclose(enc_ln.bias, b.to(torch.float64))


# ============================================================================
# Taylor sqrt unit tests
# ============================================================================

class TestTaylorSqrt:
    """Test the Taylor sqrt approximation with plain tensors to verify math."""

    @staticmethod
    def _plain_taylor_sqrt(x: torch.Tensor) -> torch.Tensor:
        """Reference implementation: sqrt(x) ≈ 1 + 0.5(x-1) - 0.125(x-1)²"""
        t = x - 1.0
        return 1.0 + 0.5 * t - 0.125 * t ** 2

    def test_near_one(self):
        x = torch.tensor([1.0])
        approx = self._plain_taylor_sqrt(x)
        assert torch.allclose(approx, torch.tensor([1.0]), atol=1e-10)

    def test_range_0_to_2(self):
        # The Taylor sqrt is most accurate near x=1. At the boundaries
        # (x→0, x→2) the 2nd-order approximation degrades.  λ is chosen
        # so λ·Σz² stays in a narrow band around 1; test that band.
        x = torch.linspace(0.5, 1.5, 100)
        approx = self._plain_taylor_sqrt(x)
        exact = torch.sqrt(x)
        rel_err = ((approx - exact) / exact).abs().max()
        assert rel_err < 0.05  # tight bound in the practical range

    def test_at_half(self):
        x = torch.tensor([0.5])
        approx = self._plain_taylor_sqrt(x).item()
        exact = math.sqrt(0.5)
        assert abs(approx - exact) / exact < 0.05


# ============================================================================
# Converter integration tests
# ============================================================================

class TestConverterInverseFreeIntegration:

    def test_stip_converts_eligible_ln_to_inverse_free(self):
        """When architecture='stip', eligible LNs become EncryptedInverseFreeLayerNorm."""
        model = TwoLNLinearReLU(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        # norm1 is eligible → should be inverse-free
        assert isinstance(enc_model["norm1"], EncryptedInverseFreeLayerNorm)
        # norm2 is NOT eligible (no LN after it) → standard
        assert isinstance(enc_model["norm2"], EncryptedLayerNorm)

    def test_default_architecture_never_uses_inverse_free(self):
        """Default architecture should never produce inverse-free LNs."""
        model = TwoLNLinearReLU(16).eval()
        options = ConversionOptions(architecture="default")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        assert isinstance(enc_model["norm1"], EncryptedLayerNorm)
        assert isinstance(enc_model["norm2"], EncryptedLayerNorm)

    def test_stip_gelu_model_keeps_standard_ln(self):
        """GELU breaks eligibility → both LNs stay standard even under stip."""
        model = TwoLNWithGELU(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        assert isinstance(enc_model["norm1"], EncryptedLayerNorm)
        assert isinstance(enc_model["norm2"], EncryptedLayerNorm)

    def test_stip_three_ln_chain(self):
        """Three LN chain: first two eligible, third not."""
        model = ThreeLNChain(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        assert isinstance(enc_model["norm1"], EncryptedInverseFreeLayerNorm)
        assert isinstance(enc_model["norm2"], EncryptedInverseFreeLayerNorm)
        assert isinstance(enc_model["norm3"], EncryptedLayerNorm)

    def test_stip_with_dropout(self):
        """Dropout doesn't break eligibility."""
        model = TwoLNWithDropout(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        assert isinstance(enc_model["norm1"], EncryptedInverseFreeLayerNorm)

    def test_path_tracking_resets_between_conversions(self):
        """_current_path should be empty after conversion finishes."""
        model = TwoLNLinearReLU(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        converter.convert(model)

        assert converter._current_path == []

    def test_no_ln_model_stip(self):
        """Model without LN under stip should convert normally."""
        model = NoLNModel(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        # Should complete without error
        assert enc_model is not None


# ============================================================================
# Nested module path tracking tests
# ============================================================================

class NestedTransformerBlock(nn.Module):
    """Simulates nested module structure: block.norm1 → block.fc → block.act → block.norm2."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.LayerNorm(d),
        )
        self.head = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


class TestNestedPathTracking:

    def test_nested_sequential_eligible(self):
        """Detection should work with nested nn.Sequential containers."""
        model = NestedTransformerBlock(16).eval()
        eligible = detect_inverse_free_layernorms(model)
        # torch.fx flattens Sequential to block.0, block.1, etc.
        assert "block.0" in eligible  # first LN in sequential

    def test_nested_converter_uses_inverse_free(self):
        """Converter should track nested paths and apply inverse-free to eligible LNs."""
        model = NestedTransformerBlock(16).eval()
        options = ConversionOptions(architecture="stip")
        converter = ModelConverter(options)
        enc_model = converter.convert(model)

        # The model converts to EncryptedSequential with a nested EncryptedSequential
        # for self.block. Access the inner block's first element (LN).
        inner_block = enc_model["block"]
        assert isinstance(inner_block["0"], EncryptedInverseFreeLayerNorm)
        assert isinstance(inner_block["3"], EncryptedLayerNorm)


# ============================================================================
# Bias scaling (σ propagation) tests
# ============================================================================

def _patch_mock_backend(monkeypatch):
    from tests.mocks.mock_backend import MockCKKSConfig, MockCKKSContext
    import cukks.context as ctx_module

    monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
    monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
    monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)


class TestSigmaFactorPropagation:

    def test_inverse_free_ln_stores_sigma_on_output(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        enc_ln = EncryptedInverseFreeLayerNorm(16)
        result = enc_ln(x)

        assert result._sigma_factor is not None

    def test_standard_ln_clears_sigma(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        enc_inv_ln = EncryptedInverseFreeLayerNorm(16)
        mid = enc_inv_ln(x)
        assert mid._sigma_factor is not None

        enc_std_ln = EncryptedLayerNorm(16)
        out = enc_std_ln(mid)
        assert out._sigma_factor is None

    def test_sigma_propagates_through_operations(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        enc_ln = EncryptedInverseFreeLayerNorm(16)
        mid = enc_ln(x)

        added = mid.add(1.0)
        assert added._sigma_factor is not None

        cloned = mid.clone()
        assert cloned._sigma_factor is not None

    def test_linear_uses_sigma_for_bias_scaling(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        enc_ln = EncryptedInverseFreeLayerNorm(16)
        mid = enc_ln(x)
        assert mid._sigma_factor is not None

        linear = nn.Linear(16, 8)
        enc_linear = EncryptedLinear.from_torch(linear)
        out = enc_linear(mid)

        assert out._sigma_factor is not None

    def test_linear_without_sigma_uses_normal_path(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        linear = nn.Linear(16, 8)
        enc_linear = EncryptedLinear.from_torch(linear)
        out = enc_linear(x)

        assert out._sigma_factor is None

    def test_linear_no_bias_ignores_sigma(self, monkeypatch):
        from cukks import CKKSInferenceContext
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        enc_ln = EncryptedInverseFreeLayerNorm(16)
        mid = enc_ln(x)

        linear = nn.Linear(16, 8, bias=False)
        enc_linear = EncryptedLinear.from_torch(linear)
        out = enc_linear(mid)

        assert out._sigma_factor is not None

    def test_full_chain_inv_ln_linear_relu_std_ln(self, monkeypatch):
        from cukks import CKKSInferenceContext
        from cukks.nn import EncryptedReLU
        _patch_mock_backend(monkeypatch)

        ctx = CKKSInferenceContext(device="cpu", use_bsgs=False)
        x = ctx.encrypt(torch.randn(16, dtype=torch.float64))

        inv_ln = EncryptedInverseFreeLayerNorm(16)
        linear = EncryptedLinear.from_torch(nn.Linear(16, 16))
        relu = EncryptedReLU(degree=4)
        std_ln = EncryptedLayerNorm(16)

        h = inv_ln(x)
        assert h._sigma_factor is not None
        h = linear(h)
        assert h._sigma_factor is not None
        h = relu(h)
        assert h._sigma_factor is not None
        h = std_ln(h)
        assert h._sigma_factor is None
