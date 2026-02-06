"""Tests for auto-parameter selection and auto-bootstrapping.

Covers:
- convert() forwarding activation_degree / use_square_activation to context
- InferenceConfig.for_depth() with and without bootstrapping
- _estimate_model_depth() accuracy
- Auto-bootstrap detection for deep networks
- maybe_bootstrap() self._context fallback
- EncryptedSequential.forward() bootstrap hook
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mocks.mock_backend import MockCKKSConfig, MockCKKSContext
from ckks_torch.context import (
    InferenceConfig,
    CKKSInferenceContext,
    _estimate_model_depth,
)
from ckks_torch.converter import convert, estimate_depth
from ckks_torch.tensor import EncryptedTensor
from ckks_torch.nn import EncryptedLinear, EncryptedSquare, EncryptedSequential


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_enc_context():
    class Ctx:
        def __init__(self):
            self._ctx = MockCKKSContext()
            self.use_bsgs = False
            self._max_rotation_dim = 1024
            self._auto_bootstrap = False
            self._bootstrap_threshold = 2

        @property
        def auto_bootstrap(self):
            return self._auto_bootstrap

        @property
        def bootstrap_threshold(self):
            return self._bootstrap_threshold

        def encrypt(self, tensor):
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(self, enc_tensor, shape=None):
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)

    return Ctx()


@pytest.fixture
def auto_bootstrap_context():
    class Ctx:
        def __init__(self):
            cfg = MockCKKSConfig(enable_bootstrap=True)
            self._ctx = MockCKKSContext(cfg)
            self.use_bsgs = False
            self._max_rotation_dim = 1024
            self._auto_bootstrap = True
            self._bootstrap_threshold = 2

        @property
        def auto_bootstrap(self):
            return self._auto_bootstrap

        @property
        def bootstrap_threshold(self):
            return self._bootstrap_threshold

        def encrypt(self, tensor):
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(self, enc_tensor, shape=None):
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)

    return Ctx()


# ============================================================
# _estimate_model_depth
# ============================================================

class TestEstimateModelDepth:

    def test_linear_only(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
        assert _estimate_model_depth(model) == 2

    def test_square_activation(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        depth = _estimate_model_depth(model, use_square_activation=True)
        # 2 linear (1 each) + 1 activation (1 for x²) = 3
        assert depth == 3

    def test_poly_activation_degree4(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        depth = _estimate_model_depth(model, activation_degree=4)
        # ceil(log2(5)) = 3 per activation
        assert depth == 2 + 3

    def test_poly_activation_degree8(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 4))
        depth = _estimate_model_depth(model, activation_degree=8)
        # ceil(log2(9)) = 4
        assert depth == 2 + 4

    def test_multiple_activations(self):
        model = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )
        depth = _estimate_model_depth(model, activation_degree=4)
        # 3 linear + 2 * ceil(log2(5)) = 3 + 6 = 9
        assert depth == 9

    def test_empty_model(self):
        model = nn.Sequential()
        assert _estimate_model_depth(model) >= 1

    def test_conv_counted(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        depth = _estimate_model_depth(model, activation_degree=4)
        # 1 conv + ceil(log2(5)) relu + 1 linear = 1 + 3 + 1 = 5
        assert depth == 5


# ============================================================
# InferenceConfig.for_depth
# ============================================================

class TestInferenceConfigForDepth:

    def test_small_depth_gets_16384(self):
        cfg = InferenceConfig.for_depth(3)
        # effective = 3 + 2 = 5 ≤ 6
        assert cfg.poly_mod_degree == 16384

    def test_medium_depth_gets_32768(self):
        cfg = InferenceConfig.for_depth(8)
        # effective = 10 ≤ 16
        assert cfg.poly_mod_degree == 32768

    def test_large_depth_gets_65536(self):
        cfg = InferenceConfig.for_depth(20)
        # effective = 22 > 16
        assert cfg.poly_mod_degree == 65536

    def test_default_scale_bits_50(self):
        cfg = InferenceConfig.for_depth(4)
        assert cfg.scale_bits == 50

    def test_default_security_128_classic(self):
        cfg = InferenceConfig.for_depth(4)
        assert cfg.security_level == "128_classic"

    def test_override_scale_bits(self):
        cfg = InferenceConfig.for_depth(4, scale_bits=40)
        assert cfg.scale_bits == 40

    def test_bootstrap_enabled_uses_65536(self):
        cfg = InferenceConfig.for_depth(5, enable_bootstrap=True)
        assert cfg.poly_mod_degree == 65536
        assert cfg.enable_bootstrap is True

    def test_bootstrap_override_poly_mod(self):
        cfg = InferenceConfig.for_depth(5, enable_bootstrap=True, poly_mod_degree=32768)
        assert cfg.poly_mod_degree == 32768

    def test_bootstrap_level_budget_forwarded(self):
        cfg = InferenceConfig.for_depth(5, enable_bootstrap=True, level_budget=(4, 4))
        assert cfg.level_budget == (4, 4)
        assert cfg.resolved_level_budget == (4, 4)

    def test_bootstrap_default_level_budget(self):
        cfg = InferenceConfig.for_depth(5, enable_bootstrap=True)
        assert cfg.resolved_level_budget == (3, 3)

    def test_no_bootstrap_no_level_budget(self):
        cfg = InferenceConfig.for_depth(5)
        assert cfg.enable_bootstrap is False
        assert cfg.resolved_level_budget is None


# ============================================================
# InferenceConfig.for_model
# ============================================================

class TestInferenceConfigForModel:

    def test_forwards_activation_degree(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        cfg_deg4 = InferenceConfig.for_model(model, activation_degree=4)
        cfg_deg8 = InferenceConfig.for_model(model, activation_degree=8)
        # degree=4 → depth=5, degree=8 → depth=6, both +2 margin
        assert cfg_deg8.mult_depth > cfg_deg4.mult_depth

    def test_forwards_square_activation(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        cfg_sq = InferenceConfig.for_model(model, use_square_activation=True)
        cfg_poly = InferenceConfig.for_model(model, activation_degree=4)
        # square: 3+2=5, poly: 5+2=7
        assert cfg_sq.mult_depth < cfg_poly.mult_depth

    def test_forwards_enable_bootstrap(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        cfg = InferenceConfig.for_model(model, enable_bootstrap=True)
        assert cfg.enable_bootstrap is True
        assert cfg.poly_mod_degree == 65536


# ============================================================
# convert() parameter forwarding
# ============================================================

class TestConvertParameterForwarding:

    def test_activation_degree_affects_config(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        _, ctx4 = convert(model, activation_degree=4)
        _, ctx8 = convert(model, activation_degree=8)
        assert ctx8.config.mult_depth > ctx4.config.mult_depth

    def test_square_activation_reduces_depth(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        _, ctx_sq = convert(model, use_square_activation=True)
        _, ctx_poly = convert(model, use_square_activation=False, activation_degree=4)
        assert ctx_sq.config.mult_depth < ctx_poly.config.mult_depth

    def test_explicit_enable_bootstrap(self):
        model = nn.Linear(16, 4)
        _, ctx = convert(model, enable_bootstrap=True)
        assert ctx.config.enable_bootstrap is True
        assert ctx._auto_bootstrap is True

    def test_explicit_disable_bootstrap(self):
        model = nn.Linear(16, 4)
        _, ctx = convert(model, enable_bootstrap=False)
        assert ctx.config.enable_bootstrap is False

    def test_auto_bootstrap_follows_enable(self):
        model = nn.Linear(16, 4)
        _, ctx = convert(model, enable_bootstrap=True, auto_bootstrap=False)
        assert ctx._auto_bootstrap is False
        assert ctx.config.enable_bootstrap is True

    def test_bootstrap_threshold_forwarded(self):
        model = nn.Linear(16, 4)
        _, ctx = convert(model, enable_bootstrap=True, bootstrap_threshold=5)
        assert ctx._bootstrap_threshold == 5


# ============================================================
# Auto-bootstrap detection for deep networks
# ============================================================

class TestAutoBootstrapDetection:

    def _deep_model(self, num_layers=8):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(16, 16))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 4))
        return nn.Sequential(*layers).eval()

    def test_shallow_model_no_bootstrap(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
        _, ctx = convert(model, activation_degree=4)
        assert ctx.config.enable_bootstrap is False

    def test_deep_model_auto_enables_bootstrap(self):
        # 8 linear + 8 relu(degree=4, ceil(log2(5))=3) = 8+24 = 32 > 14
        model = self._deep_model(num_layers=8)
        _, ctx = convert(model, activation_degree=4)
        assert ctx.config.enable_bootstrap is True
        assert ctx._auto_bootstrap is True

    def test_deep_model_with_square_may_fit(self):
        # 5 linear + 4 square = 5+4 = 9 ≤ 14 → no bootstrap
        model = self._deep_model(num_layers=4)
        _, ctx = convert(model, use_square_activation=True)
        assert ctx.config.enable_bootstrap is False

    def test_very_deep_square_needs_bootstrap(self):
        # 16 linear + 15 square = 31 > 14
        model = self._deep_model(num_layers=15)
        _, ctx = convert(model, use_square_activation=True)
        assert ctx.config.enable_bootstrap is True


# ============================================================
# maybe_bootstrap() context fallback
# ============================================================

class TestMaybeBootstrapFallback:

    def test_uses_self_context(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))
        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        # No explicit context → uses self._context
        y = x.maybe_bootstrap()
        assert y.depth == 0

    def test_explicit_context_overrides(self, auto_bootstrap_context, mock_enc_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))
        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        # mock_enc_context has auto_bootstrap=False → no bootstrap
        y = x.maybe_bootstrap(mock_enc_context)
        assert y.depth == 2

    def test_none_context_none_self_noop(self):
        cfg = MockCKKSConfig()
        mock_ctx = MockCKKSContext(cfg)
        cipher = mock_ctx.encrypt(torch.tensor([1.0]))
        t = EncryptedTensor(cipher, (1,), None, depth=5)
        # _context is None, no explicit context → should not crash
        y = t.maybe_bootstrap()
        assert y.depth == 5

    def test_below_threshold_noop(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))
        x = x.mul(2.0)
        assert x.depth == 1
        y = x.maybe_bootstrap()
        assert y.depth == 1


# ============================================================
# EncryptedSequential.forward() bootstrap hook
# ============================================================

class TestSequentialBootstrapHook:

    def test_forward_calls_maybe_bootstrap(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        w1 = torch.randn(4, 4, dtype=torch.float64)
        b1 = torch.randn(4, dtype=torch.float64)
        w2 = torch.randn(4, 4, dtype=torch.float64)
        b2 = torch.randn(4, dtype=torch.float64)

        seq = EncryptedSequential(
            EncryptedLinear(4, 4, w1, b1),
            EncryptedSquare(),
            EncryptedLinear(4, 4, w2, b2),
        )

        x = ctx.encrypt(torch.randn(4))
        out = seq(x)
        # If auto_bootstrap triggers between layers, depth resets
        # Linear(depth+1) → bootstrap? → Square(depth+1) → bootstrap? → Linear(depth+1)
        # With threshold=2: after Linear depth=1 (no bootstrap), after Square depth=2 (bootstrap→0),
        # before next Linear: maybe_bootstrap checks depth=0 (no), Linear→depth=1
        # The key point: it doesn't crash and produces a valid output
        result = ctx.decrypt(out)
        assert result.shape == (4,)

    def test_forward_without_bootstrap_still_works(self, mock_enc_context):
        ctx = mock_enc_context
        w1 = torch.randn(4, 4, dtype=torch.float64)
        b1 = torch.randn(4, dtype=torch.float64)
        w2 = torch.randn(4, 4, dtype=torch.float64)
        b2 = torch.randn(4, dtype=torch.float64)

        seq = EncryptedSequential(
            EncryptedLinear(4, 4, w1, b1),
            EncryptedSquare(),
            EncryptedLinear(4, 4, w2, b2),
        )

        x = ctx.encrypt(torch.randn(4))
        out = seq(x)
        result = ctx.decrypt(out)
        assert result.shape == (4,)

    def test_sequential_bootstrap_resets_depth(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context  # threshold=2
        w = torch.eye(4, dtype=torch.float64)
        b = torch.zeros(4, dtype=torch.float64)

        seq = EncryptedSequential(
            EncryptedLinear(4, 4, w, b),  # depth 0→1
            EncryptedSquare(),             # depth 1→2
            # Before next layer: maybe_bootstrap sees depth=2 ≥ threshold=2 → bootstrap → depth=0
            EncryptedLinear(4, 4, w, b),  # depth 0→1
            EncryptedSquare(),             # depth 1→2
            # Before next layer: bootstrap → depth=0
            EncryptedLinear(4, 4, w, b),  # depth 0→1
        )

        x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        out = seq(x)
        # Without bootstrap this would be depth=5. With bootstrap it resets.
        assert out.depth < 5
        result = ctx.decrypt(out)
        assert result.shape == (4,)


# ============================================================
# estimate_depth (public API) matches _estimate_model_depth
# ============================================================

class TestEstimateDepthPublicAPI:

    def test_matches_internal(self):
        model = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )
        assert estimate_depth(model, activation_degree=4) == _estimate_model_depth(
            model, activation_degree=4
        )

    def test_default_degree_is_4(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        assert estimate_depth(model) == estimate_depth(model, activation_degree=4)
