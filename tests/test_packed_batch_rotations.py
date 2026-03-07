"""Tests for packed-batch rotation key planning (Task 2).

Validates:
- compute_packed_batch_rotations returns O(sqrt(K) + B) rotations
- batch_size parameter flows through context and converter
- encrypt_batch validates sample count against batch_size
- batch_size=1 preserves existing single-sample behaviour
"""

import math
import pytest
import torch


# ---------------------------------------------------------------------------
# Unit tests for compute_packed_batch_rotations
# ---------------------------------------------------------------------------

class TestComputePackedBatchRotations:

    def test_batch_size_1_matches_bsgs(self):
        """batch_size=1 should return same as compute_bsgs_rotations."""
        from cukks.context import compute_bsgs_rotations, compute_packed_batch_rotations

        for K in [64, 128, 256, 1024]:
            bsgs = compute_bsgs_rotations(K)
            packed = compute_packed_batch_rotations(K, batch_size=1)
            assert packed == bsgs, f"Mismatch for K={K}"

    def test_includes_cross_block_offsets(self):
        """batch_size>1 should add {i*K for i in 1..B-1}."""
        from cukks.context import compute_packed_batch_rotations

        K, B = 128, 4
        rotations = compute_packed_batch_rotations(K, batch_size=B)

        for i in range(1, B):
            assert i * K in rotations, f"Missing cross-block offset {i * K}"

    def test_no_cross_block_for_batch_1(self):
        """batch_size=1 should not include any offsets >= K."""
        from cukks.context import compute_packed_batch_rotations

        K = 128
        rotations = compute_packed_batch_rotations(K, batch_size=1)
        assert all(r < K for r in rotations), "Unexpected offset >= K for batch_size=1"

    def test_growth_is_sqrt_k_plus_b(self):
        """Rotation count should grow as O(sqrt(K) + B), NOT O(B*K)."""
        from cukks.context import compute_packed_batch_rotations

        K = 1024
        B = 16
        rotations = compute_packed_batch_rotations(K, batch_size=B)

        # BSGS gives ~2*sqrt(K) rotations, cross-block adds B-1
        sqrt_k = int(math.ceil(math.sqrt(K)))
        upper_bound = 2 * sqrt_k + B  # generous upper bound

        assert len(rotations) <= upper_bound, (
            f"Too many rotations: {len(rotations)} > {upper_bound}. "
            f"Expected O(sqrt({K}) + {B}) = O({sqrt_k} + {B})"
        )

        # Should be much less than naive O(B*K)
        naive_count = B * K
        assert len(rotations) < naive_count // 10, (
            f"Rotation count {len(rotations)} too close to naive O(B*K)={naive_count}"
        )

    def test_returns_sorted_positive(self):
        """Returned list should be sorted and contain only positive values."""
        from cukks.context import compute_packed_batch_rotations

        rotations = compute_packed_batch_rotations(256, batch_size=8)
        assert rotations == sorted(rotations)
        assert all(r > 0 for r in rotations)

    def test_small_dim(self):
        """Edge cases for very small max_dim values."""
        from cukks.context import compute_packed_batch_rotations

        # max_dim=1, batch_size=1 -> no rotations needed
        assert compute_packed_batch_rotations(1, batch_size=1) == []
        # max_dim=1, batch_size=4 -> only cross-block offsets {1, 2, 3}
        assert compute_packed_batch_rotations(1, batch_size=4) == [1, 2, 3]
        # max_dim=0 -> degenerate, no BSGS rotations
        assert compute_packed_batch_rotations(0, batch_size=1) == []


# ---------------------------------------------------------------------------
# Integration: batch_size parameter flows correctly
# ---------------------------------------------------------------------------

class TestBatchSizeContextCreation:

    @pytest.fixture(autouse=True)
    def _setup_mock_backend(self, monkeypatch):
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

    def test_default_batch_size_is_1(self):
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu")
        assert ctx.batch_size == 1

    def test_batch_size_stored(self):
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu", batch_size=8)
        assert ctx.batch_size == 8
        assert ctx._batch_size == 8

    def test_batch_size_rotations_include_cross_block(self):
        """Context with batch_size>1 should have cross-block offsets in rotations."""
        from cukks import CKKSInferenceContext

        K = 128
        B = 4
        ctx = CKKSInferenceContext(
            device="cpu",
            max_rotation_dim=K,
            batch_size=B,
        )

        # Cross-block offsets and their negatives should be present
        for i in range(1, B):
            assert i * K in ctx._rotations, f"Missing positive cross-block offset {i * K}"
            assert -(i * K) in ctx._rotations, f"Missing negative cross-block offset {-(i * K)}"

    def test_batch_size_1_no_cross_block(self):
        """Default batch_size=1 should NOT have offsets >= max_rotation_dim."""
        from cukks import CKKSInferenceContext

        K = 128
        ctx = CKKSInferenceContext(
            device="cpu",
            max_rotation_dim=K,
            batch_size=1,
        )

        # No rotation should be >= K (absolute value) since BSGS only goes up to K-1
        for r in ctx._rotations:
            assert abs(r) < K, f"Unexpected rotation {r} >= K={K} for batch_size=1"

    def test_for_model_passes_batch_size(self):
        """CKKSInferenceContext.for_model should forward batch_size."""
        import torch.nn as nn
        from cukks import CKKSInferenceContext

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        ctx = CKKSInferenceContext.for_model(model, batch_size=4, enable_gpu=False)

        assert ctx.batch_size == 4

        # Should have cross-block offsets for the max_dim
        max_dim = ctx._max_rotation_dim
        assert max_dim is not None
        for i in range(1, 4):
            assert i * max_dim in ctx._rotations


# ---------------------------------------------------------------------------
# encrypt_batch validation
# ---------------------------------------------------------------------------

class TestEncryptBatchValidation:

    @pytest.fixture(autouse=True)
    def _setup_mock_backend(self, monkeypatch):
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

    def test_batch_size_mismatch_raises(self):
        """encrypt_batch with wrong sample count should raise ValueError."""
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu", batch_size=4)
        samples = [torch.randn(10) for _ in range(3)]  # 3 != 4

        with pytest.raises(ValueError, match="batch_size=4.*3 samples"):
            ctx.encrypt_batch(samples)

    def test_batch_size_match_works(self):
        """encrypt_batch with correct sample count should succeed."""
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu", batch_size=4)
        samples = [torch.randn(10) for _ in range(4)]

        enc = ctx.encrypt_batch(samples)
        assert enc._packed_batch is True
        assert enc._batch_size == 4

    def test_batch_size_1_no_constraint(self):
        """Default batch_size=1 should allow any sample count."""
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu")  # batch_size=1

        for n in [1, 2, 5, 16]:
            samples = [torch.randn(10) for _ in range(n)]
            enc = ctx.encrypt_batch(samples)
            assert enc._batch_size == n


# ---------------------------------------------------------------------------
# Converter batch_size plumbing
# ---------------------------------------------------------------------------

class TestConverterBatchSize:

    def test_conversion_options_batch_size(self):
        from cukks.converter import ConversionOptions

        opts = ConversionOptions(batch_size=8)
        assert opts.batch_size == 8

    def test_conversion_options_default(self):
        from cukks.converter import ConversionOptions

        opts = ConversionOptions()
        assert opts.batch_size == 1

    def test_convert_passes_batch_size(self, monkeypatch):
        """cukks.convert(model, batch_size=N) should create ctx with that batch_size."""
        import torch.nn as nn
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

        from cukks.converter import convert
        enc_model, ctx = convert(model, batch_size=4, enable_gpu=False)

        assert ctx.batch_size == 4


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestBatchSizeSerialization:

    @pytest.fixture(autouse=True)
    def _setup_mock_backend(self, monkeypatch):
        from tests.mocks.mock_backend import MockCKKSContext, MockCKKSConfig
        import cukks.context as ctx_module

        monkeypatch.setattr(ctx_module, "CKKSConfig", MockCKKSConfig, raising=False)
        monkeypatch.setattr(ctx_module, "CKKSContext", MockCKKSContext, raising=False)
        monkeypatch.setattr(ctx_module, "_BACKEND_AVAILABLE", True, raising=False)

    def test_save_load_preserves_batch_size(self, tmp_path):
        from cukks import CKKSInferenceContext

        ctx = CKKSInferenceContext(device="cpu", batch_size=8)
        path = tmp_path / "ctx.pkl"
        ctx.save_context(path)

        loaded = CKKSInferenceContext.load_context(path)
        assert loaded.batch_size == 8

    def test_load_legacy_defaults_to_1(self, tmp_path):
        """Loading a context saved before batch_size was added should default to 1."""
        import pickle
        from cukks import CKKSInferenceContext, InferenceConfig

        # Simulate legacy save (no batch_size key)
        config = InferenceConfig()
        legacy_data = {
            "poly_mod_degree": config.poly_mod_degree,
            "scale_bits": config.scale_bits,
            "security_level": config.security_level,
            "mult_depth": config.mult_depth,
            "enable_bootstrap": config.enable_bootstrap,
            "device": "cpu",
            "use_bsgs": True,
            "rotations": [1, -1],
            "max_rotation_dim": 1024,
            "auto_bootstrap": False,
            "bootstrap_threshold": 2,
            # NOTE: no "batch_size" key
        }
        path = tmp_path / "legacy_ctx.pkl"
        with open(path, "wb") as f:
            pickle.dump(legacy_data, f)

        loaded = CKKSInferenceContext.load_context(path)
        assert loaded.batch_size == 1
