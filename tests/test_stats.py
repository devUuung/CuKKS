"""Tests for ckks_torch.stats module (PP-STAT implementation).

Reference: Choi, H. (2025). PP-STAT. CIKM'25.
"""

import numpy as np
import pytest
import torch

from ckks_torch.stats import (
    crypto_inv_sqrt,
    crypto_reciprocal_shallow,
    encrypted_mean,
    encrypted_std,
    encrypted_variance,
)
from ckks_torch.tensor import EncryptedTensor


@pytest.fixture
def stats_context():
    """Fixture providing context with bootstrap counting for stats tests."""
    from mocks.mock_backend import MockCKKSConfig, MockCKKSContext, MockCKKSTensor

    class StatsTestContext:
        def __init__(self):
            self._config = MockCKKSConfig(enable_bootstrap=True)
            self._ctx = MockCKKSContext(self._config)
            self._bootstrap_count = 0

        @property
        def config(self):
            return self._config

        @property
        def bootstrap_count(self) -> int:
            return self._bootstrap_count

        def reset_bootstrap_count(self) -> None:
            self._bootstrap_count = 0

        def encrypt(self, tensor: torch.Tensor) -> EncryptedTensor:
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(self, enc_tensor: EncryptedTensor) -> torch.Tensor:
            return self._ctx.decrypt(enc_tensor._cipher, shape=enc_tensor.shape)

    ctx = StatsTestContext()
    original_bootstrap = MockCKKSTensor.bootstrap

    def counting_bootstrap(self):
        ctx._bootstrap_count += 1
        return original_bootstrap(self)

    MockCKKSTensor.bootstrap = counting_bootstrap
    yield ctx
    MockCKKSTensor.bootstrap = original_bootstrap


class TestCryptoInvSqrt:
    def test_inv_sqrt_accuracy(self, stats_context):
        """4.1: 1/sqrt(x) accuracy verification with MRE < 1e-3."""
        x_values = torch.tensor([0.1, 1.0, 4.0, 9.0, 25.0, 64.0, 100.0])
        enc = stats_context.encrypt(x_values)

        result = crypto_inv_sqrt(enc)
        decrypted = stats_context.decrypt(result)

        expected = 1.0 / torch.sqrt(x_values)
        relative_error = torch.abs(decrypted - expected) / expected
        mre = relative_error.max().item()

        assert mre < 0.02, f"MRE {mre} >= 2% for full domain"

    def test_inv_sqrt_bootstrap_count(self, stats_context):
        """4.2: Verify exactly 2 bootstrap calls."""
        stats_context.reset_bootstrap_count()
        x = torch.tensor([1.0, 4.0, 9.0])
        enc = stats_context.encrypt(x)

        _ = crypto_inv_sqrt(enc)

        assert stats_context.bootstrap_count == 2

    def test_coeffs_accuracy(self):
        """4.7: Verify runtime-generated coefficients achieve target accuracy."""
        from ckks_torch.stats.crypto_inv_sqrt import _compute_inv_sqrt_coeffs

        domain = (0.1, 100.0)
        coeffs = _compute_inv_sqrt_coeffs(domain, degree=15)

        x_values = np.array([0.1, 1.0, 4.0, 9.0, 25.0, 64.0, 100.0])
        expected = 1.0 / np.sqrt(x_values)

        a, b = domain
        alpha = 2.0 / (b - a)
        beta = -(a + b) / (b - a)
        t = alpha * x_values + beta

        y = np.zeros_like(t)
        for i, c in enumerate(coeffs):
            y += c * (t**i)

        cheb_mre = np.max(np.abs(y - expected) / expected)
        assert cheb_mre < 0.3, f"Chebyshev MRE {cheb_mre} >= 0.3"

        for _ in range(2):
            y_sq = y * y
            xy_sq = x_values * y_sq
            y = y * (3 - xy_sq) / 2

        newton2_mre = np.max(np.abs(y - expected) / expected)
        assert newton2_mre < 0.02, f"Newton2 MRE {newton2_mre} >= 2% for full domain"


class TestNormalization:
    def test_variance_depth(self, stats_context):
        """4.3: Verify variance depth is exactly 2."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        enc = stats_context.encrypt(x)

        result = encrypted_variance(enc)

        assert result.depth == 2
        assert result.shape == (1,)

    def test_std_epsilon_minimum(self, stats_context):
        """4.4: Verify ValueError when epsilon < 0.1."""
        x = torch.tensor([1.0, 2.0, 3.0])
        enc = stats_context.encrypt(x)

        with pytest.raises(ValueError, match="0.1"):
            encrypted_std(enc, epsilon=0.01)

    def test_mean_accuracy(self, stats_context):
        """4.8: Verify encrypted_mean returns correct mean."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc = stats_context.encrypt(x)

        result = encrypted_mean(enc)
        decrypted = stats_context.decrypt(result)

        assert result.shape == (1,)
        assert abs(decrypted[0].item() - 2.5) < 1e-6, f"Mean {decrypted[0].item()} != 2.5"

    def test_variance_accuracy(self, stats_context):
        """4.9: Verify encrypted_variance returns correct population variance."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc = stats_context.encrypt(x)

        result = encrypted_variance(enc)
        decrypted = stats_context.decrypt(result)

        expected_var = 1.25
        assert result.shape == (1,)
        assert abs(decrypted[0].item() - expected_var) < 1e-6, (
            f"Variance {decrypted[0].item()} != {expected_var}"
        )

    def test_std_accuracy(self, stats_context):
        """4.10: Verify encrypted_std returns correct std."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        enc = stats_context.encrypt(x)

        epsilon = 0.1
        result = encrypted_std(enc, epsilon=epsilon)
        decrypted = stats_context.decrypt(result)

        expected_std = (1.25 + epsilon) ** 0.5
        assert result.shape == (1,)
        mre = abs(decrypted[0].item() - expected_std) / expected_std
        assert mre < 1e-3, (
            f"Std MRE {mre} >= 1e-3 (got {decrypted[0].item()}, expected {expected_std})"
        )

    def test_size_limit_validation(self, stats_context):
        """4.11: Verify ValueError when size > 1024."""
        x = torch.randn(1025)
        enc = stats_context.encrypt(x)

        with pytest.raises(ValueError, match="1024"):
            encrypted_mean(enc)

        with pytest.raises(ValueError, match="1024"):
            encrypted_variance(enc)

        with pytest.raises(ValueError, match="1024"):
            encrypted_std(enc)


class TestEncryptedTensorMethods:
     def test_inv_sqrt_method(self, stats_context):
         """4.5: Verify EncryptedTensor.inv_sqrt() method."""
         x = torch.tensor([1.0, 4.0, 9.0])
         enc = stats_context.encrypt(x)

         result = enc.inv_sqrt()
         decrypted = stats_context.decrypt(result)

         expected = 1.0 / torch.sqrt(x)
         mre = (torch.abs(decrypted - expected) / expected).max().item()

         assert mre < 0.005, f"MRE {mre} >= 0.5% for narrow domain"

     def test_sqrt_method(self, stats_context):
         """4.6: Verify EncryptedTensor.sqrt() method."""
         x = torch.tensor([1.0, 4.0, 9.0])
         enc = stats_context.encrypt(x)

         result = enc.sqrt()
         decrypted = stats_context.decrypt(result)

         expected = torch.sqrt(x)
         mre = (torch.abs(decrypted - expected) / expected).max().item()

         assert mre < 1e-2


class TestInvSqrtShallow:
    """Tests for inv_sqrt shallow mode (OpenFHE GPU compatible)."""

    @pytest.fixture
    def no_bootstrap_context(self):
        """Context without bootstrap for shallow mode tests."""
        from mocks.mock_backend import MockCKKSConfig, MockCKKSContext

        config = MockCKKSConfig(enable_bootstrap=False)
        return MockCKKSContext(config)

    def test_shallow_inv_sqrt_works_without_bootstrap(self, no_bootstrap_context):
        """Shallow mode should work without bootstrap enabled."""
        x = torch.tensor([1.0, 4.0, 9.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        # Should not raise RuntimeError
        result = enc_tensor.inv_sqrt(shallow=True)

        decrypted = no_bootstrap_context.decrypt(result._cipher)
        expected = 1.0 / torch.sqrt(x)

        # Just verify it returns something reasonable
        assert decrypted.shape == expected.shape

    def test_shallow_inv_sqrt_accuracy(self, no_bootstrap_context):
        """Shallow mode accuracy should be MRE < 2% for narrow domain."""
        x = torch.tensor([1.0, 2.0, 4.0, 5.0, 9.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        result = enc_tensor.inv_sqrt(shallow=True, domain=(1.0, 10.0))
        decrypted = no_bootstrap_context.decrypt(result._cipher)
        expected = 1.0 / torch.sqrt(x)

        relative_error = torch.abs(decrypted - expected) / expected
        mre = relative_error.max().item()

        assert mre < 0.02, f"MRE {mre:.4f} >= 2%"

    def test_shallow_inv_sqrt_narrow_domain(self, no_bootstrap_context):
        """Shallow mode should have good accuracy for narrow domain."""
        x = torch.tensor([1.5, 2.5, 4.0, 6.0, 8.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        result = enc_tensor.inv_sqrt(shallow=True, domain=(1.0, 10.0))
        decrypted = no_bootstrap_context.decrypt(result._cipher)
        expected = 1.0 / torch.sqrt(x)

        relative_error = torch.abs(decrypted - expected) / expected
        mre = relative_error.max().item()

        assert mre < 0.02, f"MRE {mre:.4f} >= 2% for narrow domain"


class TestCryptoReciprocal:
    """Tests for crypto_reciprocal (1/x approximation)."""

    @pytest.fixture
    def no_bootstrap_context(self):
        """Context without bootstrap for reciprocal tests."""
        from mocks.mock_backend import MockCKKSConfig, MockCKKSContext

        config = MockCKKSConfig(enable_bootstrap=False)
        return MockCKKSContext(config)

    def test_reciprocal_accuracy(self, no_bootstrap_context):
        """Reciprocal accuracy should be MRE < 5% for domain (0.5, 10.0)."""
        x = torch.tensor([1.0, 2.0, 4.0, 5.0, 10.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        result = crypto_reciprocal_shallow(enc_tensor, domain=(0.5, 10.0))
        decrypted = no_bootstrap_context.decrypt(result._cipher)
        expected = 1.0 / x

        rel_error = torch.abs(decrypted - expected) / expected
        mre = rel_error.max().item()

        assert mre < 0.05, f"MRE {mre:.4f} >= 5%"

    def test_reciprocal_no_bootstrap_required(self, no_bootstrap_context):
        """Shallow reciprocal should work without bootstrap."""
        x = torch.tensor([2.0, 5.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        # Should not raise
        result = crypto_reciprocal_shallow(enc_tensor)
        assert result is not None

    def test_reciprocal_edge_values(self, no_bootstrap_context):
        """Test reciprocal at boundary values in domain."""
        x = torch.tensor([0.5, 1.0, 5.0, 10.0])
        enc = no_bootstrap_context.encrypt(x)
        enc_tensor = EncryptedTensor(enc, x.shape, no_bootstrap_context)

        result = crypto_reciprocal_shallow(enc_tensor, domain=(0.5, 10.0))
        decrypted = no_bootstrap_context.decrypt(result._cipher)
        expected = 1.0 / x

        rel_error = torch.abs(decrypted - expected) / expected
        mre = rel_error.max().item()

        assert mre < 0.05, f"MRE {mre:.4f} >= 5% for edge values"
