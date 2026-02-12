"""Tests for bootstrap wrapper functionality."""

import pytest
import torch


class TestManualBootstrap:

    def test_manual_bootstrap(self, bootstrap_mock_context):
        ctx = bootstrap_mock_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))

        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        x_refreshed = x.bootstrap()
        assert x_refreshed.depth == 0

        decrypted = ctx.decrypt(x_refreshed)
        torch.testing.assert_close(decrypted, torch.tensor([4.0, 8.0, 12.0]))


class TestAutoBootstrap:

    def test_auto_bootstrap_enabled(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))

        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        x_maybe = x.maybe_bootstrap(ctx)
        assert x_maybe.depth == 0

    def test_auto_bootstrap_disabled(self, bootstrap_mock_context):
        ctx = bootstrap_mock_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0, 3.0]))

        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        x_maybe = x.maybe_bootstrap(ctx)
        assert x_maybe.depth == 2


class TestBootstrapThreshold:

    def test_bootstrap_threshold_not_reached(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))

        x = x.mul(2.0)
        assert x.depth == 1

        x_maybe = x.maybe_bootstrap(ctx)
        assert x_maybe.depth == 1

    def test_bootstrap_threshold_reached(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))

        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 2

        x_maybe = x.maybe_bootstrap(ctx)
        assert x_maybe.depth == 0

    def test_bootstrap_threshold_exceeded(self, auto_bootstrap_context):
        ctx = auto_bootstrap_context
        x = ctx.encrypt(torch.tensor([1.0, 2.0]))

        x = x.mul(2.0)
        x = x.mul(2.0)
        x = x.mul(2.0)
        assert x.depth == 3

        x_maybe = x.maybe_bootstrap(ctx)
        assert x_maybe.depth == 0


@pytest.fixture
def bootstrap_mock_context():
    from mocks.mock_backend import MockCKKSConfig, MockCKKSContext
    from cukks.tensor import EncryptedTensor

    class BootstrapMockContext:
        def __init__(self):
            config = MockCKKSConfig(enable_bootstrap=True)
            self._ctx = MockCKKSContext(config)
            self._auto_bootstrap = False
            self._bootstrap_threshold = 2

        @property
        def auto_bootstrap(self) -> bool:
            return self._auto_bootstrap

        @property
        def bootstrap_threshold(self) -> int:
            return self._bootstrap_threshold

        def encrypt(self, tensor: torch.Tensor) -> EncryptedTensor:
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(self, enc_tensor: EncryptedTensor, shape=None) -> torch.Tensor:
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)

    return BootstrapMockContext()


@pytest.fixture
def auto_bootstrap_context():
    from mocks.mock_backend import MockCKKSConfig, MockCKKSContext
    from cukks.tensor import EncryptedTensor

    class AutoBootstrapContext:
        def __init__(self):
            config = MockCKKSConfig(enable_bootstrap=True)
            self._ctx = MockCKKSContext(config)
            self._auto_bootstrap = True
            self._bootstrap_threshold = 2

        @property
        def auto_bootstrap(self) -> bool:
            return self._auto_bootstrap

        @property
        def bootstrap_threshold(self) -> int:
            return self._bootstrap_threshold

        def encrypt(self, tensor: torch.Tensor) -> EncryptedTensor:
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)

        def decrypt(self, enc_tensor: EncryptedTensor, shape=None) -> torch.Tensor:
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)

    return AutoBootstrapContext()
