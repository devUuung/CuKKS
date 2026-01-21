import pytest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mocks.mock_backend import MockCKKSConfig, MockCKKSContext, MockCKKSTensor


@pytest.fixture
def mock_config():
    return MockCKKSConfig()


@pytest.fixture
def mock_context():
    return MockCKKSContext()


@pytest.fixture
def mock_enc_context():
    from ckks_torch.tensor import EncryptedTensor
    
    class EncryptedMockContext:
        def __init__(self):
            self._ctx = MockCKKSContext()
        
        def encrypt(self, tensor):
            cipher = self._ctx.encrypt(tensor)
            return EncryptedTensor(cipher, tuple(tensor.shape), self)
        
        def decrypt(self, enc_tensor, shape=None):
            target_shape = shape if shape else enc_tensor.shape
            return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)
    
    return EncryptedMockContext()


@pytest.fixture
def use_mock_backend(monkeypatch):
    from mocks import mock_backend
    
    monkeypatch.setattr("ckks_torch.context.CKKSConfig", MockCKKSConfig, raising=False)
    monkeypatch.setattr("ckks_torch.context.CKKSContext", MockCKKSContext, raising=False)
    monkeypatch.setattr("ckks_torch.tensor.EncryptedTensor", MockCKKSTensor, raising=False)


# =============================================================================
# Real GPU Backend Fixtures
# =============================================================================

def _has_real_backend():
    """Check if real CKKS backend is available."""
    try:
        from ckks import CKKSConfig, CKKSContext
        return True
    except ImportError:
        return False


def _has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# Skip markers
requires_real_backend = pytest.mark.skipif(
    not _has_real_backend(),
    reason="Real CKKS backend not available"
)

requires_cuda = pytest.mark.skipif(
    not _has_cuda(),
    reason="CUDA not available"
)

requires_gpu = pytest.mark.skipif(
    not (_has_real_backend() and _has_cuda()),
    reason="GPU backend requires both real CKKS backend and CUDA"
)


@pytest.fixture
def real_context():
    """Real CKKS context on CPU."""
    if not _has_real_backend():
        pytest.skip("Real CKKS backend not available")
    
    from ckks_torch import CKKSInferenceContext, InferenceConfig
    
    config = InferenceConfig(
        poly_mod_degree=32768,
        scale_bits=40,
        mult_depth=6,
    )
    
    ctx = CKKSInferenceContext(
        config=config,
        device="cpu",
        max_rotation_dim=2048,
        use_bsgs=True,
    )
    
    return ctx


@pytest.fixture
def gpu_context():
    """Real CKKS context on GPU (CUDA)."""
    if not _has_real_backend():
        pytest.skip("Real CKKS backend not available")
    if not _has_cuda():
        pytest.skip("CUDA not available")
    
    from ckks_torch import CKKSInferenceContext, InferenceConfig
    
    config = InferenceConfig(
        poly_mod_degree=32768,
        scale_bits=50,
        mult_depth=8,
    )
    
    ctx = CKKSInferenceContext(
        config=config,
        device="cuda",
        max_rotation_dim=2048,
        use_bsgs=True,
    )
    
    return ctx
