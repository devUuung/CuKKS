"""PP-STAT based statistical functions.

Reference: Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
Statistical Analysis Framework. CIKM'25.

Backend Compatibility:
    - Mock backend: FULLY TESTED - all functions work correctly
    - OpenFHE backend (non-bootstrap): COMPATIBLE - use `crypto_inv_sqrt_shallow`
    - OpenFHE backend (bootstrap): BLOCKED - OpenFHE GPU has a known segfault
      in `EvalCoeffsToSlotsPrecompute`. See issue tracking for updates.

For OpenFHE users who need inverse sqrt without bootstrap:
    >>> enc_result = crypto_inv_sqrt_shallow(enc_x, domain=(1.0, 10.0))
    
This provides good accuracy (MRE < 1e-5) for narrower domains without
requiring the bootstrap operation that triggers the OpenFHE GPU bug.
"""

from .crypto_inv_sqrt import crypto_inv_sqrt, crypto_inv_sqrt_shallow
from .crypto_reciprocal import crypto_reciprocal_shallow
from .normalization import encrypted_mean, encrypted_std, encrypted_variance

__all__ = [
    "crypto_inv_sqrt",
    "crypto_inv_sqrt_shallow",
    "crypto_reciprocal_shallow",
    "encrypted_mean",
    "encrypted_variance",
    "encrypted_std",
]
