"""
ckks_torch.nn - Encrypted neural network modules.

This module provides PyTorch-like layers for encrypted inference.
Each layer mirrors its torch.nn counterpart but operates on encrypted data.

Example:
    >>> import torch.nn as nn
    >>> import ckks_torch
    >>> 
    >>> # Train a plaintext model
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 128),
    ...     nn.ReLU(),
    ...     nn.Linear(128, 10)
    ... )
    >>> train(model, data)
    >>> 
    >>> # Convert to encrypted model
    >>> enc_model = ckks_torch.convert(model, ctx)
    >>> 
    >>> # Run encrypted inference
    >>> enc_output = enc_model(enc_input)
"""

from .module import EncryptedModule, EncryptedIdentity
from .linear import EncryptedLinear
from .tt_linear import EncryptedTTLinear
from .tt_conv import EncryptedTTConv2d
from .activations import (
    EncryptedSquare,
    EncryptedReLU,
    EncryptedGELU,
    EncryptedSiLU,
    EncryptedSigmoid,
    EncryptedTanh,
)
from .conv import EncryptedConv2d
from .pooling import EncryptedAvgPool2d, EncryptedMaxPool2d
from .flatten import EncryptedFlatten
from .sequential import EncryptedSequential
from .batchnorm import EncryptedBatchNorm1d, EncryptedBatchNorm2d
from .layernorm import EncryptedLayerNorm
from .dropout import EncryptedDropout
from .residual import EncryptedResidualBlock
from .attention import EncryptedApproxAttention

__all__ = [
    "EncryptedModule",
    "EncryptedIdentity",
    "EncryptedLinear",
    "EncryptedTTLinear",
    "EncryptedTTConv2d",
    "EncryptedSquare",
    "EncryptedReLU",
    "EncryptedGELU",
    "EncryptedSiLU",
    "EncryptedSigmoid",
    "EncryptedTanh",
    "EncryptedConv2d",
    "EncryptedAvgPool2d",
    "EncryptedMaxPool2d",
    "EncryptedFlatten",
    "EncryptedSequential",
    "EncryptedBatchNorm1d",
    "EncryptedBatchNorm2d",
    "EncryptedLayerNorm",
    "EncryptedDropout",
    "EncryptedResidualBlock",
    "EncryptedApproxAttention",
]
