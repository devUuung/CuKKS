"""
cukks.nn - Encrypted neural network modules.

This module provides PyTorch-like layers for encrypted inference.
Each layer mirrors its torch.nn counterpart but operates on encrypted data.

Example:
    >>> import torch.nn as nn
    >>> import cukks
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
    >>> enc_model = cukks.convert(model, ctx)
    >>> 
    >>> # Run encrypted inference
    >>> enc_output = enc_model(enc_input)
"""

from .module import EncryptedModule, EncryptedIdentity
from .linear import EncryptedLinear
from .block_diagonal import BlockDiagonalLinear
from .block_diagonal_low_rank import BlockDiagLowRankLinear
from .encrypted_block_diag_lr import EncryptedBlockDiagLowRank
from .activations import (
    EncryptedSquare,
    EncryptedReLU,
    EncryptedGELU,
    EncryptedSiLU,
    EncryptedSigmoid,
    EncryptedTanh,
)
from .conv import EncryptedConv2d
from .conv1d import EncryptedConv1d
from .conv_transpose import EncryptedConvTranspose2d
from .pooling import EncryptedAvgPool2d, EncryptedMaxPool2d
from .adaptive_pooling import EncryptedAdaptiveAvgPool2d
from .flatten import EncryptedFlatten
from .sequential import EncryptedSequential
from .batchnorm import EncryptedBatchNorm1d, EncryptedBatchNorm2d
from .layernorm import EncryptedLayerNorm
from .inverse_free_layernorm import EncryptedInverseFreeLayerNorm
from .groupnorm import EncryptedGroupNorm
from .instancenorm import EncryptedInstanceNorm1d, EncryptedInstanceNorm2d
from .dropout import EncryptedDropout
from .residual import EncryptedResidualBlock
from .attention import EncryptedApproxAttention
from .upsample import EncryptedUpsample
from .embedding import EncryptedEmbedding
from .pixel_shuffle import EncryptedPixelShuffle, EncryptedPixelUnshuffle
from .padding import (
    EncryptedZeroPad2d,
    EncryptedConstantPad2d,
    EncryptedReflectionPad2d,
    EncryptedReplicationPad2d,
)

__all__ = [
    "EncryptedModule",
    "EncryptedIdentity",
    "EncryptedLinear",
    "BlockDiagonalLinear",
    "BlockDiagLowRankLinear",
    "EncryptedBlockDiagLowRank",
    "EncryptedSquare",
    "EncryptedReLU",
    "EncryptedGELU",
    "EncryptedSiLU",
    "EncryptedSigmoid",
    "EncryptedTanh",
    "EncryptedConv2d",
    "EncryptedConv1d",
    "EncryptedConvTranspose2d",
    "EncryptedAvgPool2d",
    "EncryptedMaxPool2d",
    "EncryptedAdaptiveAvgPool2d",
    "EncryptedFlatten",
    "EncryptedSequential",
    "EncryptedBatchNorm1d",
    "EncryptedBatchNorm2d",
    "EncryptedLayerNorm",
    "EncryptedInverseFreeLayerNorm",
    "EncryptedGroupNorm",
    "EncryptedInstanceNorm1d",
    "EncryptedInstanceNorm2d",
    "EncryptedDropout",
    "EncryptedResidualBlock",
    "EncryptedApproxAttention",
    "EncryptedUpsample",
    "EncryptedEmbedding",
    "EncryptedPixelShuffle",
    "EncryptedPixelUnshuffle",
    "EncryptedZeroPad2d",
    "EncryptedConstantPad2d",
    "EncryptedReflectionPad2d",
    "EncryptedReplicationPad2d",
]
