from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

from .nn import EncryptedModule
from .nn.block_diagonal import BlockDiagonalLinear
from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear


def build_converter_registry(
    converter: object,
    activation_map: Dict[Type[nn.Module], Type[EncryptedModule]],
) -> Dict[Type[nn.Module], Callable]:
    registry: Dict[Type[nn.Module], Callable] = {
        nn.Linear: converter._convert_linear,
        BlockDiagonalLinear: converter._convert_block_diagonal,
        BlockDiagLowRankLinear: converter._convert_block_diag_low_rank,
        nn.Conv1d: converter._convert_conv1d,
        nn.Conv2d: converter._convert_conv2d,
        nn.ConvTranspose2d: converter._convert_conv_transpose2d,
        nn.AvgPool2d: converter._convert_avgpool2d,
        nn.AdaptiveAvgPool2d: converter._convert_adaptive_avgpool2d,
        nn.Flatten: converter._convert_flatten,
        nn.Sequential: converter._convert_sequential,
        nn.BatchNorm1d: converter._convert_batchnorm1d,
        nn.BatchNorm2d: converter._convert_batchnorm2d,
        nn.GroupNorm: converter._convert_groupnorm,
        nn.InstanceNorm1d: converter._convert_instancenorm1d,
        nn.InstanceNorm2d: converter._convert_instancenorm2d,
        nn.Dropout: converter._convert_dropout,
        nn.Dropout2d: converter._convert_dropout,
        nn.MaxPool2d: converter._convert_maxpool2d,
        nn.LayerNorm: converter._convert_layernorm,
        nn.MultiheadAttention: converter._convert_attention,
        nn.Upsample: converter._convert_upsample,
        nn.UpsamplingNearest2d: converter._convert_upsample,
        nn.UpsamplingBilinear2d: converter._convert_upsample,
        nn.Embedding: converter._convert_embedding,
        nn.PixelShuffle: converter._convert_pixel_shuffle,
        nn.PixelUnshuffle: converter._convert_pixel_unshuffle,
        nn.ZeroPad2d: converter._convert_zeropad2d,
        nn.ConstantPad2d: converter._convert_constantpad2d,
        nn.ReflectionPad2d: converter._convert_reflectionpad2d,
        nn.ReplicationPad2d: converter._convert_replicationpad2d,
    }
    for torch_type in activation_map:
        registry[torch_type] = converter._convert_activation
    return registry
