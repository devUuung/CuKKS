from __future__ import annotations

import math

import torch


def estimate_model_depth(
    model: torch.nn.Module,
    activation_degree: int = 4,
    use_square_activation: bool = False,
    enable_gpu: bool = False,
    attention_normalization_mode: str = "power_softmax",
    inverse_free_ln_names: frozenset[str] = frozenset(),
) -> int:
    depth = 0
    if use_square_activation:
        poly_depth = 1
    elif enable_gpu and activation_degree >= 4:
        poly_depth = 3
    else:
        poly_depth = max(1, math.ceil(math.log2(activation_degree + 1)))

    from .nn.block_diagonal import BlockDiagonalLinear
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear

    module_to_name = {id(module): name for name, module in model.named_modules()}

    for module in model.modules():
        if isinstance(module, BlockDiagLowRankLinear):
            depth += 2 if module.rank > 0 else 1
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, BlockDiagonalLinear)):
            depth += 1
            if enable_gpu:
                depth += 1
        elif isinstance(
            module,
            (
                torch.nn.ReLU,
                torch.nn.GELU,
                torch.nn.SiLU,
                torch.nn.Sigmoid,
                torch.nn.Tanh,
            ),
        ):
            depth += poly_depth
        elif isinstance(module, torch.nn.MultiheadAttention):
            depth += 9 if attention_normalization_mode == "gaussian" else 10
        elif isinstance(module, torch.nn.LayerNorm):
            module_name = module_to_name.get(id(module), "")
            depth += 5 if module_name in inverse_free_ln_names else 18
        elif isinstance(module, torch.nn.GroupNorm):
            depth += activation_degree + 3
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            depth += 1
    return max(1, depth)
