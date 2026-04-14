from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class RotationRequirements:
    generic_dims: List[int]
    reduction_lengths: List[int]
    pack_shifts: List[int]
    extra_rotations: List[int]


def collect_rotation_requirements(
    model: torch.nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    *,
    small_output_threshold: int = 16,
) -> RotationRequirements:
    from .nn.block_diagonal import BlockDiagonalLinear
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear
    from .nn.conv import EncryptedConv2d as _EC

    generic_dims: List[int] = []
    reduction_lengths: List[int] = []
    pack_shifts: set[int] = set()
    extra_rotations: set[int] = set()

    spatial_h, spatial_w = 28, 28
    if input_shape is not None:
        if len(input_shape) == 4:
            spatial_h, spatial_w = int(input_shape[2]), int(input_shape[3])
        elif len(input_shape) >= 2:
            spatial_h, spatial_w = int(input_shape[-2]), int(input_shape[-1])

    current_channels: Optional[int] = None
    if input_shape is not None:
        if len(input_shape) == 4:
            current_channels = int(input_shape[1])
        elif len(input_shape) == 3:
            current_channels = int(input_shape[0])

    pending_sparse_pool: Optional[Tuple[int, int]] = None

    for module in model.modules():
        if isinstance(module, BlockDiagLowRankLinear):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
            generic_dims.extend([module.in_features, module.out_features])
            pending_sparse_pool = None
            continue

        if isinstance(module, BlockDiagonalLinear):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
            generic_dims.extend([module.in_features, module.out_features])
            pending_sparse_pool = None
            continue

        if isinstance(module, torch.nn.MultiheadAttention):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
            generic_dims.extend([module.embed_dim, module.embed_dim * 8])
            pending_sparse_pool = None
            continue

        if isinstance(module, torch.nn.Conv2d):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
            kh, kw = (
                (module.kernel_size, module.kernel_size)
                if isinstance(module.kernel_size, int)
                else module.kernel_size
            )
            sh, sw = (
                (module.stride, module.stride) if isinstance(module.stride, int) else module.stride
            )
            if isinstance(module.padding, int):
                ph = pw = int(module.padding)
            elif isinstance(module.padding, tuple):
                ph, pw = int(module.padding[0]), int(module.padding[1])
            else:
                pending_sparse_pool = None
                continue

            kh, kw = int(kh), int(kw)
            sh, sw = int(sh), int(sw)

            out_h = (spatial_h + 2 * ph - kh) // sh + 1
            out_w = (spatial_w + 2 * pw - kw) // sw + 1
            num_patches = out_h * out_w
            patch_features = module.in_channels * kh * kw

            extra_rotations.update(
                _EC.required_packed_compact_rotations(
                    num_patches,
                    patch_features,
                    module.out_channels,
                )
            )

            spatial_h, spatial_w = out_h, out_w
            current_channels = int(module.out_channels)
            pending_sparse_pool = None
            continue

        if isinstance(module, torch.nn.GroupNorm):
            sample_size = (
                spatial_h * spatial_w * current_channels
                if current_channels is not None
                else int(module.num_channels)
            )
            generic_dims.append(sample_size)
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
                pending_sparse_pool = None
            continue

        if isinstance(module, torch.nn.AvgPool2d):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
                pending_sparse_pool = None

            if isinstance(module.kernel_size, int):
                pk_h = pk_w = module.kernel_size
            else:
                pk_h, pk_w = int(module.kernel_size[0]), int(module.kernel_size[1])

            stride = getattr(module, "stride", None)
            if isinstance(stride, int):
                ps_h = ps_w = stride
            elif stride is None:
                ps_h, ps_w = pk_h, pk_w
            else:
                ps_h, ps_w = int(stride[0]), int(stride[1])

            padding = getattr(module, "padding", 0)
            if isinstance(padding, int):
                p_h = p_w = padding
            else:
                p_h, p_w = int(padding[0]), int(padding[1])

            pool_total_slots = (
                spatial_h * spatial_w * current_channels if current_channels is not None else None
            )

            if (
                current_channels is not None
                and pk_h == 2
                and pk_w == 2
                and ps_h == 2
                and ps_w == 2
                and p_h == 0
                and p_w == 0
            ):
                extra_rotations.update(
                    compute_cnn_rotations(
                        spatial_h,
                        spatial_w,
                        current_channels,
                        pool_size=pk_h,
                        pool_stride=ps_h,
                    )
                )
                if pk_h == 2 and ps_h == 2:
                    compact_features = (spatial_h // ps_h) * (spatial_w // ps_w) * current_channels
                    total_slots = spatial_h * spatial_w * current_channels
                    pending_sparse_pool = (compact_features, total_slots)
                else:
                    pending_sparse_pool = None
            else:
                if pool_total_slots is not None:
                    generic_dims.append(pool_total_slots)
                pending_sparse_pool = None

            spatial_h = (spatial_h + 2 * p_h - pk_h) // ps_h + 1
            spatial_w = (spatial_w + 2 * p_w - pk_w) // ps_w + 1
            continue

        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            if pending_sparse_pool is not None:
                generic_dims.append(pending_sparse_pool[1])
                pending_sparse_pool = None

            if current_channels is not None:
                generic_dims.append(spatial_h * spatial_w * current_channels)

            output_size = module.output_size
            if isinstance(output_size, int):
                out_h = out_w = int(output_size)
            elif output_size is not None:
                out_h_s, out_w_s = output_size
                out_h = int(out_h_s) if out_h_s is not None else spatial_h
                out_w = int(out_w_s) if out_w_s is not None else spatial_w
            else:
                out_h, out_w = spatial_h, spatial_w
            spatial_h, spatial_w = out_h, out_w
            continue

        if isinstance(module, torch.nn.Linear):
            if pending_sparse_pool is not None and pending_sparse_pool[0] != module.in_features:
                generic_dims.append(pending_sparse_pool[1])
                pending_sparse_pool = None

            if (
                module.out_features <= small_output_threshold
                and module.out_features < module.in_features
            ):
                reduction_length = int(module.in_features)
                if pending_sparse_pool is not None and pending_sparse_pool[0] == module.in_features:
                    reduction_length = pending_sparse_pool[1]
                reduction_lengths.append(reduction_length)
                pack_shifts.update(range(1, int(module.out_features)))
            else:
                if pending_sparse_pool is not None:
                    generic_dims.append(pending_sparse_pool[1])
                    pending_sparse_pool = None
                generic_dims.extend([module.in_features, module.out_features])
            pending_sparse_pool = None
            continue

        if isinstance(
            module,
            (
                torch.nn.Flatten,
                torch.nn.ReLU,
                torch.nn.GELU,
                torch.nn.SiLU,
                torch.nn.Sigmoid,
                torch.nn.Tanh,
                torch.nn.Dropout,
            ),
        ):
            continue

        if pending_sparse_pool is not None:
            generic_dims.append(pending_sparse_pool[1])
            pending_sparse_pool = None

    if pending_sparse_pool is not None:
        generic_dims.append(pending_sparse_pool[1])

    return RotationRequirements(
        generic_dims=generic_dims,
        reduction_lengths=sorted(set(reduction_lengths)),
        pack_shifts=sorted(pack_shifts),
        extra_rotations=sorted(extra_rotations),
    )


def compute_reduction_rotations(length: int) -> List[int]:
    if length <= 1:
        return []

    rotations: List[int] = []
    step = 1
    while step < length:
        rotations.append(step)
        step *= 2
    return rotations


def compute_bsgs_rotations(max_dim: int, bsgs_n1: Optional[int] = None) -> List[int]:
    if max_dim <= 1:
        return []

    if bsgs_n1 is None:
        bsgs_n1 = max(1, int(math.ceil(math.sqrt(max_dim))))

    bsgs_n2 = (max_dim + bsgs_n1 - 1) // bsgs_n1
    baby_steps = list(range(1, bsgs_n1))
    giant_steps = [i * bsgs_n1 for i in range(1, bsgs_n2 + 1) if i * bsgs_n1 < max_dim]
    return sorted(set(baby_steps + giant_steps))


def compute_packed_batch_rotations(
    max_dim: int,
    batch_size: int = 1,
    bsgs_n1: Optional[int] = None,
) -> List[int]:
    rotations = set(compute_bsgs_rotations(max_dim, bsgs_n1))
    if batch_size > 1:
        for i in range(1, batch_size):
            rotations.add(i * max_dim)
    return sorted(rotations)


def compute_cnn_rotations(
    image_height: int,
    image_width: int,
    channels: int,
    pool_size: int = 2,
    pool_stride: int = 2,
) -> List[int]:
    rotations = set()
    if pool_size == 2 and pool_stride == 2:
        offsets = [channels, image_width * channels, (image_width + 1) * channels]
        rotations.update(offsets)
    else:
        for dy in range(pool_size):
            for dx in range(pool_size):
                if dy == 0 and dx == 0:
                    continue
                rotations.add((dy * image_width + dx) * channels)
    return sorted(rotations)


def compute_rotations_for_model(model: torch.nn.Module, use_bsgs: bool = True) -> List[int]:
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear

    requirements = collect_rotation_requirements(model)
    rotation_set = set(requirements.extra_rotations) | set(requirements.pack_shifts)
    max_dim = max(requirements.generic_dims) if requirements.generic_dims else 0

    if max_dim > 1 and use_bsgs:
        rotation_set.update(compute_bsgs_rotations(max_dim))
    elif max_dim > 1:
        rotation_set.update(range(1, max_dim))

    for length in requirements.reduction_lengths:
        rotation_set.update(compute_reduction_rotations(length))

    for module in model.modules():
        if isinstance(module, BlockDiagLowRankLinear) and module.rank > 0:
            step = 1
            while step <= max_dim * 64:
                rotation_set.add(step)
                step *= 2
            break

    return sorted(rotation_set | {-rotation for rotation in rotation_set if rotation > 0})
