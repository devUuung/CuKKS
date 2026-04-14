from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .tensor import EncryptedTensor


def copy_cnn_layout(layout: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if layout is None:
        return None
    return copy.deepcopy(layout)


def copy_runtime_metadata(target: "EncryptedTensor", source: "EncryptedTensor") -> None:
    target._original_size = source._original_size
    target._cnn_layout = copy_cnn_layout(source._cnn_layout)
    target._packed_batch = source._packed_batch
    target._batch_size = source._batch_size
    target._slots_per_sample = source._slots_per_sample
    target._packed_sample_shape = source._packed_sample_shape
    target._packing_layout = source._packing_layout
    target._sigma_factor = source._sigma_factor


def propagate_runtime_metadata(
    source: "EncryptedTensor", target: "EncryptedTensor"
) -> "EncryptedTensor":
    if source._packed_batch:
        target._packed_batch = True
        target._batch_size = source._batch_size
        target._slots_per_sample = source._slots_per_sample
        target._packed_sample_shape = source._packed_sample_shape
    target._cnn_layout = copy_cnn_layout(source._cnn_layout)
    target._packing_layout = source._packing_layout
    target._sigma_factor = source._sigma_factor
    return target


def packed_sample_dims(tensor: "EncryptedTensor") -> Optional[Tuple[int, ...]]:
    if not tensor._packed_batch:
        return None
    if tensor._packed_sample_shape is not None:
        return tensor._packed_sample_shape
    if (
        tensor._batch_size is not None
        and len(tensor._shape) > 1
        and tensor._shape[0] == tensor._batch_size
    ):
        return tuple(tensor._shape[1:])
    return None


def refresh_packed_shape_metadata(tensor: "EncryptedTensor") -> None:
    if not tensor._packed_batch or tensor._batch_size is None:
        return

    total_size = tensor.size
    if tensor._batch_size <= 0 or total_size % tensor._batch_size != 0:
        tensor._packed_batch = False
        tensor._batch_size = None
        tensor._slots_per_sample = None
        tensor._packed_sample_shape = None
        return

    tensor._slots_per_sample = total_size // tensor._batch_size
    if len(tensor._shape) > 1 and tensor._shape[0] == tensor._batch_size:
        sample_size = int(math.prod(tensor._shape[1:])) if len(tensor._shape) > 1 else 1
        if sample_size == tensor._slots_per_sample:
            tensor._packed_sample_shape = tuple(tensor._shape[1:])
