from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .tensor import EncryptedTensor


@dataclass
class InferenceConfig:
    poly_mod_degree: int
    scale_bits: int
    security_level: Optional[str]
    mult_depth: int
    enable_bootstrap: bool
    level_budget: Optional[Tuple[int, int]]

    @classmethod
    def for_depth(cls, mult_depth: int, **kwargs: Any) -> InferenceConfig: ...

    @classmethod
    def for_model(
        cls,
        model: nn.Module,
        activation_degree: int = 4,
        **kwargs: Any,
    ) -> InferenceConfig: ...


class CKKSInferenceContext:
    config: InferenceConfig
    device: str

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        *,
        device: Optional[str] = None,
        rotations: Optional[List[int]] = None,
        use_bsgs: bool = True,
        max_rotation_dim: Optional[int] = None,
        auto_bootstrap: bool = False,
        bootstrap_threshold: int = 2,
        cnn_config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = True,
        batch_size: int = 1,
        architecture: str = "default",
    ) -> None: ...

    def encrypt(self, tensor: torch.Tensor) -> EncryptedTensor: ...
    def decrypt(self, encrypted: EncryptedTensor, shape: Optional[Sequence[int]] = None) -> torch.Tensor: ...
    def encrypt_batch(self, samples: List[torch.Tensor], slots_per_sample: Optional[int] = None) -> EncryptedTensor: ...
    def decrypt_batch(
        self,
        encrypted: EncryptedTensor,
        num_samples: Optional[int] = None,
        sample_shape: Optional[Sequence[int]] = None,
    ) -> List[torch.Tensor]: ...
    def encrypt_sequence(
        self,
        tensor: torch.Tensor,
        *,
        num_heads: int,
        block_size: Optional[int] = None,
    ) -> EncryptedTensor: ...
    def decrypt_sequence(self, encrypted: EncryptedTensor) -> torch.Tensor: ...
    def encrypt_cnn_input(self, image: torch.Tensor, conv_params: List[Dict[str, Any]]) -> EncryptedTensor: ...
    def encrypt_cnn_input_batch(
        self,
        images: List[torch.Tensor],
        conv_params: List[Dict[str, Any]],
    ) -> EncryptedTensor: ...

    @classmethod
    def for_model(
        cls,
        model: nn.Module,
        activation_degree: int = 4,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> CKKSInferenceContext: ...

    @classmethod
    def for_depth(cls, depth: int, **kwargs: Any) -> CKKSInferenceContext: ...

    def save_context(self, path: Union[str, Path], *, allow_unsafe_pickle: bool = False) -> None: ...

    @classmethod
    def load_context(
        cls,
        path: Union[str, Path],
        *,
        allow_unsafe_pickle: bool = False,
        device: Optional[str] = None,
        enable_gpu: Optional[bool] = None,
    ) -> CKKSInferenceContext: ...

    def close(self) -> None: ...
    def plain_cache_info(self) -> Dict[str, Any]: ...
