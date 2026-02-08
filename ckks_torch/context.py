"""
CKKS Inference Context - Easy setup for encrypted deep learning inference.

This module provides a high-level interface for setting up CKKS encryption
parameters optimized for neural network inference.
"""

from __future__ import annotations

import math
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

if TYPE_CHECKING:
    from .tensor import EncryptedTensor

from .batching.packing import SlotPacker

CKKSConfig = None  # type: ignore
CKKSContext = None  # type: ignore
_BACKEND_AVAILABLE: bool | None = None


@dataclass
class InferenceConfig:
    poly_mod_degree: int = 16384
    scale_bits: int = 40
    security_level: Optional[str] = "128_classic"
    mult_depth: int = 4
    enable_bootstrap: bool = False
    level_budget: Optional[Tuple[int, int]] = None
    
    _coeff_mod_bits: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._coeff_mod_bits is None:
            middle_bits = tuple([self.scale_bits] * self.mult_depth)
            object.__setattr__(self, '_coeff_mod_bits', (60,) + middle_bits + (60,))
    
    @property
    def coeff_mod_bits(self) -> Tuple[int, ...]:
        if self._coeff_mod_bits is None:
            middle_bits = tuple([self.scale_bits] * self.mult_depth)
            return (60,) + middle_bits + (60,)
        return self._coeff_mod_bits
    
    @property
    def num_slots(self) -> int:
        return self.poly_mod_degree // 2
    
    @property
    def resolved_level_budget(self) -> Optional[Tuple[int, int]]:
        if self.level_budget is not None:
            return self.level_budget
        if self.enable_bootstrap:
            return (3, 3)
        return None
    
    @classmethod
    def for_depth(cls, mult_depth: int, **kwargs: Any) -> "InferenceConfig":
        mult_depth = max(1, mult_depth)
        enable_bootstrap = kwargs.pop("enable_bootstrap", False)
        level_budget = kwargs.pop("level_budget", None)
        security_level = kwargs.pop("security_level", "128_classic")
        scale_bits = kwargs.pop("scale_bits", 50)
        
        # Check if poly_mod_degree is explicitly passed
        poly_mod_degree = kwargs.pop("poly_mod_degree", None)

        if enable_bootstrap:
            if poly_mod_degree is None:
                poly_mod_degree = 65536
            effective_depth = mult_depth + 2
        else:
            effective_depth = mult_depth + 2
            if poly_mod_degree is None:
                if effective_depth <= 6:
                    poly_mod_degree = 16384
                elif effective_depth <= 16:
                    poly_mod_degree = 32768
                else:
                    poly_mod_degree = 65536

        return cls(
            poly_mod_degree=poly_mod_degree,
            mult_depth=effective_depth,
            scale_bits=scale_bits,
            security_level=security_level,
            enable_bootstrap=enable_bootstrap,
            level_budget=level_budget,
            **kwargs,
        )
    
    @classmethod
    def for_model(
        cls,
        model: torch.nn.Module,
        activation_degree: int = 4,
        use_square_activation: bool = False,
        **kwargs: Any,
    ) -> "InferenceConfig":
        depth = _estimate_model_depth(
            model,
            activation_degree=activation_degree,
            use_square_activation=use_square_activation,
        )
        return cls.for_depth(depth, **kwargs)


def _estimate_model_depth(
    model: torch.nn.Module,
    activation_degree: int = 4,
    use_square_activation: bool = False,
) -> int:
    depth = 0
    if use_square_activation:
        poly_depth = 1
    else:
        poly_depth = max(1, math.ceil(math.log2(activation_degree + 1)))

    from .nn.block_diagonal import BlockDiagonalLinear
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear

    for module in model.modules():
        if isinstance(module, BlockDiagLowRankLinear):
            depth += 2 if module.rank > 0 else 1
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, BlockDiagonalLinear)):
            depth += 1
        elif isinstance(module, (
            torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU,
            torch.nn.Sigmoid, torch.nn.Tanh,
        )):
            depth += poly_depth
    return max(1, depth)


def _get_model_dimensions(model: torch.nn.Module, input_shape: Optional[Tuple[int, ...]] = None) -> List[int]:
    dims = []
    has_conv = False
    
    from .nn.block_diagonal import BlockDiagonalLinear
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear

    for module in model.modules():
        if isinstance(module, (BlockDiagonalLinear, BlockDiagLowRankLinear)):
            dims.append(module.in_features)
            dims.append(module.out_features)
        elif isinstance(module, torch.nn.Linear):
            dims.append(module.in_features)
            dims.append(module.out_features)
        elif isinstance(module, torch.nn.Conv2d):
            has_conv = True
            dims.append(module.in_channels * module.kernel_size[0] * module.kernel_size[1])
            dims.append(module.out_channels * module.kernel_size[0] * module.kernel_size[1])
    
    if has_conv and not dims:
        dims.append(4096)
    
    return dims


def compute_bsgs_rotations(max_dim: int, bsgs_n1: Optional[int] = None) -> List[int]:
    if max_dim <= 1:
        return []
    
    if bsgs_n1 is None:
        bsgs_n1 = max(1, int(math.ceil(math.sqrt(max_dim))))
    
    bsgs_n2 = (max_dim + bsgs_n1 - 1) // bsgs_n1
    
    baby_steps = list(range(1, bsgs_n1))
    giant_steps = [i * bsgs_n1 for i in range(1, bsgs_n2 + 1) if i * bsgs_n1 < max_dim]
    
    rotations = sorted(set(baby_steps + giant_steps))
    return rotations


def compute_cnn_rotations(
    image_height: int,
    image_width: int,
    channels: int,
    pool_size: int = 2,
    pool_stride: int = 2,
) -> List[int]:
    rotations = set()
    if pool_size == 2 and pool_stride == 2:
        W = image_width
        offsets = [
            channels,
            W * channels,
            (W + 1) * channels,
        ]
        rotations.update(offsets)
    else:
        for dy in range(pool_size):
            for dx in range(pool_size):
                if dy == 0 and dx == 0:
                    continue
                offset = (dy * image_width + dx) * channels
                rotations.add(offset)
    return sorted(rotations)


def compute_rotations_for_model(model: torch.nn.Module, use_bsgs: bool = True) -> List[int]:
    from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear

    dims = _get_model_dimensions(model)
    if not dims:
        return [1, -1]
    
    max_dim = max(dims)
    
    if use_bsgs:
        rotations = compute_bsgs_rotations(max_dim)
    else:
        rotations = list(range(1, max_dim))
    
    for module in model.modules():
        if isinstance(module, BlockDiagLowRankLinear) and module.rank > 0:
            step = 1
            while step <= max_dim * 64:
                rotations.append(step)
                step *= 2
            break

    neg_rotations = [-r for r in rotations if r > 0]
    return sorted(set(rotations + neg_rotations))


class CKKSInferenceContext:
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
    ):
        if cnn_config is not None and config is None:
            config = InferenceConfig(mult_depth=6, security_level=None)
        
        self.config = config or InferenceConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bsgs = use_bsgs
        self._auto_bootstrap = auto_bootstrap
        self._bootstrap_threshold = bootstrap_threshold
        self._cnn_config = cnn_config
        self._enable_gpu = enable_gpu
        
        _resolved_max_dim = max_rotation_dim
        if rotations is None:
            max_dim = max_rotation_dim or min(self.config.num_slots, 1024)
            _resolved_max_dim = max_dim
            if use_bsgs:
                rotations = compute_bsgs_rotations(max_dim)
                neg_rotations = [-r for r in rotations if r > 0]
                rotations = sorted(set(rotations + neg_rotations))
            else:
                rotations = list(range(1, max_dim)) + list(range(-max_dim + 1, 0))
            
            if cnn_config is not None:
                cnn_rots = compute_cnn_rotations(
                    image_height=cnn_config.get('image_height', 8),
                    image_width=cnn_config.get('image_width', 8),
                    channels=cnn_config.get('channels', 4),
                    pool_size=cnn_config.get('pool_size', 2),
                    pool_stride=cnn_config.get('pool_stride', 2),
                )
                cnn_neg_rots = [-r for r in cnn_rots if r > 0]
                all_rots = set(rotations) | set(cnn_rots) | set(cnn_neg_rots)
                
                H = cnn_config.get('image_height', 8)
                W = cnn_config.get('image_width', 8)
                C = cnn_config.get('channels', 4)
                total_sparse_slots = H * W * C
                
                fc_rots = compute_bsgs_rotations(total_sparse_slots)
                fc_neg_rots = [-r for r in fc_rots if r > 0]
                all_rots = all_rots | set(fc_rots) | set(fc_neg_rots)
                
                rotations = sorted(all_rots)
        
        self._rotations = rotations
        self._max_rotation_dim = _resolved_max_dim
        self._initialized = False
        self._init_lock = threading.Lock()
    
    @staticmethod
    def _load_backend():
        global CKKSConfig, CKKSContext, _BACKEND_AVAILABLE
        if _BACKEND_AVAILABLE is not None:
            return
        try:
            from ckks import CKKSConfig as _Cfg, CKKSContext as _Ctx
            CKKSConfig = _Cfg
            CKKSContext = _Ctx
            _BACKEND_AVAILABLE = True
        except ImportError:
            _BACKEND_AVAILABLE = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            self._load_backend()
            
            if CKKSConfig is None or CKKSContext is None:
                raise RuntimeError(
                    "CKKS backend not available. Please install openfhe_backend:\n"
                    "  pip install -e bindings/openfhe_backend"
                )
            
            ckks_config = CKKSConfig(
                poly_mod_degree=self.config.poly_mod_degree,
                coeff_mod_bits=self.config.coeff_mod_bits,
                scale_bits=self.config.scale_bits,
                security_level=self.config.security_level,
                enable_bootstrap=self.config.enable_bootstrap,
                level_budget=self.config.resolved_level_budget,
                rotations=self._rotations,
                relin=True,
                generate_conjugate_keys=True,
            )
            
            self._ctx = CKKSContext(ckks_config, device=self.device, enable_gpu=self._enable_gpu)
            self._initialized = True
    
    @property
    def num_slots(self) -> int:
        return self.config.num_slots
    
    @property
    def auto_bootstrap(self) -> bool:
        return self._auto_bootstrap
    
    @property
    def bootstrap_threshold(self) -> int:
        return self._bootstrap_threshold
    
    @property
    def backend(self) -> Any:
        self._ensure_initialized()
        return self._ctx
    
    def encrypt(self, tensor: torch.Tensor) -> "EncryptedTensor":
        from .tensor import EncryptedTensor
        
        self._ensure_initialized()
        
        flat = tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        original_size = flat.numel()
        
        if original_size > self.num_slots:
            raise ValueError(
                f"Tensor size {original_size} exceeds available slots {self.num_slots}. "
                f"Consider using a larger poly_mod_degree or batching."
            )
        
        if self.use_bsgs and original_size < self.num_slots:
            indices = torch.arange(self.num_slots) % original_size
            flat = flat[indices]
        
        cipher = self._ctx.encrypt(flat)
        enc_tensor = EncryptedTensor(cipher, tuple(tensor.shape), self)
        enc_tensor._original_size = original_size
        return enc_tensor
    
    def decrypt(
        self,
        encrypted: "EncryptedTensor",
        shape: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        self._ensure_initialized()
        
        cipher = encrypted
        if getattr(encrypted, '_needs_rescale', False):
            cipher = encrypted.rescale()
        
        target_shape = tuple(shape) if shape else cipher.shape
        target_numel = int(math.prod(target_shape)) if target_shape else 1
        
        result = self._ctx.decrypt(cipher._cipher, shape=None)
        result = result[:target_numel].reshape(target_shape)
        
        if self.device != "cpu":
            result = result.to(self.device)
            
        return result
    
    def encrypt_batch(
        self,
        samples: List[torch.Tensor],
        slots_per_sample: Optional[int] = None,
    ) -> "EncryptedTensor":
        from .tensor import EncryptedTensor
        
        self._ensure_initialized()

        import warnings
        warnings.warn(
            "encrypt_batch() packs multiple samples into a single ciphertext. "
            "Standard encrypted layers (EncryptedLinear, etc.) do not support batched layout "
            "and will produce incorrect results. Use per-sample encryption for standard layers.",
            stacklevel=2,
        )
        
        if not samples:
            raise ValueError("Cannot encrypt empty list of samples")
        
        first_sample_size = samples[0].numel()
        if slots_per_sample is None:
            slots_per_sample = first_sample_size
        
        packer = SlotPacker(slots_per_sample, self.num_slots)
        packed = packer.pack(samples)
        
        cipher = self._ctx.encrypt(packed.to(dtype=torch.float64, device="cpu"))
        
        batch_shape = (len(samples), slots_per_sample)
        
        return EncryptedTensor(cipher, batch_shape, self)
    
    def decrypt_batch(
        self,
        encrypted: "EncryptedTensor",
        num_samples: Optional[int] = None,
        sample_shape: Optional[Sequence[int]] = None,
    ) -> List[torch.Tensor]:
        self._ensure_initialized()
        
        if num_samples is None:
            if len(encrypted.shape) >= 1:
                num_samples = encrypted.shape[0]
            else:
                raise ValueError(
                    "Cannot infer num_samples from encrypted tensor shape. "
                    "Please provide num_samples explicitly."
                )
        
        if len(encrypted.shape) >= 2:
            slots_per_sample = encrypted.shape[1]
        else:
            slots_per_sample = encrypted.size // num_samples
        
        full_result = self._ctx.decrypt(encrypted._cipher, shape=None)
        
        packer = SlotPacker(slots_per_sample, self.num_slots)
        samples = packer.unpack(full_result, num_samples)
        
        if sample_shape is not None:
            target_numel = int(math.prod(sample_shape))
            reshaped_samples = []
            for sample in samples:
                trimmed = sample[:target_numel].to(torch.float32)
                reshaped_samples.append(trimmed.reshape(sample_shape))
            samples = reshaped_samples
        else:
            samples = [s.to(torch.float32) for s in samples]
        
        if self.device != "cpu":
            samples = [s.to(self.device) for s in samples]
        
        return samples
    
    def encrypt_cnn_input(
        self,
        image: torch.Tensor,
        conv_params: List[Dict[str, Any]],
    ) -> "EncryptedTensor":
        import torch.nn.functional as F
        
        self._ensure_initialized()
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        if not conv_params:
            return self.encrypt(image)
        
        first_conv = conv_params[0]
        kernel_size = first_conv['kernel_size']
        stride = first_conv.get('stride', (1, 1))
        padding = first_conv.get('padding', (0, 0))
        
        if padding != (0, 0):
            image = F.pad(image, (padding[1], padding[1], padding[0], padding[0]))
        
        patches = F.unfold(image.to(torch.float64), kernel_size, stride=stride)
        patches = patches.transpose(1, 2)
        patches = patches.squeeze(0)
        
        num_patches, patch_features = patches.shape
        flat_patches = patches.flatten()
        
        enc_tensor = self.encrypt(flat_patches)
        
        enc_tensor._cnn_layout = {
            'num_patches': num_patches,
            'patch_features': patch_features,
            'original_shape': tuple(image.shape),
        }
        enc_tensor._shape = (num_patches, patch_features)
        
        return enc_tensor

    @classmethod
    def for_model(
        cls,
        model: torch.nn.Module,
        use_bsgs: bool = True,
        activation_degree: int = 4,
        use_square_activation: bool = False,
        **kwargs: Any,
    ) -> "CKKSInferenceContext":
        enable_bootstrap = kwargs.pop("enable_bootstrap", False)
        level_budget = kwargs.pop("level_budget", None)
        scale_bits = kwargs.pop("scale_bits", 50)
        security_level = kwargs.pop("security_level", "128_classic")
        poly_mod_degree = kwargs.pop("poly_mod_degree", None)

        config = InferenceConfig.for_model(
            model,
            activation_degree=activation_degree,
            use_square_activation=use_square_activation,
            enable_bootstrap=enable_bootstrap,
            level_budget=level_budget,
            scale_bits=scale_bits,
            security_level=security_level,
            poly_mod_degree=poly_mod_degree,
        )
        rotations = compute_rotations_for_model(model, use_bsgs=use_bsgs)
        extra_rotations = kwargs.pop("rotations", [])
        if extra_rotations:
            rotations.extend(extra_rotations)
            rotations = sorted(list(set(rotations)))
        dims = _get_model_dimensions(model)
        max_dim = max(dims) if dims else 1024
        
        has_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
        if has_conv:
            max_dim = max(max_dim, 8192)
        
        return cls(config, rotations=rotations, use_bsgs=use_bsgs, max_rotation_dim=max_dim, **kwargs)
    
    @classmethod
    def for_depth(
        cls,
        depth: int,
        **kwargs: Any,
    ) -> "CKKSInferenceContext":
        config = InferenceConfig.for_depth(depth)
        return cls(config, **kwargs)
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return (
            f"CKKSInferenceContext("
            f"slots={self.num_slots}, "
            f"depth={self.config.mult_depth}, "
            f"device='{self.device}', "
            f"status={status})"
        )
    
    def save_context(self, path: Union[str, Path]) -> None:
        config_data = {
            "poly_mod_degree": self.config.poly_mod_degree,
            "scale_bits": self.config.scale_bits,
            "security_level": self.config.security_level,
            "mult_depth": self.config.mult_depth,
            "enable_bootstrap": self.config.enable_bootstrap,
            "device": self.device,
            "use_bsgs": self.use_bsgs,
            "rotations": self._rotations,
            "max_rotation_dim": self._max_rotation_dim,
            "auto_bootstrap": self._auto_bootstrap,
            "bootstrap_threshold": self._bootstrap_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(config_data, f)
    
    @classmethod
    def load_context(cls, path: Union[str, Path]) -> "CKKSInferenceContext":
        with open(path, "rb") as f:
            config_data = pickle.load(f)
        import warnings
        warnings.warn(
            "CKKSInferenceContext.load_context() uses pickle deserialization which can "
            "execute arbitrary code. Only load context files from trusted sources.",
            stacklevel=2,
        )
        
        inference_config = InferenceConfig(
            poly_mod_degree=config_data["poly_mod_degree"],
            scale_bits=config_data["scale_bits"],
            security_level=config_data["security_level"],
            mult_depth=config_data["mult_depth"],
            enable_bootstrap=config_data["enable_bootstrap"],
        )
        
        return cls(
            config=inference_config,
            device=config_data["device"],
            rotations=config_data["rotations"],
            use_bsgs=config_data["use_bsgs"],
            max_rotation_dim=config_data["max_rotation_dim"],
            auto_bootstrap=config_data["auto_bootstrap"],
            bootstrap_threshold=config_data["bootstrap_threshold"],
        )
    
    load = load_context
