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

# Low-level backend is imported lazily in _ensure_initialized() to avoid
# loading the GPU C++ extension (and initializing CUDA) at module import time.
# This prevents CUDA state corruption when mock tests import ckks_torch
# without needing the real backend.
CKKSConfig = None  # type: ignore
CKKSContext = None  # type: ignore
_BACKEND_AVAILABLE: bool | None = None


@dataclass
class InferenceConfig:
    """Configuration for encrypted inference.
    
    This provides sensible defaults for neural network inference while
    allowing customization for specific use cases.
    
    Attributes:
        poly_mod_degree: Ring dimension (power of 2). Larger = more slots but slower.
            Common values: 8192, 16384, 32768
        scale_bits: Precision bits for CKKS encoding. Higher = more precision but fewer levels.
        security_level: Security level string or bits (128, 192, 256). None to disable.
        mult_depth: Multiplicative depth needed for the network.
            Each layer with activation typically needs 2 multiplications.
        enable_bootstrap: Whether to enable bootstrapping for deep networks.
        level_budget: Bootstrapping level budget [coeffs_to_slots_levels, slots_to_coeffs_levels].
            Default [3, 3] for enable_bootstrap=True. Higher = better precision but more depth.
    """
    poly_mod_degree: int = 16384
    scale_bits: int = 40
    security_level: Optional[str] = "128_classic"
    mult_depth: int = 4
    enable_bootstrap: bool = False
    level_budget: Optional[Tuple[int, int]] = None
    
    # Auto-computed (use field with default_factory to avoid mutable default)
    _coeff_mod_bits: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._coeff_mod_bits is None:
            # Build coefficient modulus chain
            # First and last primes are larger for key switching
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
        """Number of plaintext slots available."""
        return self.poly_mod_degree // 2
    
    @property
    def resolved_level_budget(self) -> Optional[Tuple[int, int]]:
        """Level budget for bootstrapping. Returns [3, 3] if enabled but not set."""
        if self.level_budget is not None:
            return self.level_budget
        if self.enable_bootstrap:
            return (3, 3)
        return None
    
    @classmethod
    def for_depth(cls, mult_depth: int, **kwargs: Any) -> "InferenceConfig":
        """Create a config for a given multiplicative depth.

        Args:
            mult_depth: Total multiplicative levels the network needs.
            **kwargs: Overrides forwarded to __init__.
                Special keys consumed here:
                - ``security_level`` (default ``"none"``)
                - ``scale_bits`` (default ``50``)
                - ``enable_bootstrap`` (default ``False``)
                - ``level_budget`` (forwarded when bootstrapping)
        """
        mult_depth = max(1, mult_depth)
        enable_bootstrap = kwargs.pop("enable_bootstrap", False)
        level_budget = kwargs.pop("level_budget", None)
        security_level = kwargs.pop("security_level", "128_classic")
        scale_bits = kwargs.pop("scale_bits", 50)

        if enable_bootstrap:
            # When bootstrapping is enabled the C++ backend overrides
            # mult_depth to ``levelsAfterBootstrap + bootstrapDepth``.
            # levelsAfterBootstrap is hardcoded to 10 in both backends,
            # and bootstrapDepth ≈ 10-14 for level_budget=[3,3].
            # We need poly_mod_degree large enough for ~20-24 total depth.
            poly_mod_degree = kwargs.pop("poly_mod_degree", 65536)
            # mult_depth passed to __init__ doesn't matter much (C++ will
            # recompute), but set a reasonable value for coeff_mod_bits.
            effective_depth = mult_depth + 2
        else:
            # +2 safety margin for add_plain level matching and rescale headroom
            effective_depth = mult_depth + 2

            if effective_depth <= 6:
                poly_mod_degree = 16384
            elif effective_depth <= 16:
                poly_mod_degree = 32768
            else:
                poly_mod_degree = 65536
            poly_mod_degree = kwargs.pop("poly_mod_degree", poly_mod_degree)

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
        """Analyze a PyTorch model and create optimal config.
        
        Args:
            model: The PyTorch model to analyze.
            activation_degree: Polynomial degree for activation approximation.
            use_square_activation: If True, activations use x^2 (1 level each).
            **kwargs: Additional config overrides forwarded to for_depth().
        """
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
    """Estimate multiplicative depth of a model.

    Each Linear/Conv2d costs 1 level (cipher-plain matmul).
    Each activation costs ceil(log2(degree+1)) levels via Paterson-Stockmeyer.
    Square activation (x^2) costs exactly 1 level.
    """
    depth = 0
    if use_square_activation:
        poly_depth = 1
    else:
        poly_depth = max(1, math.ceil(math.log2(activation_degree + 1)))

    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
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
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
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
    """
    Compute rotation keys for Baby-step Giant-step algorithm.
    Reduces O(n) rotations to O(√n).
    """
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
    """
    Compute rotation keys needed for CNN pooling operations.
    
    For 2x2 average pooling on a (H, W, C) layout flattened to (H*W*C,),
    the rotations needed are the offsets to adjacent pool positions:
    - offset 0: (0, 0) - current position
    - offset C: (0, 1) - horizontal neighbor
    - offset W*C: (1, 0) - vertical neighbor
    - offset (W+1)*C: (1, 1) - diagonal neighbor
    
    Args:
        image_height: Height of the feature map after conv.
        image_width: Width of the feature map after conv.
        channels: Number of channels in the feature map.
        pool_size: Size of pooling window (default 2 for 2x2 pooling).
        pool_stride: Stride of pooling (default 2).
        
    Returns:
        List of rotation indices needed for pooling.
    """
    rotations = set()
    
    # For 2x2 pooling with stride 2
    if pool_size == 2 and pool_stride == 2:
        # Offsets for the 4 positions in the pool window
        # In (H*W, C) layout where each patch is stored as C consecutive values:
        W = image_width
        offsets = [
            channels,              # (0, 1) - horizontal neighbor  
            W * channels,          # (1, 0) - vertical neighbor
            (W + 1) * channels,    # (1, 1) - diagonal neighbor
        ]
        rotations.update(offsets)
    else:
        # General case: all offsets in the pool window
        for dy in range(pool_size):
            for dx in range(pool_size):
                if dy == 0 and dx == 0:
                    continue  # Skip (0, 0) - no rotation needed
                offset = (dy * image_width + dx) * channels
                rotations.add(offset)
    
    return sorted(rotations)


def compute_rotations_for_model(model: torch.nn.Module, use_bsgs: bool = True) -> List[int]:
    dims = _get_model_dimensions(model)
    if not dims:
        return [1, -1]
    
    max_dim = max(dims)
    
    if use_bsgs:
        rotations = compute_bsgs_rotations(max_dim)
    else:
        rotations = list(range(1, max_dim))
    
    neg_rotations = [-r for r in rotations if r > 0]
    return sorted(set(rotations + neg_rotations))


class CKKSInferenceContext:
    """High-level context for encrypted inference.
    
    This class wraps the low-level CKKS context and provides a convenient
    interface for deep learning practitioners.
    
    Example:
        >>> ctx = CKKSInferenceContext.for_model(my_model)
        >>> encrypted_input = ctx.encrypt(input_tensor)
        >>> # ... run encrypted inference ...
        >>> output = ctx.decrypt(encrypted_output)
    """
    
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
        # For CNN inference, we need more multiplicative depth
        # Conv(1) + Square(1) + Pool(1) + FC(2) = 5 minimum
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
            
            # Add CNN-specific rotations if cnn_config is provided
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
                
                # Add rotations for FC layer with sparse pooling output
                # The sparse layout has total_slots = H * W * C values
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
        """Import the CKKS backend on first use (lazy)."""
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
        """Lazy initialization of the CKKS context."""
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
        """Number of available plaintext slots."""
        return self.config.num_slots
    
    @property
    def auto_bootstrap(self) -> bool:
        """Whether auto-bootstrap is enabled."""
        return self._auto_bootstrap
    
    @property
    def bootstrap_threshold(self) -> int:
        """Depth threshold for auto-bootstrap."""
        return self._bootstrap_threshold
    
    @property
    def backend(self) -> Any:
        """Access the underlying CKKS context."""
        self._ensure_initialized()
        return self._ctx
    
    def encrypt(self, tensor: torch.Tensor) -> "EncryptedTensor":
        """Encrypt a PyTorch tensor.
        
        Args:
            tensor: Input tensor to encrypt. Will be flattened.
            
        Returns:
            EncryptedTensor wrapper around the ciphertext.
        """
        from .tensor import EncryptedTensor
        
        self._ensure_initialized()
        
        # Ensure proper dtype and flatten
        flat = tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        original_size = flat.numel()
        
        if original_size > self.num_slots:
            raise ValueError(
                f"Tensor size {original_size} exceeds available slots {self.num_slots}. "
                f"Consider using a larger poly_mod_degree or batching."
            )
        
        # For BSGS matmul: cyclically replicate data to fill all slots
        # This prevents zeros from wrapping into active positions during rotation
        if self.use_bsgs and original_size < self.num_slots:
            indices = torch.arange(self.num_slots) % original_size
            flat = flat[indices]
        
        cipher = self._ctx.encrypt(flat)
        enc_tensor = EncryptedTensor(cipher, tuple(tensor.shape), self)
        # Store original size for correct decryption and matmul
        enc_tensor._original_size = original_size
        return enc_tensor
    
    def decrypt(
        self,
        encrypted: "EncryptedTensor",
        shape: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        self._ensure_initialized()
        
        # Auto-rescale before decryption to avoid decode failures.
        # Under FLEXIBLEAUTO the ciphertext may still carry an un-rescaled
        # scale factor (noiseScaleDeg > 1) which causes OpenFHE's Decode()
        # to fail with "approximation error is too high".
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
        """Encrypt multiple samples into a single ciphertext using slot packing.
        
        This packs multiple samples contiguously into the CKKS slots, enabling
        SIMD-style parallel processing of all samples in a single ciphertext.
        
        Args:
            samples: List of tensors to encrypt. Each tensor will be flattened.
                    All samples must have the same number of elements.
            slots_per_sample: Number of slots to allocate per sample.
                             If None, uses the size of the first sample.
                             
        Returns:
            EncryptedTensor containing all packed samples.
            
        Raises:
            ValueError: If samples list is empty.
            ValueError: If total samples exceed available slots.
            
        Example:
            >>> ctx = CKKSInferenceContext()
            >>> samples = [torch.randn(784) for _ in range(8)]
            >>> enc_batch = ctx.encrypt_batch(samples)
            >>> # Run inference on all 8 samples at once
            >>> enc_output = enc_model(enc_batch)
            >>> outputs = ctx.decrypt_batch(enc_output, num_samples=8)
        """
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
        """Decrypt a batched ciphertext into individual samples.
        
        Args:
            encrypted: EncryptedTensor containing packed samples.
            num_samples: Number of samples to extract. If None, inferred from
                        the encrypted tensor's shape (first dimension).
            sample_shape: Shape for each output sample. If None, each sample
                         is returned as a 1D tensor.
                         
        Returns:
            List of decrypted tensors, one per sample.
            
        Example:
            >>> outputs = ctx.decrypt_batch(enc_output, num_samples=8)
            >>> for i, out in enumerate(outputs):
            ...     print(f"Sample {i}: {out}")
        """
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
        """Encrypt CNN input with pre-applied im2col for all conv layers.
        
        This applies im2col (unfold) on the plaintext image BEFORE encryption,
        allowing convolution to be computed as pure HE matrix multiplication.
        This is the secure approach - no decryption happens on the server.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W).
            conv_params: List of conv layer parameters, each containing:
                - kernel_size: Tuple (kH, kW)
                - stride: Tuple (sH, sW)  
                - padding: Tuple (pH, pW)
                - out_channels: Number of output channels
                
        Returns:
            EncryptedTensor with im2col-transformed and encrypted patches.
            Shape will be (num_patches, patch_features) for the first conv.
            
        Example:
            >>> conv_params = [
            ...     {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 8},
            ...     {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 16},
            ... ]
            >>> enc_input = ctx.encrypt_cnn_input(image, conv_params)
        """
        import torch.nn.functional as F
        
        self._ensure_initialized()
        
        # Ensure 4D: (1, C, H, W)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # Apply im2col for the first conv layer only
        # Subsequent layers operate on the transformed representation
        if not conv_params:
            return self.encrypt(image)
        
        first_conv = conv_params[0]
        kernel_size = first_conv['kernel_size']
        stride = first_conv.get('stride', (1, 1))
        padding = first_conv.get('padding', (0, 0))
        
        # Apply padding
        if padding != (0, 0):
            image = F.pad(image, (padding[1], padding[1], padding[0], padding[0]))
        
        # im2col: (N, C, H, W) -> (N, C*kH*kW, num_patches)
        patches = F.unfold(image.to(torch.float64), kernel_size, stride=stride)
        
        # Transpose: (N, C*kH*kW, num_patches) -> (N, num_patches, C*kH*kW)
        patches = patches.transpose(1, 2)
        
        # Squeeze batch: (num_patches, C*kH*kW)
        patches = patches.squeeze(0)
        
        # For HE: we need to encrypt each patch row as a separate operation
        # OR pack all patches into slots using channel-first packing
        # 
        # Approach: Flatten all patches into a single vector for encryption
        # The matmul will be done row-by-row via rotation
        num_patches, patch_features = patches.shape
        
        # Store layout info for later reconstruction
        # Flatten to 1D for encryption
        flat_patches = patches.flatten()
        
        enc_tensor = self.encrypt(flat_patches)
        
        # Store CNN-specific metadata
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
        """Create a context optimally configured for a specific model.

        Args:
            model: The PyTorch model to analyze.
            use_bsgs: Whether to use Baby-step Giant-step for rotations.
            activation_degree: Polynomial degree for activation approximation.
            use_square_activation: If True, activations use x² (1 level each).
            **kwargs: Additional arguments forwarded to __init__
                (e.g. enable_gpu, cnn_config, auto_bootstrap, ...).
        """
        # Pop config-level kwargs before forwarding the rest to __init__
        enable_bootstrap = kwargs.pop("enable_bootstrap", False)
        level_budget = kwargs.pop("level_budget", None)
        scale_bits = kwargs.pop("scale_bits", 50)
        security_level = kwargs.pop("security_level", "128_classic")

        config = InferenceConfig.for_model(
            model,
            activation_degree=activation_degree,
            use_square_activation=use_square_activation,
            enable_bootstrap=enable_bootstrap,
            level_budget=level_budget,
            scale_bits=scale_bits,
            security_level=security_level,
        )
        rotations = compute_rotations_for_model(model, use_bsgs=use_bsgs)
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
        """Create a context for a specific multiplicative depth.
        
        Args:
            depth: Number of multiplicative layers.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            CKKSInferenceContext configured for the depth.
        """
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
        """Save the context configuration to a file.
        
        This saves the Python configuration parameters, not the underlying
        CKKS cryptographic context (which would require OpenFHE serialization).
        
        Args:
            path: Path to save the context configuration.
        """
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
        """Load a context configuration from a file.
        
        This creates a new context with the saved configuration.
        The underlying CKKS context will be lazily initialized.
        
        Args:
            path: Path to the saved context configuration.
            
        Returns:
            A new CKKSInferenceContext with the loaded configuration.
        """
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
