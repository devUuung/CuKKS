"""
Model Converter - Convert PyTorch models to encrypted versions.

This module provides automatic conversion of trained PyTorch models
to their encrypted equivalents for CKKS inference.
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from .context import CKKSInferenceContext
from .nn import (
    EncryptedModule,
    EncryptedLinear,
    EncryptedConv2d,
    EncryptedAvgPool2d,
    EncryptedMaxPool2d,
    EncryptedFlatten,
    EncryptedSequential,
    EncryptedSquare,
    EncryptedReLU,
    EncryptedGELU,
    EncryptedSiLU,
    EncryptedSigmoid,
    EncryptedTanh,
    EncryptedBatchNorm1d,
    EncryptedBatchNorm2d,
    EncryptedLayerNorm,
    EncryptedDropout,
    EncryptedApproxAttention,
)
from .nn.batchnorm import fold_batchnorm_into_linear, fold_batchnorm_into_conv
from .nn.block_diagonal import BlockDiagonalLinear
from .nn.block_diagonal_low_rank import BlockDiagLowRankLinear
from .nn.encrypted_block_diag_lr import EncryptedBlockDiagLowRank


# =============================================================================
# Default Activation Mapping
# =============================================================================

DEFAULT_ACTIVATION_MAP: Dict[Type[nn.Module], Type[EncryptedModule]] = {
    nn.ReLU: EncryptedReLU,
    nn.GELU: EncryptedGELU,
    nn.SiLU: EncryptedSiLU,
    nn.Sigmoid: EncryptedSigmoid,
    nn.Tanh: EncryptedTanh,
}


# =============================================================================
# Conversion Options
# =============================================================================

class ConversionOptions:
    """Options for model conversion.
    
    Attributes:
        fold_batchnorm: Whether to fold BatchNorm into preceding layers.
        activation_degree: Default polynomial degree for activation approximations.
        activation_map: Mapping from PyTorch activations to encrypted versions.
        use_square_activation: If True, replace all activations with x^2.
        optimize_cnn: If True, apply CNN-specific optimizations (Flatten absorption).
    """
    
    def __init__(
        self,
        fold_batchnorm: bool = True,
        activation_degree: int = 4,
        activation_map: Optional[Dict[Type[nn.Module], Type[EncryptedModule]]] = None,
        use_square_activation: bool = False,
        optimize_cnn: bool = True,
    ):
        self.fold_batchnorm = fold_batchnorm
        self.activation_degree = activation_degree
        self.activation_map = activation_map if activation_map is not None else DEFAULT_ACTIVATION_MAP.copy()
        self.use_square_activation = use_square_activation
        self.optimize_cnn = optimize_cnn


# =============================================================================
# Model Converter
# =============================================================================

class ModelConverter:
    """Converts PyTorch models to encrypted versions.
    
    This class handles the conversion of trained PyTorch models to their
    CKKS-compatible encrypted counterparts.
    
    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 128),
        ...     nn.ReLU(),
        ...     nn.Linear(128, 10)
        ... )
        >>> 
        >>> converter = ModelConverter()
        >>> enc_model = converter.convert(model)
    """
    
    def __init__(self, options: Optional[ConversionOptions] = None, input_shape: Optional[Tuple[int, ...]] = None):
        """Initialize the converter.
        
        Args:
            options: Conversion options. If None, uses defaults.
            input_shape: Optional input shape tuple for layout computation.
        """
        self.options = options or ConversionOptions()
        self.input_shape = input_shape
        
        # Layer converters: torch_type -> converter_function
        self._converters: Dict[Type[nn.Module], Callable] = {
            nn.Linear: self._convert_linear,
            BlockDiagonalLinear: self._convert_block_diagonal,
            BlockDiagLowRankLinear: self._convert_block_diag_low_rank,
            nn.Conv2d: self._convert_conv2d,
            nn.AvgPool2d: self._convert_avgpool2d,
            nn.Flatten: self._convert_flatten,
            nn.Sequential: self._convert_sequential,
            nn.BatchNorm1d: self._convert_batchnorm1d,
            nn.BatchNorm2d: self._convert_batchnorm2d,
            nn.Dropout: self._convert_dropout,
            nn.Dropout2d: self._convert_dropout,
            nn.MaxPool2d: self._convert_maxpool2d,
            nn.LayerNorm: self._convert_layernorm,
            nn.MultiheadAttention: self._convert_attention,
        }
        
        # Add activation converters
        for torch_type in self.options.activation_map:
            self._converters[torch_type] = self._convert_activation
        
        # CNN optimization state
        self._is_cnn_model = False
        self._last_cnn_layout: Optional[Dict] = None
        self._pending_flatten = False
    
    def convert(
        self,
        model: nn.Module,
        ctx: Optional[CKKSInferenceContext] = None,
    ) -> EncryptedModule:
        """Convert a PyTorch model to an encrypted version.
        
        Args:
            model: The PyTorch model to convert.
            ctx: Optional CKKS context. If None, one will be created
                 based on the model's requirements.
                 
        Returns:
            Converted EncryptedModule.
        """
        # Put model in eval mode
        model = model.eval()
        
        # Optionally fold BatchNorm layers
        if self.options.fold_batchnorm:
            model = self._fold_batchnorms(model)
        
        # Detect if this is a CNN model
        if self.options.optimize_cnn:
            self._is_cnn_model = self._detect_cnn(model)
            if self._is_cnn_model:
                self._analyze_cnn_structure(model)
        
        # Convert the model
        return self._convert_module(model)
    
    def _detect_cnn(self, model: nn.Module) -> bool:
        """Detect if model contains Conv2d layers before Linear layers."""
        found_conv = False
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                found_conv = True
            if isinstance(module, nn.Linear) and found_conv:
                return True  # Conv before Linear = CNN
        return False
    
    def _analyze_cnn_structure(self, model: nn.Module) -> None:
        """Analyze CNN structure to compute layouts for optimization."""
        # Extract conv/pool parameters to compute final CNN layout
        # This will be used for Flatten -> FC optimization
        self._conv_params = []
        self._pool_params = []
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self._conv_params.append({
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'out_channels': module.out_channels,
                    'in_channels': module.in_channels,
                })
            elif isinstance(module, nn.AvgPool2d):
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                stride = module.stride or kernel_size
                if isinstance(stride, int):
                    stride = (stride, stride)
                self._pool_params.append({
                    'kernel_size': kernel_size,
                    'stride': stride,
                })
    
    def _convert_module(self, module: nn.Module) -> EncryptedModule:
        """Convert a single module."""
        module_type = type(module)
        
        # Check for exact type match
        if module_type in self._converters:
            return self._converters[module_type](module)
        
        # Check for subclass match
        for parent_type, converter in self._converters.items():
            if isinstance(module, parent_type):
                return converter(module)
        
        # Check if it's a container with children
        children = list(module.named_children())
        if children:
            return self._convert_container(module, children)
        
        raise NotImplementedError(
            f"No converter found for {module_type.__name__}. "
            f"Supported types: {list(self._converters.keys())}"
        )
    
    def _convert_container(
        self,
        module: nn.Module,
        children: List[Tuple[str, nn.Module]],
    ) -> EncryptedSequential:
        """Convert a container module with children."""
        converted = OrderedDict()
        skip_next = False
        
        for i, (name, child) in enumerate(children):
            if skip_next:
                skip_next = False
                continue
            
            # CNN optimization: Flatten + Linear -> optimized Linear
            if (self.options.optimize_cnn and 
                self._is_cnn_model and
                isinstance(child, nn.Flatten) and 
                i + 1 < len(children)):
                
                _, next_child = children[i + 1]
                if isinstance(next_child, nn.Linear):
                    cnn_layout = self._compute_cnn_layout_from_linear(next_child)
                    
                    if cnn_layout is not None:
                        converted[name] = EncryptedFlatten._with_absorbed_permutation(
                            child.start_dim, 
                            child.end_dim
                        )
                        next_name = children[i + 1][0]
                        converted[next_name] = EncryptedLinear.from_torch_cnn(
                            next_child, 
                            cnn_layout
                        )
                        skip_next = True
                        continue
            
            try:
                converted[name] = self._convert_module(child)
            except NotImplementedError:
                if isinstance(child, nn.Identity):
                    continue
                raise
        return EncryptedSequential(converted)
    
    def _convert_linear(self, module: nn.Linear) -> EncryptedModule:
        """Convert a Linear layer."""
        return EncryptedLinear.from_torch(module)

    def _convert_block_diagonal(self, module: BlockDiagonalLinear) -> EncryptedModule:
        """Convert a BlockDiagonalLinear by expanding to dense weight."""
        return EncryptedLinear.from_torch(module.to_linear())

    def _convert_block_diag_low_rank(self, module: BlockDiagLowRankLinear) -> EncryptedModule:
        """Convert a BlockDiagLowRankLinear to encrypted BD + low-rank module."""
        return EncryptedBlockDiagLowRank.from_module(module)

    def _convert_conv2d(self, module: nn.Conv2d) -> EncryptedConv2d:
        """Convert a Conv2d layer."""
        return EncryptedConv2d.from_torch(module)
    
    def _convert_avgpool2d(self, module: nn.AvgPool2d) -> EncryptedAvgPool2d:
        """Convert an AvgPool2d layer."""
        return EncryptedAvgPool2d.from_torch(module)
    
    def _convert_flatten(self, module: nn.Flatten) -> EncryptedFlatten:
        """Convert a Flatten layer."""
        return EncryptedFlatten(module.start_dim, module.end_dim)
    
    def _convert_sequential(self, module: nn.Sequential) -> EncryptedSequential:
        """Convert a Sequential container with CNN optimizations."""
        converted = OrderedDict()
        children = list(module.named_children())
        skip_next = False
        
        for i, (name, child) in enumerate(children):
            if skip_next:
                skip_next = False
                continue
            
            # CNN optimization: Flatten + Linear -> optimized Linear (absorb permutation)
            if (self.options.optimize_cnn and 
                self._is_cnn_model and
                isinstance(child, nn.Flatten) and 
                i + 1 < len(children)):
                
                _, next_child = children[i + 1]
                if isinstance(next_child, nn.Linear):
                    # Compute CNN layout from the Linear layer's input size
                    cnn_layout = self._compute_cnn_layout_from_linear(next_child)
                    
                    if cnn_layout is not None:
                        # Create Flatten with internal flag (no-op)
                        converted[name] = EncryptedFlatten._with_absorbed_permutation(
                            child.start_dim, 
                            child.end_dim
                        )
                        # Create Linear with permuted weights
                        next_name = children[i + 1][0]
                        converted[next_name] = EncryptedLinear.from_torch_cnn(
                            next_child, 
                            cnn_layout
                        )
                        skip_next = True
                        continue
            
            try:
                converted[name] = self._convert_module(child)
            except NotImplementedError:
                if isinstance(child, nn.Identity):
                    continue
                raise
                
        return EncryptedSequential(converted)
    
    def _compute_cnn_layout_from_linear(self, linear: nn.Linear) -> Optional[Dict]:
        """Compute the CNN layout from Linear layer's input features.
        
        Uses the last conv's output channels and Linear's input features
        to infer num_patches = in_features / channels.
        
        If 2x2/stride 2 pooling is used, computes sparse layout for rotation-based
        pooling optimization.
        """
        if not hasattr(self, '_conv_params') or not self._conv_params:
            return None
        
        last_conv = self._conv_params[-1]
        channels = last_conv['out_channels']
        in_features = linear.in_features
        
        if in_features % channels != 0:
            return None
        
        num_patches = in_features // channels
        
        layout: Dict[str, Any] = {
            'num_patches': num_patches,
            'patch_features': channels,
        }
        
        if hasattr(self, '_pool_params') and self._pool_params:
            last_pool = self._pool_params[-1]
            kh, kw = last_pool['kernel_size']
            sh, sw = last_pool['stride']
            
            if kh == 2 and kw == 2 and sh == 2 and sw == 2:
                pre_pool_h, pre_pool_w = self._compute_pre_pool_dimensions()
                if pre_pool_h is not None and pre_pool_w is not None:
                    out_h = pre_pool_h // 2
                    out_w = pre_pool_w // 2
                    
                    sparse_positions = []
                    for out_y in range(out_h):
                        for out_x in range(out_w):
                            in_y = 2 * out_y
                            in_x = 2 * out_x
                            in_patch_idx = in_y * pre_pool_w + in_x
                            for c in range(channels):
                                sparse_positions.append(in_patch_idx * channels + c)
                    
                    layout['sparse'] = True
                    layout['sparse_positions'] = sparse_positions
                    layout['total_slots'] = pre_pool_h * pre_pool_w * channels
                    layout['pre_pool_h'] = pre_pool_h
                    layout['pre_pool_w'] = pre_pool_w
        
        return layout
    
    def _compute_pre_pool_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        """Compute spatial dimensions before the last pooling layer."""
        if not hasattr(self, '_conv_params') or not self._conv_params:
            return None, None
        
        h, w = 8, 8
        if self.input_shape is not None:
            shape = tuple(self.input_shape)
            if len(shape) >= 2:
                h, w = int(shape[-2]), int(shape[-1])
        
        pool_idx = 0
        num_pools = len(self._pool_params) if hasattr(self, '_pool_params') else 0
        
        for conv in self._conv_params:
            kernel_size = conv['kernel_size']
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            stride = conv['stride']
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = conv['padding']
            if isinstance(padding, int):
                padding = (padding, padding)
            
            kh, kw = kernel_size
            sh, sw = stride
            ph, pw = padding
            
            h = (h + 2 * ph - kh) // sh + 1
            w = (w + 2 * pw - kw) // sw + 1
            
            if pool_idx < num_pools - 1:
                pool = self._pool_params[pool_idx]
                psh, psw = pool['stride']
                h = h // psh
                w = w // psw
                pool_idx += 1
        
        return h, w
    
    def _compute_cnn_layout_before_flatten(self) -> Optional[Dict]:
        """Compute the CNN layout (num_patches, channels) before Flatten layer.
        
        This uses the accumulated conv/pool parameters to determine the
        spatial dimensions at the Flatten point.
        """
        if not hasattr(self, '_conv_params') or not self._conv_params:
            return None
        
        h, w = 8, 8
        if self.input_shape is not None:
            shape = tuple(self.input_shape)
            if len(shape) >= 2:
                h, w = int(shape[-2]), int(shape[-1])
        
        # Apply each conv and pool to track spatial dimensions
        channels = 1  # Start with 1 channel
        pool_idx = 0
        
        for conv in self._conv_params:
            kernel_size = conv['kernel_size']
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            stride = conv['stride']
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = conv['padding']
            if isinstance(padding, int):
                padding = (padding, padding)

            kh, kw = kernel_size
            sh, sw = stride
            ph, pw = padding

            h = (h + 2 * ph - kh) // sh + 1
            w = (w + 2 * pw - kw) // sw + 1
            channels = conv['out_channels']

            if hasattr(self, '_pool_params') and pool_idx < len(self._pool_params):
                pool = self._pool_params[pool_idx]
                sh, sw = pool['stride']
                h = h // sh
                w = w // sw
                pool_idx += 1
        
        num_patches = h * w
        return {
            'num_patches': num_patches,
            'patch_features': channels,
        }
    
    def _convert_activation(self, module: nn.Module) -> EncryptedModule:
        """Convert an activation function."""
        if self.options.use_square_activation:
            return EncryptedSquare()
        
        module_type = type(module)
        if module_type in self.options.activation_map:
            enc_type = self.options.activation_map[module_type]
            try:
                return enc_type(degree=self.options.activation_degree)  # type: ignore[call-arg]
            except TypeError:
                return enc_type()
        
        raise NotImplementedError(f"No activation converter for {module_type.__name__}")
    
    def _convert_batchnorm1d(self, module: nn.BatchNorm1d) -> EncryptedBatchNorm1d:
        """Convert a BatchNorm1d layer."""
        return EncryptedBatchNorm1d.from_torch(module)
    
    def _convert_batchnorm2d(self, module: nn.BatchNorm2d) -> EncryptedBatchNorm2d:
        """Convert a BatchNorm2d layer."""
        return EncryptedBatchNorm2d.from_torch(module)
    
    def _convert_dropout(self, module: nn.Module) -> EncryptedDropout:
        warnings.warn(
            "Dropout layers are ignored during inference (no-op)",
            UserWarning,
            stacklevel=3,
        )
        p = getattr(module, 'p', 0.5)
        return EncryptedDropout(p=p)
    
    def _convert_layernorm(self, module: nn.LayerNorm) -> EncryptedLayerNorm:
        return EncryptedLayerNorm.from_torch(module)
    
    def _convert_attention(self, module: nn.MultiheadAttention) -> EncryptedApproxAttention:
        return EncryptedApproxAttention.from_torch(module)
    
    def _convert_maxpool2d(self, module: nn.MaxPool2d) -> EncryptedMaxPool2d:
        """Convert a MaxPool2d layer."""
        return EncryptedMaxPool2d.from_torch(module)
    
    def _fold_batchnorms(self, model: nn.Module) -> nn.Module:
        """Fold BatchNorm layers into preceding Linear/Conv layers."""
        model = _fold_bn_recursive(model)
        return model


def _fold_bn_recursive(module: nn.Module) -> nn.Module:
    """Recursively fold BatchNorm into preceding layers."""
    # Check if this is a Sequential-like container
    children = list(module.named_children())
    if not children:
        return module
    
    new_children = []
    skip_next = False
    
    for i, (name, child) in enumerate(children):
        if skip_next:
            skip_next = False
            continue
        
        # Check if next layer is BatchNorm
        if i + 1 < len(children):
            _, next_child = children[i + 1]
            
            if isinstance(child, nn.Linear) and isinstance(next_child, nn.BatchNorm1d):
                folded = fold_batchnorm_into_linear(child, next_child)
                new_children.append((name, folded))
                skip_next = True
                continue
            
            if isinstance(child, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d):
                folded = fold_batchnorm_into_conv(child, next_child)
                new_children.append((name, folded))
                skip_next = True
                continue
        
        # Recursively process child
        new_children.append((name, _fold_bn_recursive(child)))
    
    # Rebuild the module
    if isinstance(module, nn.Sequential):
        return nn.Sequential(OrderedDict(new_children))
    
    # For other containers, replace children in-place and remove folded BN layers
    new_names = {name for name, _ in new_children}
    for name, _ in children:
        if name not in new_names:
            delattr(module, name)
    for name, new_child in new_children:
        setattr(module, name, new_child)
    
    return module


# =============================================================================
# Convenience Functions
# =============================================================================

def _build_cnn_config(
    converter: ModelConverter,
    input_shape: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, int]]:
    """Build cnn_config dict from converter's collected CNN parameters.
    
    Args:
        converter: The ModelConverter instance with CNN analysis.
        input_shape: Optional input shape tuple (channels, height, width).
        
    Returns:
        Dictionary with keys: image_height, image_width, channels, pool_size, pool_stride.
        Returns None if CNN parameters are insufficient.
    """
    if not hasattr(converter, '_conv_params') or not converter._conv_params:
        return None
    
    last_conv = converter._conv_params[-1]
    channels = last_conv['out_channels']
    
    if input_shape is not None and len(input_shape) >= 2:
        image_height = input_shape[-2]
        image_width = input_shape[-1]
    else:
        image_height = 8
        image_width = 8
    
    pool_size = 2
    pool_stride = 2
    if hasattr(converter, '_pool_params') and converter._pool_params:
        last_pool = converter._pool_params[-1]
        pool_size = last_pool['kernel_size'][0] if isinstance(last_pool['kernel_size'], tuple) else last_pool['kernel_size']
        pool_stride = last_pool['stride'][0] if isinstance(last_pool['stride'], tuple) else last_pool['stride']
    
    return {
        'image_height': image_height,
        'image_width': image_width,
        'channels': channels,
        'pool_size': pool_size,
        'pool_stride': pool_stride,
    }


def convert(
    model: nn.Module,
    ctx: Optional[CKKSInferenceContext] = None,
    *,
    input_shape: Optional[Tuple[int, ...]] = None,
    fold_batchnorm: bool = True,
    activation_degree: int = 4,
    use_square_activation: bool = False,
    optimize_cnn: bool = True,
    enable_gpu: bool = True,
    enable_bootstrap: Optional[bool] = None,
    auto_bootstrap: Optional[bool] = None,
    bootstrap_threshold: int = 8,
    pre_encode: bool = False,
) -> Tuple[EncryptedModule, CKKSInferenceContext]:
    """Convert a PyTorch model to an encrypted version.
    
    This is the main entry point for model conversion.
    
    Args:
        model: The PyTorch model to convert.
        ctx: Optional CKKS context. If None, one will be created.
        input_shape: Optional input shape tuple (e.g., (1, 28, 28) for MNIST images).
                     Used for layout computation in CNN models.
        fold_batchnorm: Whether to fold BatchNorm into preceding layers.
        activation_degree: Default polynomial degree for activations.
        use_square_activation: If True, replace all activations with x^2.
        optimize_cnn: If True, apply CNN optimizations (Flatten absorption into FC).
        enable_bootstrap: Enable bootstrapping. If None, auto-detected from depth.
        auto_bootstrap: Call maybe_bootstrap() between layers. If None, follows
            enable_bootstrap.
        bootstrap_threshold: Depth at which auto-bootstrap triggers (default 8).
        pre_encode: If True, run a warmup forward pass at conversion time to
            pre-encode all weight plaintexts into the GPU cache. This eliminates
            the cold-start latency on the first real inference at the cost of
            additional GPU memory and a one-time computation during conversion.
            Requires ``input_shape`` for CNN models.
        
    Returns:
        Tuple of (encrypted_model, context).
        
    Example:
        >>> import torch.nn as nn
        >>> import cukks
        >>> 
        >>> # Train your model
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 128),
        ...     nn.ReLU(),
        ...     nn.Linear(128, 10)
        ... )
        >>> train(model, data)
        >>> 
        >>> # Convert to encrypted
        >>> enc_model, ctx = cukks.convert(model)
        >>> 
        >>> # Run encrypted inference
        >>> enc_input = ctx.encrypt(test_input)
        >>> enc_output = enc_model(enc_input)
        >>> output = ctx.decrypt(enc_output)
    """
    from .context import _estimate_model_depth

    options = ConversionOptions(
        fold_batchnorm=fold_batchnorm,
        activation_degree=activation_degree,
        use_square_activation=use_square_activation,
        optimize_cnn=optimize_cnn,
    )
    
    converter = ModelConverter(options, input_shape=input_shape)
    enc_model = converter.convert(model, ctx)
    
    cnn_config = None
    if optimize_cnn and converter._is_cnn_model:
        cnn_config = _build_cnn_config(converter, input_shape)

    # Auto-detect whether bootstrapping is needed.
    # The C++ backend gives 10 levels after bootstrap (levelsAfterBootstrap=10).
    # Without bootstrap, 32768 poly_mod_degree supports ~16 effective levels.
    # If estimated depth > 14, we need bootstrapping.
    _AUTO_BOOTSTRAP_DEPTH_THRESHOLD = 14
    if enable_bootstrap is None:
        est_depth = _estimate_model_depth(
            model,
            activation_degree=activation_degree,
            use_square_activation=use_square_activation,
        )
        enable_bootstrap = est_depth > _AUTO_BOOTSTRAP_DEPTH_THRESHOLD
    if auto_bootstrap is None:
        auto_bootstrap = enable_bootstrap

    if ctx is None:
        ctx = CKKSInferenceContext.for_model(
            model,
            activation_degree=activation_degree,
            use_square_activation=use_square_activation,
            enable_gpu=enable_gpu,
            cnn_config=cnn_config,
            auto_bootstrap=auto_bootstrap,
            bootstrap_threshold=bootstrap_threshold,
            enable_bootstrap=enable_bootstrap,
        )
    
    if cnn_config is not None and hasattr(converter, '_last_cnn_layout') and converter._last_cnn_layout:
        enc_model._cnn_layout = converter._last_cnn_layout
    
    if pre_encode:
        _warmup_cache(enc_model, ctx, converter, input_shape)
    
    return enc_model, ctx


def _warmup_cache(
    enc_model: EncryptedModule,
    ctx: CKKSInferenceContext,
    converter: ModelConverter,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    """Run a single forward pass to populate the GPU plaintext cache.

    This encrypts a zero-filled dummy input and runs it through the
    encrypted model so that every ``matmul_bsgs`` call encodes its weight
    diagonals into the C++ GPU cache.  Subsequent inference calls will
    hit the warm cache, eliminating the cold-start encoding latency.

    Args:
        enc_model: The converted encrypted model.
        ctx: The CKKS inference context (must be fully initialised).
        converter: The ``ModelConverter`` used during conversion.
        input_shape: Input shape for CNN models. For MLP models the
            first layer's ``in_features`` is used to infer the size.
    """
    logger = logging.getLogger(__name__)
    logger.info("pre_encode: running warmup forward pass to populate GPU plaintext cache")

    ctx.set_plain_cache_limit(0)

    is_cnn = getattr(converter, '_is_cnn_model', False)

    if is_cnn:
        if input_shape is None:
            raise ValueError(
                "pre_encode=True requires input_shape for CNN models "
                "(e.g. input_shape=(1, 28, 28) for MNIST)."
            )
        conv_params = getattr(converter, '_conv_params', None)
        if not conv_params:
            raise ValueError(
                "pre_encode=True: CNN model detected but no conv parameters "
                "were recorded during conversion."
            )
        if len(input_shape) == 4:
            dummy = torch.zeros(input_shape, dtype=torch.float64)
        elif len(input_shape) == 3:
            dummy = torch.zeros((1,) + tuple(input_shape), dtype=torch.float64)
        elif len(input_shape) == 2:
            dummy = torch.zeros((1, 1) + tuple(input_shape), dtype=torch.float64)
        else:
            raise ValueError(
                f"pre_encode: cannot interpret input_shape={input_shape} for CNN model"
            )

        first_conv = conv_params[0]
        kernel_size = first_conv['kernel_size']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        stride = first_conv.get('stride', (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = first_conv.get('padding', (0, 0))
        if isinstance(padding, int):
            padding = (padding, padding)

        enc_input = ctx.encrypt_cnn_input(
            dummy,
            [{'kernel_size': kernel_size, 'stride': stride, 'padding': padding}],
        )
    else:
        first_dim = _infer_input_dim(enc_model)
        if first_dim is None and input_shape is not None:
            import math as _math
            first_dim = int(_math.prod(input_shape))
        if first_dim is None:
            raise ValueError(
                "pre_encode=True: cannot infer input dimension. "
                "Provide input_shape or ensure the model starts with "
                "EncryptedLinear / EncryptedConv2d."
            )
        dummy = torch.zeros(first_dim, dtype=torch.float64)
        enc_input = ctx.encrypt(dummy)

    _ = enc_model(enc_input)

    ctx.pin_plain_cache()
    ctx.set_plain_cache_limit(2048)

    cache_info = ctx.plain_cache_info()
    logger.info(
        "pre_encode: warmup complete, %d plaintexts pinned in GPU cache",
        cache_info.get("pinned", 0),
    )


def _infer_input_dim(enc_model: EncryptedModule) -> Optional[int]:
    """Walk the model tree and return the first layer's input dimension."""
    for module in enc_model.modules():
        if isinstance(module, EncryptedLinear):
            return module.in_features
        if isinstance(module, EncryptedConv2d):
            return None
    return None


def warm_cache(
    enc_model: EncryptedModule,
    ctx: CKKSInferenceContext,
    input_data: torch.Tensor,
) -> None:
    """Manually warm the GPU plaintext cache by running one forward pass.

    Use this when you did not set ``pre_encode=True`` during conversion
    but still want to eliminate cold-start latency before serving.

    Args:
        enc_model: The converted encrypted model.
        ctx: The CKKS inference context.
        input_data: A representative input tensor (will be encrypted
            and run through the model; output is discarded).

    Example:
        >>> enc_model, ctx = cukks.convert(model)
        >>> dummy = torch.zeros(784)
        >>> cukks.warm_cache(enc_model, ctx, dummy)
        >>> # First real inference is now fast
        >>> enc_out = enc_model(ctx.encrypt(real_input))
    """
    enc_input = ctx.encrypt(input_data)
    _ = enc_model(enc_input)


def estimate_depth(model: nn.Module, activation_degree: int = 4) -> int:
    """Estimate the multiplicative depth required for a model.
    
    Args:
        model: The PyTorch model to analyze.
        activation_degree: Polynomial degree used for activation approximation.
            Default is 4, matching ConversionOptions default.
        
    Returns:
        Estimated multiplicative depth.
    """
    import math
    
    # Polynomial of degree d requires ceil(log2(d+1)) levels
    # (Paterson-Stockmeyer or baby-step/giant-step evaluation)
    poly_depth = max(1, math.ceil(math.log2(activation_degree + 1)))
    
    depth = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            depth += 1  # matmul
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            depth += poly_depth  # polynomial approximation
    return max(1, depth)
