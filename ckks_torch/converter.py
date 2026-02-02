"""
Model Converter - Convert PyTorch models to encrypted versions.

This module provides automatic conversion of trained PyTorch models
to their encrypted equivalents for CKKS inference.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch.nn as nn

from .context import CKKSInferenceContext
from .nn import (
    EncryptedModule,
    EncryptedLinear,
    EncryptedTTLinear,
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
        tt: If True, decompose Linear layers into TT format when possible.
    """
    
    def __init__(
        self,
        fold_batchnorm: bool = True,
        activation_degree: int = 4,
        activation_map: Optional[Dict[Type[nn.Module], Type[EncryptedModule]]] = None,
        use_square_activation: bool = False,
        optimize_cnn: bool = True,
        tt: bool = False,
    ):
        self.fold_batchnorm = fold_batchnorm
        self.activation_degree = activation_degree
        self.activation_map = activation_map or DEFAULT_ACTIVATION_MAP
        self.use_square_activation = use_square_activation
        self.optimize_cnn = optimize_cnn
        self.tt = tt


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
        for name, child in children:
            try:
                converted[name] = self._convert_module(child)
            except NotImplementedError:
                if isinstance(child, nn.Identity):
                    continue
                raise
        return EncryptedSequential(converted)
    
    def _convert_linear(self, module: nn.Linear) -> EncryptedModule:
        """Convert a Linear layer."""
        if self.options.tt:
            tt_layer = EncryptedTTLinear.from_torch(module)
            if tt_layer is not None:
                return tt_layer
            # Fall back to regular Linear if TT returns None (layer too small)
        return EncryptedLinear.from_torch(module)
    
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
        """
        if not hasattr(self, '_conv_params') or not self._conv_params:
            return None
        
        # Get the last conv's output channels
        last_conv = self._conv_params[-1]
        channels = last_conv['out_channels']
        
        # Linear input features = num_patches * channels
        in_features = linear.in_features
        
        if in_features % channels != 0:
            # Can't divide evenly, skip optimization
            return None
        
        num_patches = in_features // channels
        
        return {
            'num_patches': num_patches,
            'patch_features': channels,
        }
    
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
    
    # For other containers, replace children in-place
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
    tt: bool = False,
    enable_gpu: bool = True,
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
        tt: If True, decompose Linear layers into TT format when possible.
        
    Returns:
        Tuple of (encrypted_model, context).
        
    Example:
        >>> import torch.nn as nn
        >>> import ckks_torch
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
        >>> enc_model, ctx = ckks_torch.convert(model)
        >>> 
        >>> # Run encrypted inference
        >>> enc_input = ctx.encrypt(test_input)
        >>> enc_output = enc_model(enc_input)
        >>> output = ctx.decrypt(enc_output)
    """
    options = ConversionOptions(
        fold_batchnorm=fold_batchnorm,
        activation_degree=activation_degree,
        use_square_activation=use_square_activation,
        optimize_cnn=optimize_cnn,
        tt=tt,
    )
    
    converter = ModelConverter(options, input_shape=input_shape)
    enc_model = converter.convert(model, ctx)
    
    cnn_config = None
    if optimize_cnn and converter._is_cnn_model:
        cnn_config = _build_cnn_config(converter, input_shape)
    
    if ctx is None:
        ctx = CKKSInferenceContext.for_model(model, enable_gpu=enable_gpu, cnn_config=cnn_config)
    
    if cnn_config is not None and hasattr(converter, '_last_cnn_layout') and converter._last_cnn_layout:
        enc_model._cnn_layout = converter._last_cnn_layout
    
    return enc_model, ctx


def estimate_depth(model: nn.Module) -> int:
    """Estimate the multiplicative depth required for a model.
    
    Args:
        model: The PyTorch model to analyze.
        
    Returns:
        Estimated multiplicative depth.
    """
    depth = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            depth += 1  # matmul
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            depth += 2  # polynomial approximation (varies by degree)
        elif isinstance(module, nn.Sigmoid):
            depth += 2
    return max(1, depth)
