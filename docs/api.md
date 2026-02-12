# API Reference

This document provides complete API documentation for CuKKS.

## Table of Contents

- [Core Classes](#core-classes)
  - [CKKSInferenceContext](#ckksinferencecontext)
  - [InferenceConfig](#inferenceconfig)
  - [EncryptedTensor](#encryptedtensor)
- [Model Conversion](#model-conversion)
  - [convert()](#convert)
  - [ModelConverter](#modelconverter)
  - [ConversionOptions](#conversionoptions)
  - [estimate_depth()](#estimate_depth)
- [Neural Network Modules](#neural-network-modules)
  - [EncryptedModule](#encryptedmodule)
  - [EncryptedLinear](#encryptedlinear)
  - [EncryptedConv2d](#encryptedconv2d)
  - [Activation Functions](#activation-functions)
  - [EncryptedSequential](#encryptedsequential)
  - [Other Layers](#other-layers)
    - [EncryptedFlatten](#encryptedflatten)
    - [EncryptedAvgPool2d](#encryptedavgpool2d)
    - [EncryptedMaxPool2d](#encryptedmaxpool2d)
    - [EncryptedBatchNorm1d / EncryptedBatchNorm2d](#encryptedbatchnorm1d--encryptedbatchnorm2d)
    - [EncryptedLayerNorm](#encryptedlayernorm)
    - [EncryptedDropout](#encrypteddropout)
    - [EncryptedResidualBlock](#encryptedresidualblock)
    - [EncryptedApproxAttention](#encryptedapproxattention)
- [Utilities](#utilities)
  - [SlotPacker](#slotpacker)

---

## Core Classes

### CKKSInferenceContext

High-level context for encrypted inference. Manages encryption parameters and provides methods for encrypting/decrypting tensors.

```python
from cukks import CKKSInferenceContext, InferenceConfig
```

#### Constructor

```python
CKKSInferenceContext(
    config: Optional[InferenceConfig] = None,
    *,
    device: Optional[str] = None,
    rotations: Optional[List[int]] = None,
    use_bsgs: bool = True,
    max_rotation_dim: Optional[int] = None,
    auto_bootstrap: bool = False,
    bootstrap_threshold: int = 2,
    enable_gpu: bool = True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `config` | `InferenceConfig` | `None` | Encryption configuration. If None, uses defaults. |
| `device` | `str` | `"cuda"` or `"cpu"` | Device for computation. |
| `rotations` | `List[int]` | Auto | Rotation keys to generate. Auto-computed if None. |
| `use_bsgs` | `bool` | `True` | Use Baby-step Giant-step optimization for rotations. |
| `max_rotation_dim` | `int` | `None` | Maximum dimension for rotation keys. |
| `auto_bootstrap` | `bool` | `False` | Enable automatic bootstrapping when depth is exhausted. |
| `bootstrap_threshold` | `int` | `2` | Depth at which to trigger auto-bootstrap. |
| `enable_gpu` | `bool` | `True` | Enable GPU acceleration for HE operations. |

#### Class Methods

##### `for_model()`

Create a context optimized for a specific model.

```python
ctx = CKKSInferenceContext.for_model(
    model: torch.nn.Module,
    use_bsgs: bool = True,
    **kwargs
) -> CKKSInferenceContext
```

##### `for_depth()`

Create a context for a specific multiplicative depth.

```python
ctx = CKKSInferenceContext.for_depth(
    depth: int,
    **kwargs
) -> CKKSInferenceContext
```

##### `load_context()`

Load a saved context configuration.

```python
ctx = CKKSInferenceContext.load_context(path: str) -> CKKSInferenceContext
```

#### Instance Methods

##### `encrypt_cnn_input()`

Encrypt CNN input with pre-applied im2col for true HE convolution.

```python
enc_tensor = ctx.encrypt_cnn_input(
    x: torch.Tensor,
    conv_params: List[Dict]
) -> EncryptedTensor
```

**Parameters:**
- `x`: Input tensor of shape (B, C, H, W).
- `conv_params`: List of conv layer parameters. Each dict contains:
  - `kernel_size`: Tuple[int, int]
  - `stride`: Tuple[int, int]
  - `padding`: Tuple[int, int]
  - `out_channels`: int

**Returns:** `EncryptedTensor` with CNN layout metadata.

**Example:**

```python
conv_params = [{'kernel_size': (3,3), 'stride': (1,1), 'padding': (1,1), 'out_channels': 8}]
enc_x = ctx.encrypt_cnn_input(image, conv_params)
# enc_x._cnn_layout = {'num_patches': 64, 'patch_features': 9, ...}
```

##### `encrypt()`

Encrypt a PyTorch tensor.

```python
enc_tensor = ctx.encrypt(tensor: torch.Tensor) -> EncryptedTensor
```

**Parameters:**
- `tensor`: Input tensor to encrypt. Will be flattened.

**Returns:** `EncryptedTensor` wrapper around the ciphertext.

**Raises:** `ValueError` if tensor size exceeds available slots.

##### `decrypt()`

Decrypt an encrypted tensor.

```python
tensor = ctx.decrypt(
    encrypted: EncryptedTensor,
    shape: Optional[Sequence[int]] = None
) -> torch.Tensor
```

**Parameters:**
- `encrypted`: The encrypted tensor to decrypt.
- `shape`: Optional shape for the output tensor.

**Returns:** Decrypted PyTorch tensor.

##### `encrypt_batch()`

Encrypt multiple samples into a single ciphertext using slot packing.

```python
enc_batch = ctx.encrypt_batch(
    samples: List[torch.Tensor],
    slots_per_sample: Optional[int] = None
) -> EncryptedTensor
```

**Parameters:**
- `samples`: List of tensors to encrypt.
- `slots_per_sample`: Slots per sample. If None, uses first sample's size.

**Returns:** `EncryptedTensor` containing all packed samples.

##### `decrypt_batch()`

Decrypt a batched ciphertext into individual samples.

```python
samples = ctx.decrypt_batch(
    encrypted: EncryptedTensor,
    num_samples: Optional[int] = None,
    sample_shape: Optional[Sequence[int]] = None
) -> List[torch.Tensor]
```

##### `save_context()`

Save the context configuration.

```python
ctx.save_context(path: str) -> None
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_slots` | `int` | Number of available plaintext slots |
| `config` | `InferenceConfig` | The configuration object |
| `device` | `str` | Device for computation |
| `auto_bootstrap` | `bool` | Whether auto-bootstrap is enabled |
| `bootstrap_threshold` | `int` | Depth threshold for auto-bootstrap |
| `backend` | `CKKSContext` | Underlying CKKS context (lazily initialized) |

---

### InferenceConfig

Configuration dataclass for CKKS encryption parameters.

```python
from cukks import InferenceConfig
```

#### Constructor

```python
InferenceConfig(
    poly_mod_degree: int = 16384,
    scale_bits: int = 40,
    security_level: str = "128_classic",
    mult_depth: int = 4,
    enable_bootstrap: bool = False,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `poly_mod_degree` | `int` | `16384` | Ring dimension. Common values: 8192, 16384, 32768. |
| `scale_bits` | `int` | `40` | Precision bits for encoding. |
| `security_level` | `str` | `"128_classic"` | Security level: "128_classic", "192_classic", "256_classic". |
| `mult_depth` | `int` | `4` | Multiplicative depth for the circuit. |
| `enable_bootstrap` | `bool` | `False` | Enable bootstrapping for deep networks. |

#### Class Methods

##### `for_depth()`

Create config optimized for a specific depth.

```python
config = InferenceConfig.for_depth(depth: int, **kwargs) -> InferenceConfig
```

##### `for_model()`

Analyze a model and create optimal config.

```python
config = InferenceConfig.for_model(model: torch.nn.Module, **kwargs) -> InferenceConfig
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_slots` | `int` | Number of plaintext slots (poly_mod_degree // 2) |
| `coeff_mod_bits` | `Tuple[int, ...]` | Coefficient modulus bit sizes |

---

### EncryptedTensor

A tensor wrapper for CKKS-encrypted data with tensor-like operations.

```python
from cukks import EncryptedTensor
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `Tuple[int, ...]` | Logical shape of the tensor |
| `size` | `int` | Total number of elements |
| `ndim` | `int` | Number of dimensions |
| `context` | `CKKSInferenceContext` | Associated context |
| `depth` | `int` | Current multiplicative depth consumed |
| `level` | `int` | Remaining multiplicative levels |
| `scale` | `float` | Current scale factor |
| `metadata` | `dict` | Ciphertext metadata |

#### Arithmetic Operations

```python
# Addition (cipher + cipher or cipher + plain)
result = enc_tensor.add(other: Union[EncryptedTensor, torch.Tensor, float]) -> EncryptedTensor
result = enc_tensor + other  # Operator form

# Subtraction
result = enc_tensor.sub(other) -> EncryptedTensor
result = enc_tensor - other

# Multiplication
result = enc_tensor.mul(other) -> EncryptedTensor
result = enc_tensor * other

# Negation
result = enc_tensor.neg() -> EncryptedTensor
result = -enc_tensor

# Division by scalar
result = enc_tensor.div(divisor: float) -> EncryptedTensor

# Square
result = enc_tensor.square() -> EncryptedTensor
```

#### CKKS-specific Operations

```python
# Rescale to manage scale growth
result = enc_tensor.rescale() -> EncryptedTensor

# Rotate slots
result = enc_tensor.rotate(steps: int) -> EncryptedTensor

# Sum all slots
result = enc_tensor.sum_slots() -> EncryptedTensor

# Complex conjugation
result = enc_tensor.conjugate() -> EncryptedTensor

# Bootstrap to refresh ciphertext
result = enc_tensor.bootstrap() -> EncryptedTensor

# Auto-bootstrap if threshold reached
result = enc_tensor.maybe_bootstrap(context) -> EncryptedTensor
```

#### Matrix Operations

```python
# Matrix-vector multiplication
result = enc_tensor.matmul(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> EncryptedTensor

# Polynomial evaluation
result = enc_tensor.poly_eval(coeffs: Sequence[float]) -> EncryptedTensor
```

#### Shape Operations

```python
# Reshape (logical only)
result = enc_tensor.view(*shape) -> EncryptedTensor
result = enc_tensor.reshape(shape) -> EncryptedTensor

# Flatten
result = enc_tensor.flatten() -> EncryptedTensor

# Clone
result = enc_tensor.clone() -> EncryptedTensor
```

#### Decryption

```python
# Decrypt using associated context
tensor = enc_tensor.decrypt(shape: Optional[Sequence[int]] = None) -> torch.Tensor
```

#### Serialization

```python
# Save encrypted tensor
enc_tensor.save(path: str) -> None

# Load encrypted tensor
enc_tensor = EncryptedTensor.load(path: str, context: CKKSInferenceContext) -> EncryptedTensor
```

---

## Model Conversion

### convert()

Main entry point for model conversion. Converts a PyTorch model to an encrypted version.

```python
from cukks import convert

enc_model, ctx = convert(
    model: torch.nn.Module,
    ctx: Optional[CKKSInferenceContext] = None,
    *,
    fold_batchnorm: bool = True,
    activation_degree: int = 4,
    use_square_activation: bool = False,
    optimize_cnn: bool = True,
    tt: bool = False,
    enable_gpu: bool = True,
) -> Tuple[EncryptedModule, CKKSInferenceContext]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `torch.nn.Module` | Required | PyTorch model to convert |
| `ctx` | `CKKSInferenceContext` | `None` | Context to use. Created automatically if None. |
| `fold_batchnorm` | `bool` | `True` | Fold BatchNorm into preceding layers |
| `activation_degree` | `int` | `4` | Polynomial degree for activation approximations |
| `use_square_activation` | `bool` | `False` | Replace all activations with x² |
| `optimize_cnn` | `bool` | `True` | Auto-detect CNN and apply optimizations (Flatten absorption, Pool rotation) |
| `enable_gpu` | `bool` | `True` | Enable GPU acceleration for HE operations |

**Returns:** Tuple of (encrypted_model, context)

**Example:**

```python
import torch.nn as nn
import cukks

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

enc_model, ctx = cukks.convert(model)
```

---

### ModelConverter

Class for fine-grained control over model conversion.

```python
from cukks.converter import ModelConverter, ConversionOptions
```

#### Constructor

```python
converter = ModelConverter(options: Optional[ConversionOptions] = None)
```

#### Methods

##### `convert()`

```python
enc_model = converter.convert(
    model: torch.nn.Module,
    ctx: Optional[CKKSInferenceContext] = None
) -> EncryptedModule
```

---

### ConversionOptions

Configuration for model conversion.

```python
from cukks.converter import ConversionOptions

options = ConversionOptions(
    fold_batchnorm: bool = True,
    activation_degree: int = 4,
    activation_map: Optional[Dict[Type, Type]] = None,
    use_square_activation: bool = False,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `fold_batchnorm` | `bool` | `True` | Fold BatchNorm into preceding layers |
| `activation_degree` | `int` | `4` | Default polynomial degree for activations |
| `activation_map` | `Dict` | `None` | Custom mapping from PyTorch to encrypted activations |
| `use_square_activation` | `bool` | `False` | Replace all activations with x² |

---

### estimate_depth()

Estimate multiplicative depth required for a model.

```python
from cukks import estimate_depth

depth = estimate_depth(model: torch.nn.Module) -> int
```

---

## Neural Network Modules

### EncryptedModule

Base class for all encrypted neural network modules.

```python
from cukks.nn import EncryptedModule
```

#### Abstract Methods

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    """Forward pass on encrypted input."""
    pass
```

#### Methods

```python
# Make module callable
result = module(x)  # Equivalent to module.forward(x)

# Iterate over modules
for mod in module.modules():
    ...

for name, mod in module.named_modules():
    ...

# Iterate over children
for child in module.children():
    ...

# Iterate over parameters
for param in module.parameters():
    ...

# Estimate multiplicative depth
depth = module.mult_depth() -> int
```

---

### EncryptedLinear

Encrypted fully-connected layer.

```python
from cukks.nn import EncryptedLinear
```

#### Constructor

```python
EncryptedLinear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
)
```

#### Class Methods

```python
# Create from PyTorch Linear layer
enc_linear = EncryptedLinear.from_torch(linear: torch.nn.Linear) -> EncryptedLinear
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `in_features` | `int` | Input dimension |
| `out_features` | `int` | Output dimension |
| `weight` | `torch.Tensor` | Weight matrix |
| `bias` | `torch.Tensor` | Bias vector (may be None) |

##### `from_torch_cnn()`

Create from a PyTorch Linear layer following CNN layers, absorbing the Flatten permutation.

```python
enc_linear = EncryptedLinear.from_torch_cnn(
    linear: torch.nn.Linear,
    cnn_layout: Dict
) -> EncryptedLinear
```

**Parameters:**
- `linear`: The PyTorch Linear layer to convert.
- `cnn_layout`: Dictionary with `'num_patches'` and `'patch_features'`.

**Returns:** `EncryptedLinear` with permuted weights (Flatten absorbed).

**Example:**

```python
cnn_layout = {'num_patches': 16, 'patch_features': 8}  # 4x4 spatial, 8 channels
enc_fc = EncryptedLinear.from_torch_cnn(fc_layer, cnn_layout)
```

---

### EncryptedConv2d

Encrypted 2D convolution layer using im2col method.

```python
from cukks.nn import EncryptedConv2d
```

#### Constructor

```python
EncryptedConv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `in_channels` | `int` | Required | Number of input channels |
| `out_channels` | `int` | Required | Number of output channels |
| `kernel_size` | `int` or `Tuple[int, int]` | Required | Size of the convolution kernel |
| `weight` | `torch.Tensor` | Required | Convolution kernel weights |
| `bias` | `torch.Tensor` | `None` | Optional bias vector |
| `stride` | `int` or `Tuple[int, int]` | `1` | Stride of the convolution |
| `padding` | `int` or `Tuple[int, int]` | `0` | Padding added to input |
| `groups` | `int` | `1` | Number of blocked connections from input to output channels |
| `dilation` | `int` or `Tuple[int, int]` | `1` | Spacing between kernel elements |

#### Class Methods

```python
enc_conv = EncryptedConv2d.from_torch(conv: torch.nn.Conv2d) -> EncryptedConv2d
```

---

### Activation Functions

All activation modules support polynomial approximation.

#### EncryptedSquare

Exact x² activation (recommended for accuracy).

```python
from cukks.nn import EncryptedSquare

activation = EncryptedSquare()
```

#### EncryptedReLU

Polynomial approximation of ReLU.

```python
from cukks.nn import EncryptedReLU

activation = EncryptedReLU(
    degree: int = 4,
    domain: tuple = (-1, 1),
    method: str = "chebyshev"  # or "minimax"
)
```

#### EncryptedGELU

Polynomial approximation of GELU.

```python
from cukks.nn import EncryptedGELU

activation = EncryptedGELU(degree: int = 4)
```

#### EncryptedSiLU

Polynomial approximation of SiLU (Swish).

```python
from cukks.nn import EncryptedSiLU

activation = EncryptedSiLU(degree: int = 4)
```

#### EncryptedSigmoid

Polynomial approximation of sigmoid.

```python
from cukks.nn import EncryptedSigmoid

activation = EncryptedSigmoid(degree: int = 4)
```

#### EncryptedTanh

Polynomial approximation of tanh.

```python
from cukks.nn import EncryptedTanh

activation = EncryptedTanh(degree: int = 5)
```

#### EncryptedPolynomial

Custom polynomial activation.

```python
from cukks.nn import EncryptedPolynomial

# coeffs = [a0, a1, a2, ...] for a0 + a1*x + a2*x² + ...
activation = EncryptedPolynomial(coeffs: Sequence[float])
```

---

### EncryptedSequential

Container for sequential execution of encrypted modules.

```python
from cukks.nn import EncryptedSequential

model = EncryptedSequential(
    EncryptedLinear(...),
    EncryptedReLU(),
    EncryptedLinear(...),
)

# Or with named modules
model = EncryptedSequential(OrderedDict([
    ('fc1', EncryptedLinear(...)),
    ('act1', EncryptedReLU()),
    ('fc2', EncryptedLinear(...)),
]))
```

---

### Other Layers

#### EncryptedFlatten

```python
from cukks.nn import EncryptedFlatten

flatten = EncryptedFlatten(
    start_dim: int = 0, 
    end_dim: int = -1
)
```

**Parameters:**
- `start_dim`: First dimension to flatten (default: 0).
- `end_dim`: Last dimension to flatten (default: -1).

**Note:** For CNN models, the converter automatically optimizes Flatten by absorbing the permutation into the following Linear layer's weights. This is handled internally.

#### EncryptedAvgPool2d

```python
from cukks.nn import EncryptedAvgPool2d

pool = EncryptedAvgPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0
)

# Create from PyTorch layer
pool = EncryptedAvgPool2d.from_torch(avg_pool: torch.nn.AvgPool2d)
```

**Parameters:**
- `kernel_size`: Size of the pooling window.
- `stride`: Stride of the pooling (default: same as kernel_size).
- `padding`: Padding to add (default: 0).

**Note:** 4D input is no longer supported; use `encrypt_cnn_input()` to preprocess CNN inputs. For 2x2 pooling, rotation-based optimization is automatically applied for better performance.

#### EncryptedMaxPool2d

Approximate max pooling using polynomial approximation.

```python
from cukks.nn import EncryptedMaxPool2d

pool = EncryptedMaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    degree: int = 4,
)

# Create from PyTorch layer
pool = EncryptedMaxPool2d.from_torch(max_pool: torch.nn.MaxPool2d)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `kernel_size` | `int` or `Tuple[int, int]` | Required | Size of the pooling window |
| `stride` | `int` or `Tuple[int, int]` | `None` | Stride of the pooling. Defaults to `kernel_size`. |
| `padding` | `int` or `Tuple[int, int]` | `0` | Padding to add |
| `degree` | `int` | `4` | Polynomial degree for |x| approximation. Higher = more accurate but deeper circuit. |

**Note:** Max pooling is approximated using `max(a, b) ≈ (a + b + |a - b|) / 2`, where |x| is fitted via polynomial approximation. 4D input is no longer supported; use `encrypt_cnn_input()` to preprocess CNN inputs. The polynomial path is used for HE CNN layouts. Accuracy depends on input normalization; best for values in [-1, 1].

#### EncryptedBatchNorm1d / EncryptedBatchNorm2d

Note: BatchNorm is typically folded into preceding layers during conversion.

```python
from cukks.nn import EncryptedBatchNorm1d, EncryptedBatchNorm2d

bn = EncryptedBatchNorm1d.from_torch(bn: torch.nn.BatchNorm1d)
```

#### EncryptedLayerNorm

Encrypted layer normalization using pure HE polynomial approximation.

```python
from cukks.nn import EncryptedLayerNorm

ln = EncryptedLayerNorm(
    normalized_shape: Union[int, List[int], Tuple[int, ...]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
)

# Create from PyTorch layer
ln = EncryptedLayerNorm.from_torch(layer_norm: torch.nn.LayerNorm)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `normalized_shape` | `int` or `List[int]` | Required | Input shape for normalization |
| `weight` | `torch.Tensor` | `None` | Learnable scale parameter (gamma) |
| `bias` | `torch.Tensor` | `None` | Learnable shift parameter (beta) |
| `eps` | `float` | `1e-5` | Numerical stability constant |

**Note:** LayerNorm uses pure HE polynomial approximation with sum_and_broadcast for mean/variance computation and Chebyshev degree-15 polynomial for 1/sqrt(var+eps). Multiplicative depth: ~18 (includes polynomial evaluation for inverse square root).

#### EncryptedDropout

Dropout layer for encrypted inference. Acts as a no-op (returns input unchanged) since dropout is only active during training.

```python
from cukks.nn import EncryptedDropout

dropout = EncryptedDropout(p: float = 0.5)

# Create from PyTorch layer
dropout = EncryptedDropout.from_torch(dropout_layer: torch.nn.Dropout)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p` | `float` | `0.5` | Dropout probability. Stored for compatibility but has no effect during inference. |

**Note:** During inference, dropout is disabled (no-op). This module exists for model conversion compatibility so that `nn.Dropout` layers are properly handled by `convert()`.

#### EncryptedResidualBlock

Residual connection block.

```python
from cukks.nn import EncryptedResidualBlock

block = EncryptedResidualBlock(main_branch: EncryptedModule)
```

#### EncryptedApproxAttention

Approximate multi-head attention for encrypted transformer inference using pure HE cipher-cipher operations. Uses polynomial approximation for softmax via Taylor expansion of exp(x).

```python
from cukks.nn import EncryptedApproxAttention

attention = EncryptedApproxAttention(
    embed_dim: int,
    num_heads: int,
    softmax_degree: int = 4,
)

# Create from PyTorch layer
attention = EncryptedApproxAttention.from_torch(
    attention: torch.nn.MultiheadAttention,
    softmax_degree: int = 4,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `embed_dim` | `int` | Required | Total embedding dimension |
| `num_heads` | `int` | Required | Number of attention heads. `embed_dim` must be divisible by `num_heads`. |
| `softmax_degree` | `int` | `4` | Polynomial degree for exp(x) Taylor approximation. Higher = more accurate but deeper circuit. |

**Methods:**

- `forward(x)`: Self-attention — uses `x` as query, key, and value. Supports seq_len=1 only in pure HE mode.
- `forward_attention(query, key, value)`: Full attention with separate Q, K, V inputs. Supports seq_len=1 only in pure HE mode.
- `mult_depth()`: Returns estimated multiplicative depth (includes projections + Q@K^T + softmax polynomial + attn@V + output projection).

**Note:** This is an approximation using cipher-cipher multiplication for Q@K^T and sum_and_broadcast for attention aggregation. Accuracy depends on input range and polynomial degree. Best results when attention scores are normalized to a small range (e.g., [-2, 2]). `from_torch()` extracts Q, K, V, and output projection weights from `nn.MultiheadAttention`. Limited to seq_len=1 in pure HE mode.

---

## Utilities

### SlotPacker

Utility for packing multiple samples into CKKS slots.

```python
from cukks.batching import SlotPacker

packer = SlotPacker(
    slots_per_sample: int,
    total_slots: int
)
```

#### Methods

```python
# Pack samples into a single tensor
packed = packer.pack(samples: List[torch.Tensor]) -> torch.Tensor

# Unpack a tensor into samples
samples = packer.unpack(
    packed: torch.Tensor,
    num_samples: int
) -> List[torch.Tensor]
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `slots_per_sample` | `int` | Slots allocated per sample |
| `total_slots` | `int` | Total available slots |
| `max_samples` | `int` | Maximum samples that can be packed |

---

## Module Functions

### Top-level Functions

```python
import cukks

# Check if backend is available
available = cukks.is_available() -> bool

# Get backend information
info = cukks.get_backend_info() -> dict
# Returns: {"backend": "openfhe-gpu", "available": True, "cuda": True}
```
