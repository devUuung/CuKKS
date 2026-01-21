# CKKS-Torch

**PyTorch-compatible encrypted deep learning inference using CKKS homomorphic encryption with GPU acceleration.**

CKKS-Torch enables you to run trained PyTorch models on encrypted data, preserving privacy while maintaining model accuracy. Built on top of OpenFHE with CUDA acceleration.

## Features

- **PyTorch-like API**: Familiar interface for deep learning practitioners
- **Automatic Model Conversion**: Convert trained PyTorch models with one function call
- **GPU Acceleration**: CUDA-accelerated CKKS operations via OpenFHE
- **Polynomial Activations**: Pre-computed approximations for ReLU, GELU, SiLU, etc.
- **BatchNorm Folding**: Automatic optimization for efficient inference
- **Batch Processing**: Pack multiple samples into a single ciphertext for SIMD parallelism
- **Flexible Configuration**: Easy parameter tuning for different security/performance tradeoffs

## Quick Start

### MLP Example

```python
import torch.nn as nn
import ckks_torch

# 1. Train your model normally in PyTorch
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
train(model, data)

# 2. Convert to encrypted model
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 3. Encrypt input and run inference
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)

# 4. Decrypt output
output = ctx.decrypt(enc_output)
```

### CNN Example

```python
import torch.nn as nn
import ckks_torch

# Define CNN with all operations as layer attributes
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()           # Will be replaced with x^2
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()     # Must be a layer attribute
        self.fc = nn.Linear(8 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MNISTCNN()
model.eval()

# Convert to encrypted model
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# Encrypt and run inference
enc_input = ctx.encrypt(image)  # shape: (1, 1, 28, 28)
enc_output = enc_model(enc_input)
prediction = ctx.decrypt(enc_output).argmax()
```

> **Important**: All operations in `forward()` must be layer attributes (e.g., `self.act1`, `self.flatten`), not inline operations like `x ** 2` or `x.flatten(1)`.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- CMake 3.16+ (for building OpenFHE backend)
- GCC 9+ or Clang 10+

### Step 1: Install Python Package (High-Level API)

```bash
git clone https://github.com/ckks-torch/ckks-torch.git
cd ckks-torch
pip install -e .
```

### Step 2: Build OpenFHE Backend (Required for Encryption)

The CKKS backend requires building the OpenFHE GPU extension:

```bash
# Build OpenFHE with GPU support
cd openfhe-gpu-public
mkdir build && cd build
cmake .. -DWITH_CUDA=ON
make -j$(nproc)

# Build Python bindings
cd ../../bindings/openfhe_backend
export LD_LIBRARY_PATH="$PWD/../../openfhe-gpu-public/build/lib:$LD_LIBRARY_PATH"
pip install -e .
```

### Verify Installation

```python
import ckks_torch

# Check if backend is available
print(ckks_torch.is_available())  # True if OpenFHE backend is installed
print(ckks_torch.get_backend_info())
```

## Documentation

- **[API Reference](docs/api.md)**: Complete API documentation
- **[GPU Acceleration](docs/gpu-acceleration.md)**: GPU setup and performance tuning
- **[CKKS Concepts](docs/concepts.md)**: Understanding homomorphic encryption
- **[Examples](docs/examples/)**: Working code examples

## Supported Layers

| PyTorch Layer | Encrypted Version | Notes |
|--------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | Full support with BSGS optimization |
| `nn.Conv2d` | `EncryptedConv2d` | Via im2col method |
| `nn.ReLU` | `EncryptedReLU` | Polynomial approximation |
| `nn.GELU` | `EncryptedGELU` | Polynomial approximation |
| `nn.SiLU` | `EncryptedSiLU` | Polynomial approximation |
| `nn.Sigmoid` | `EncryptedSigmoid` | Polynomial approximation |
| `nn.Tanh` | `EncryptedTanh` | Polynomial approximation |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | Full support |
| `nn.MaxPool2d` | `EncryptedMaxPool2d` | Approximate via polynomial |
| `nn.Flatten` | `EncryptedFlatten` | Logical reshape |
| `nn.BatchNorm1d/2d` | Folded | Merged into preceding layer |
| `nn.Sequential` | `EncryptedSequential` | Full support |
| `nn.Dropout` | Ignored | No-op during inference |

## Activation Functions

CKKS only supports polynomial operations, so non-linear activations must be approximated:

### Option 1: Square Activation (Recommended for Maximum Accuracy)

```python
# Use x^2 activation - exact in CKKS, no approximation error
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)
```

### Option 2: Polynomial Approximation

```python
# Use Chebyshev polynomial approximation of ReLU
enc_model, ctx = ckks_torch.convert(
    model,
    use_square_activation=False,
    activation_degree=4  # Higher = more accurate but deeper circuit
)
```

## Configuration

### Inference Context

```python
from ckks_torch import CKKSInferenceContext, InferenceConfig

# Auto-configure based on model
ctx = CKKSInferenceContext.for_model(model)

# Or manually configure
config = InferenceConfig(
    poly_mod_degree=16384,      # Ring dimension (power of 2)
    scale_bits=40,              # Precision bits
    security_level="128_classic",
    mult_depth=6,               # Multiplicative depth
    enable_bootstrap=False,     # For very deep networks
)
ctx = CKKSInferenceContext(config)
```

### Conversion Options

```python
from ckks_torch.converter import ModelConverter, ConversionOptions

options = ConversionOptions(
    fold_batchnorm=True,        # Fold BN into preceding layers
    activation_degree=4,        # Polynomial degree for activations
    use_square_activation=False # Use x^2 instead of approximations
)

converter = ModelConverter(options)
enc_model = converter.convert(model)
```

## Examples

### Run the Demo

```bash
# Model conversion demo (no GPU required)
python -m ckks_torch.examples.encrypted_inference --demo conversion

# Full encrypted inference (requires CKKS backend)
python -m ckks_torch.examples.encrypted_inference --demo inference
```

### MNIST Encrypted Inference

```bash
# Run MNIST example with synthetic data
python examples/mnist_encrypted.py --hidden 64 --samples 5

# Use real MNIST dataset
python examples/mnist_encrypted.py --use-mnist --samples 10
```

See [docs/examples/mnist.py](docs/examples/mnist.py) for a simplified example.

### Custom Polynomial Activations

```python
from ckks_torch.utils.approximations import chebyshev_coefficients

# Compute custom ReLU approximation
def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

coeffs = chebyshev_coefficients(relu, degree=7, domain=(-1, 1))

# Use in model
from ckks_torch.nn import EncryptedPolynomial
custom_activation = EncryptedPolynomial(coeffs)
```

### Batch Processing

```python
# Pack multiple samples into a single ciphertext
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)

# Run inference on all 8 samples at once
enc_output = enc_model(enc_batch)

# Decrypt individual results
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

## Architecture

```
ckks_torch/
├── __init__.py          # Main exports
├── context.py           # CKKSInferenceContext
├── tensor.py            # EncryptedTensor
├── converter.py         # PyTorch → Encrypted conversion
├── batching/            # Batch processing utilities
│   └── packing.py       # SlotPacker for SIMD batching
├── nn/                  # Encrypted neural network layers
│   ├── module.py        # Base class
│   ├── linear.py        # EncryptedLinear
│   ├── conv.py          # EncryptedConv2d
│   ├── activations.py   # Polynomial activations
│   ├── pooling.py       # EncryptedAvgPool2d, EncryptedMaxPool2d
│   ├── batchnorm.py     # BatchNorm (for folding)
│   ├── residual.py      # EncryptedResidualBlock
│   └── attention.py     # EncryptedApproxAttention
└── utils/
    └── approximations.py # Polynomial fitting utilities
```

## GPU Acceleration

CKKS-Torch supports GPU acceleration for homomorphic encryption operations:

### Enabling GPU

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(
    poly_mod_degree=8192,
    scale_bits=40,
    rotations=list(range(-16, 17)),
    coeff_mod_bits=[60, 40, 40, 40, 40, 60],
)

# GPU enabled by default
ctx = CKKSContext(config, enable_gpu=True)
print(f"GPU enabled: {ctx.gpu_enabled}")

# Disable GPU
ctx = CKKSContext(config, enable_gpu=False)
```

### GPU-Accelerated Operations

| Operation | Acceleration |
|-----------|-------------|
| Add/Sub | GPU |
| Mul/Square | CPU (fallback) |
| Rotate | CPU |
| Encrypt/Decrypt | CPU |

### Lazy Synchronization

GPU results are synced to CPU only when needed (e.g., for decryption), providing ~2x efficiency improvement for chained operations:

```python
# Efficient: chain operations, decrypt once
enc_result = enc_x.add(enc_x).add(enc_x).add(enc_x)
result = ctx.decrypt(enc_result)  # Sync happens here
```

See [GPU Acceleration Guide](docs/gpu-acceleration.md) for detailed documentation.

## Performance Tips

1. **Use square activation** when possible - it's exact in CKKS
2. **Minimize multiplicative depth** - each mult consumes precision
3. **Fold BatchNorm** before conversion (done automatically)
4. **Choose appropriate ring dimension** - larger = more slots but slower
5. **Use batch processing** - pack multiple samples for SIMD parallelism
6. **Consider bootstrapping** for very deep networks (>10 layers)
7. **Use BSGS optimization** for matrix-vector products (enabled by default)

### CNN-Specific Optimizations

CKKS-Torch includes several CNN-specific optimizations that are automatically applied:

#### Flatten Absorption
The permutation operation in `Flatten` is absorbed into the following `Linear` layer's weights, eliminating a costly matmul operation at runtime.

```python
# Automatic optimization when converting CNN models
enc_model, ctx = ckks_torch.convert(cnn_model, optimize_cnn=True)
```

#### Pool Rotation Optimization
For 2x2 average pooling, rotation-based summation is used instead of sparse matrix multiplication, reducing the number of HE operations.

#### Lazy Rescale
Rescale operations are deferred until needed (e.g., before the next multiplication), reducing unnecessary rescale calls and preserving precision levels.

### CNN Performance Benchmarks

Tested on 8×8 downsampled MNIST with Conv(8ch) → Square → AvgPool(2) → FC(10):

| Optimization | Time | Improvement |
|--------------|------|-------------|
| Baseline | 3.12s | - |
| + Flatten Absorption | 2.81s | 10% |
| + Pool Rotation + Lazy Rescale | 2.74s | **12%** |

### True HE CNN Inference

CKKS-Torch implements *true* homomorphic CNN inference where all operations run on encrypted data without decryption:

```python
# Manual HE CNN pipeline (for advanced users)
ctx = ckks_torch.CKKSInferenceContext(max_rotation_dim=576, use_bsgs=True)

# Pre-apply im2col before encryption
conv_params = [{'kernel_size': (3,3), 'stride': (1,1), 'padding': (1,1), 'out_channels': 8}]
enc_x = ctx.encrypt_cnn_input(image, conv_params)

# All operations run on encrypted data
enc_out = EncryptedConv2d.from_torch(conv)(enc_x)   # HE matmul
enc_out = EncryptedSquare()(enc_out)                 # HE square
enc_out = EncryptedAvgPool2d.from_torch(pool)(enc_out)  # HE rotation-based pool
enc_out = EncryptedFlatten(absorb_permutation=True)(enc_out)  # No-op
enc_out = EncryptedLinear.from_torch_cnn(fc, cnn_layout)(enc_out)  # HE matmul with permuted weights

result = ctx.decrypt(enc_out)
```

## Limitations

- **Inference only**: No encrypted training support
- **Approximate activations**: ReLU/GELU are polynomial approximations
- **Fixed precision**: Accumulated errors grow with network depth
- **Memory intensive**: CKKS operations require significant GPU memory

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `poly_mod_degree` (e.g., 8192 instead of 16384)
- Reduce `mult_depth` if possible
- Use fewer rotation keys

**Low Accuracy**:
- Increase `activation_degree` for better approximations
- Consider using `use_square_activation=True`
- Normalize inputs to [-1, 1] range

**Slow Performance**:
- Enable BSGS optimization (default)
- Use batch processing for multiple samples
- Reduce network depth if possible

## License

Apache License 2.0

## Citation

If you use CKKS-Torch in your research, please cite:

```bibtex
@software{ckks_torch,
  title = {CKKS-Torch: PyTorch-compatible Encrypted Deep Learning},
  year = {2024},
  url = {https://github.com/ckks-torch/ckks-torch}
}
```

## Related Resources

- [OpenFHE](https://github.com/openfheorg/openfhe-development): The underlying HE library
- [CKKS Paper](https://eprint.iacr.org/2016/421): Original CKKS scheme
- [Microsoft SEAL](https://github.com/microsoft/SEAL): Alternative HE library
