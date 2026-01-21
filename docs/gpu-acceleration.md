# GPU Acceleration for ckks-torch

## Overview

ckks-torch supports GPU acceleration for homomorphic encryption operations using the OpenFHE-GPU library. The implementation uses a hybrid CPU/GPU approach:

- **GPU accelerated**: Add, Sub operations
- **CPU fallback**: Mul, Square, Rotate (due to GPU-CPU data transfer compatibility)
- **Lazy synchronization**: GPU results are synced to CPU only when needed (e.g., decryption)

## Requirements

- NVIDIA GPU with CUDA support
- OpenFHE-GPU library (included in `openfhe-gpu-public/`)
- CUDA toolkit

## Installation

The GPU backend is built automatically with the standard installation:

```bash
cd bindings/openfhe_backend
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This builds both:
- `ckks_openfhe_backend.so` - CPU-only backend
- `ckks_openfhe_gpu_backend.so` - GPU-accelerated backend

## Usage

### Basic Usage

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

# Encrypt and compute
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
enc_x = ctx.encrypt(x)

# GPU-accelerated add
enc_add = enc_x.add(enc_x)

# CPU fallback mul
enc_mul = enc_x.mul(enc_x)

# Decrypt
result = ctx.decrypt(enc_add)
```

### Disabling GPU

```python
# Disable GPU acceleration
ctx = CKKSContext(config, enable_gpu=False)

# Or via environment variable
import os
os.environ['CKKS_USE_GPU'] = '0'
```

## Performance

### Lazy Synchronization

The GPU backend uses lazy synchronization - GPU computation results are only transferred to CPU when needed (e.g., for decryption). This provides significant performance benefits for chained operations:

| Operation | Time |
|-----------|------|
| Single add + decrypt | 11.88 ms |
| 3 chained adds (no decrypt) | 8.16 ms |
| 3 chained adds + decrypt | 17.38 ms |

**Efficiency gain**: ~2.1x faster when batching operations

### Benchmark Results (poly_mod_degree=16384)

| Operation | GPU Backend | CPU Backend |
|-----------|-------------|-------------|
| Add | Uses GPU | CPU only |
| Sub | Uses GPU | CPU only |
| Mul | CPU fallback | CPU only |
| Rotate | CPU | CPU only |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python API                            │
│              (ckks/torch_api.py)                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │  CPU Backend    │      │     GPU Backend          │  │
│  │ (ckks_openfhe_  │      │ (ckks_openfhe_gpu_      │  │
│  │  backend.so)    │      │  backend.so)             │  │
│  └────────┬────────┘      └───────────┬─────────────┘  │
│           │                           │                  │
│           ▼                           ▼                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │              OpenFHE Library                     │    │
│  │         (CPU HE operations)                      │    │
│  └─────────────────────────────────────────────────┘    │
│                           │                              │
│                           ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │           OpenFHE-GPU Library                    │    │
│  │    (GPU accelerated: Add, Sub, etc.)            │    │
│  └─────────────────────────────────────────────────┘    │
│                           │                              │
│                           ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │              CUDA Runtime                        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Implementation Details

### GPU Context Initialization

When `enable_gpu=True`, the context initializes:
1. Standard OpenFHE crypto context
2. GPU context via `GenGPUContext()`
3. GPU evaluation keys via `LoadEvalMultRelinKey()`

### Data Flow

1. **Encryption**: Always on CPU (uses secret key)
2. **GPU Operations**: Data loaded to GPU via `LoadAccurateCiphertext()`
3. **Lazy Sync**: GPU results stored in `CtAccurate` struct
4. **Decryption**: GPU data synced to CPU via `loadIntoDCRTPoly()`

### Supported Operations

| Operation | GPU Support | Notes |
|-----------|-------------|-------|
| `encrypt()` | CPU | Secret key operation |
| `decrypt()` | CPU | Secret key operation |
| `add()` | GPU | Full GPU acceleration |
| `sub()` | GPU | Full GPU acceleration |
| `mul()` | GPU | Full GPU acceleration |
| `square()` | GPU | Uses mul internally |
| `rotate()` | GPU | Full GPU acceleration |
| `add_plain()` | GPU | GPU acceleration |
| `mul_plain()` | GPU | GPU acceleration |
| `rescale()` | GPU | GPU acceleration |
| `bootstrap()` | GPU | Full GPU acceleration |

## CNN Inference on GPU

GPU-accelerated CNN inference is fully supported with the following configuration:

```python
from ckks_torch import CKKSInferenceContext
from ckks_torch.nn import (
    EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d, 
    EncryptedFlatten, EncryptedLinear
)

# Create context with CNN configuration
ctx = CKKSInferenceContext(
    max_rotation_dim=576,
    use_bsgs=True,
    cnn_config={
        'image_height': 8,      # Feature map height after conv
        'image_width': 8,       # Feature map width after conv
        'channels': 4,          # Number of output channels
        'pool_size': 2,         # Pooling kernel size
        'pool_stride': 2,       # Pooling stride
    }
)

# Convert PyTorch layers
conv_params = [{'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'out_channels': 4}]
enc_conv = EncryptedConv2d.from_torch(model.conv)
enc_square = EncryptedSquare()
enc_pool = EncryptedAvgPool2d.from_torch(model.pool)
enc_flatten = EncryptedFlatten._with_absorbed_permutation()

# Run inference
enc_input = ctx.encrypt_cnn_input(image.squeeze(0), conv_params)
enc_x = enc_conv(enc_input)
enc_x = enc_square(enc_x)
enc_x = enc_pool(enc_x)

# FC layer needs sparse layout info from pooling
cnn_layout = enc_x._cnn_layout
enc_fc = EncryptedLinear.from_torch_cnn(model.fc, cnn_layout)

enc_x = enc_flatten(enc_x)
enc_output = enc_fc(enc_x)

# Decrypt result
output = ctx.decrypt(enc_output)[:10]
```

### CNN Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `image_height` | Height of feature map after conv | 8 |
| `image_width` | Width of feature map after conv | 8 |
| `channels` | Number of output channels from conv | 4 |
| `pool_size` | Pooling window size | 2 |
| `pool_stride` | Pooling stride | 2 |

### CNN Accuracy

| Model | Cosine Similarity |
|-------|-------------------|
| Conv(1→4) + Square + Pool(2×2) + FC(64→10) | **0.8975** |

## GPU Bootstrapping

GPU-accelerated bootstrapping is supported for deep network inference:

```python
from ckks_torch.context import InferenceConfig, CKKSInferenceContext
import torch

# Create context with bootstrapping enabled
config = InferenceConfig(
    poly_mod_degree=32768,
    mult_depth=20,            # Higher depth for bootstrapping
    enable_bootstrap=True,
    level_budget=(5, 5),      # Bootstrapping level budget
    security_level=None,      # Required for custom depth settings
)

ctx = CKKSInferenceContext(config=config, max_rotation_dim=256)

# Encrypt and perform operations
data = torch.randn(100, dtype=torch.float64)
enc = ctx.backend.encrypt(data)

# Bootstrap to refresh ciphertext level
enc_refreshed = enc.bootstrap()

# Decrypt
dec = ctx.backend.decrypt(enc_refreshed)[:10]
```

### Bootstrapping Configuration

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `poly_mod_degree` | Ring dimension | 32768 (minimum for bootstrapping) |
| `mult_depth` | Multiplicative depth | 16-20 |
| `level_budget` | (coeffsToSlots, slotsToCoeffs) levels | (4, 4) to (5, 5) |
| `security_level` | Must be None for custom depth | None |

### Bootstrapping Accuracy

| Setting | Max Error |
|---------|-----------|
| level_budget=(5,5), mult_depth=20 | **1.6e-07** |

## Troubleshooting

### GPU not detected

```python
ctx = CKKSContext(config, enable_gpu=True)
print(ctx.gpu_enabled)  # Should be True
```

If `False`, check:
1. CUDA installation: `nvidia-smi`
2. GPU backend built: check for `ckks_openfhe_gpu_backend.so`

### Memory issues with large parameters

For large `poly_mod_degree` (e.g., 32768), GPU memory may be insufficient. Reduce parameters or use CPU backend.

### Segfault with repeated operations

Ensure proper garbage collection between heavy operations:
```python
import gc
gc.collect()
```

## Future Improvements

1. **Multi-GPU**: Distribute operations across multiple GPUs
2. **Memory optimization**: Reduce GPU memory usage for large parameters
3. **Streaming encryption**: Pipeline encryption with GPU operations
