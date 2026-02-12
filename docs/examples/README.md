# CuKKS Examples

This directory contains example scripts demonstrating various features of the CuKKS library.

## Prerequisites

```bash
# Install CuKKS
pip install -e .

# For GPU examples, ensure OpenFHE GPU backend is built
# See main README for build instructions
```

## Examples Overview

| Example | Description | Difficulty |
|---------|-------------|------------|
| [01_basic_encryption.py](01_basic_encryption.py) | Basic encryption/decryption | Beginner |
| [02_mlp_inference.py](02_mlp_inference.py) | MLP neural network inference | Beginner |
| [03_cnn_inference.py](03_cnn_inference.py) | CNN image classification | Intermediate |
| [04_statistics.py](04_statistics.py) | PP-STAT statistical functions | Intermediate |
| [05_bootstrapping.py](05_bootstrapping.py) | GPU-accelerated bootstrapping | Advanced |

---

## 01. Basic Encryption

**File:** `01_basic_encryption.py`

Learn the fundamentals of CKKS homomorphic encryption:
- Creating a CKKS context with `CKKSConfig`
- Encrypting and decrypting vectors
- Basic operations: addition, multiplication, scalar multiplication
- Error analysis

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(
    poly_mod_degree=8192,
    coeff_mod_bits=[60, 40, 40, 60],
    scale_bits=40,
)
ctx = CKKSContext(config, enable_gpu=True)

enc_x = ctx.encrypt(x)
enc_result = enc_x + enc_y  # Homomorphic addition
result = ctx.decrypt(enc_result)
```

---

## 02. MLP Inference

**File:** `02_mlp_inference.py`

Perform encrypted inference on a Multi-Layer Perceptron:
- Convert PyTorch layers to encrypted layers
- BSGS optimization for matrix multiplication
- Square activation (CKKS-friendly alternative to ReLU)

```python
from cukks import CKKSInferenceContext
from cukks.nn import EncryptedLinear, EncryptedSquare

ctx = CKKSInferenceContext(max_rotation_dim=64, use_bsgs=True)

enc_fc1 = EncryptedLinear.from_torch(model.fc1)
enc_square = EncryptedSquare()
enc_fc2 = EncryptedLinear.from_torch(model.fc2)

enc_x = ctx.encrypt(x)
enc_out = enc_fc2(enc_square(enc_fc1(enc_x)))
```

---

## 03. CNN Inference

**File:** `03_cnn_inference.py`

GPU-accelerated encrypted CNN inference:
- Conv2d → Square → AvgPool2d → Flatten → Linear pipeline
- im2col transformation for efficient convolution
- CNN-specific context configuration
- Rotation-based pooling

```python
from cukks import CKKSInferenceContext
from cukks.nn import (
    EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
    EncryptedFlatten, EncryptedLinear
)

ctx = CKKSInferenceContext(
    max_rotation_dim=576,
    use_bsgs=True,
    cnn_config={
        'image_height': 8,
        'image_width': 8,
        'channels': 4,
        'pool_size': 2,
        'pool_stride': 2,
    }
)

enc_input = ctx.encrypt_cnn_input(image, conv_params)
# Conv → Square → Pool → Flatten → FC
enc_out = enc_fc(enc_flatten(enc_pool(enc_square(enc_conv(enc_input)))))
```

---

## 04. Statistics (PP-STAT)

**File:** `04_statistics.py`

Privacy-preserving statistical functions based on the PP-STAT paper:
- `encrypted_mean`: Encrypted mean calculation
- `encrypted_variance`: Encrypted variance calculation  
- `encrypted_std`: Encrypted standard deviation (requires bootstrap)
- `crypto_inv_sqrt`: Encrypted inverse square root (1/√x)
- `crypto_inv_sqrt_shallow`: Bootstrap-free variant

```python
from cukks.stats import (
    encrypted_mean, encrypted_variance, encrypted_std,
    crypto_inv_sqrt_shallow
)

enc_x = ctx.encrypt(x)

# Mean and variance
enc_mean = encrypted_mean(enc_x)
enc_var = encrypted_variance(enc_x)

# Inverse square root (for normalization)
enc_inv_sqrt = crypto_inv_sqrt_shallow(enc_var, domain=(1.0, 50.0))
```

**Reference:**
> Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving Statistical Analysis Framework. CIKM'25.

---

## 05. Bootstrapping

**File:** `05_bootstrapping.py`

GPU-accelerated CKKS bootstrapping for deep computations:
- Noise and level restoration
- Deep computation patterns
- Performance benchmarking

```python
config = CKKSConfig(
    poly_mod_degree=32768,
    coeff_mod_bits=[60] + [40] * 14 + [60],
    scale_bits=40,
    enable_bootstrap=True,
)

ctx = CKKSContext(config, enable_gpu=True)
enc_x = ctx.encrypt(x)

# After many operations...
enc_result = enc_x
for _ in range(5):
    enc_result = enc_result.mul(enc_result).rescale()

# Restore levels with bootstrap
enc_result = enc_result.bootstrap()
```

**Note:** Bootstrapping is computationally intensive. Prefer `crypto_inv_sqrt_shallow` when bootstrap-free alternatives are available.

---

## Running Examples

```bash
# Run individual examples
python docs/examples/01_basic_encryption.py
python docs/examples/02_mlp_inference.py
python docs/examples/03_cnn_inference.py
python docs/examples/04_statistics.py
python docs/examples/05_bootstrapping.py
```

## Troubleshooting

### GPU Not Available
```python
ctx = CKKSContext(config, enable_gpu=False)  # Force CPU mode
```

### Bootstrap Errors
- Ensure `poly_mod_degree >= 32768`
- Try `crypto_inv_sqrt_shallow` instead

### Memory Issues
- Reduce `poly_mod_degree` (e.g., 16384 instead of 32768)
- Process data in smaller batches

---

## See Also

- [API Reference](../api.md)
- [Korean Documentation](../ko/)
- [Main README](../../README.md)
