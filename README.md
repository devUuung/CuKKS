<p align="center">
  <a href="README.md">English</a> |
  <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">CuKKS</h1>

<p align="center">
  <strong>GPU-accelerated CKKS Homomorphic Encryption for PyTorch</strong>
</p>

<p align="center">
  <a href="https://github.com/devUuung/CuKKS/actions"><img src="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/devUuung/CuKKS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
</p>

<p align="center">
  Run trained PyTorch models on <strong>encrypted data</strong> — preserving privacy while maintaining accuracy.<br>
  Built on OpenFHE with CUDA acceleration.
</p>

---

## Quick Start

```python
import torch.nn as nn
import ckks_torch

# 1. Define and train your model (standard PyTorch)
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. Convert to encrypted model
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 3. Run encrypted inference
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)
output = ctx.decrypt(enc_output)
```

## Installation

```bash
pip install cukks-cu121  # For CUDA 12.1 (most common)
```

<details>
<summary><strong>Other CUDA versions</strong></summary>

| Package | CUDA | Supported GPUs |
|---------|------|----------------|
| `cukks-cu118` | 11.8 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu121` | 12.1 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu124` | 12.4 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu128` | 12.8 | All above + **RTX 50xx** |
| `cukks` | - | CPU only |

```bash
pip install cukks-cu118  # CUDA 11.8
pip install cukks-cu124  # CUDA 12.4
pip install cukks-cu128  # CUDA 12.8 (RTX 50xx)
pip install cukks        # CPU only
```

</details>

<details>
<summary><strong>Docker images</strong></summary>

| CUDA | Compatible Docker Images |
|------|-------------------------|
| 11.8 | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` |
| 12.1 | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| 12.4 | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` |
| 12.8 | `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04` |

```bash
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install cukks-cu121
```

</details>

<details>
<summary><strong>Build from source</strong></summary>

```bash
git clone https://github.com/devUuung/CuKKS.git && cd CuKKS
pip install -e .

# Build OpenFHE backend
cd openfhe-gpu-public && mkdir build && cd build
cmake .. -DWITH_CUDA=ON && make -j$(nproc)

cd ../../bindings/openfhe_backend
pip install -e .
```

</details>

## Features

| Feature | Description |
|---------|-------------|
| **PyTorch API** | Familiar interface — just call `ckks_torch.convert(model)` |
| **GPU Acceleration** | CUDA-accelerated HE operations via OpenFHE |
| **Auto Optimization** | BatchNorm folding, BSGS matrix multiplication |
| **Wide Layer Support** | Linear, Conv2d, ReLU/GELU/SiLU, Pool, LayerNorm, Attention |

## Supported Layers

| Layer | Encrypted Version | Notes |
|-------|------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS optimization |
| `nn.Conv2d` | `EncryptedConv2d` | im2col method |
| `nn.ReLU/GELU/SiLU` | Polynomial approx | Or use `x²` for exact |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | Rotation-based |
| `nn.BatchNorm` | Folded | Merged into prev layer |
| `nn.LayerNorm` | `EncryptedLayerNorm` | Polynomial approx |
| `nn.Attention` | `EncryptedApproxAttention` | seq_len=1 |

<details>
<summary><strong>Full layer support table</strong></summary>

| PyTorch Layer | Encrypted Version | Notes |
|--------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | Full support with BSGS optimization |
| `nn.Linear` | `EncryptedTTLinear` | TT-decomposed for large layers |
| `nn.Conv2d` | `EncryptedConv2d` | Via im2col method |
| `nn.Conv2d` | `EncryptedTTConv2d` | TT-decomposed for large kernels |
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
| `nn.Dropout` | `EncryptedDropout` | No-op during inference |
| `nn.LayerNorm` | `EncryptedLayerNorm` | Pure HE polynomial approximation |
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | Polynomial softmax (seq_len=1) |

</details>

## Activation Functions

CKKS only supports polynomial operations. Choose one:

```python
# Option 1: Square activation (recommended - exact, no error)
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# Option 2: Polynomial approximation (closer to original ReLU/GELU)
enc_model, ctx = ckks_torch.convert(model, use_square_activation=False, activation_degree=4)
```

## GPU Acceleration

| Operation | Accelerated |
|-----------|-------------|
| Add/Sub/Mul/Square | ✅ GPU |
| Rotate/Rescale | ✅ GPU |
| Bootstrap | ✅ GPU |
| Encrypt/Decrypt | CPU |

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(poly_mod_degree=8192, scale_bits=40)
ctx = CKKSContext(config, enable_gpu=True)  # GPU enabled by default
```

## Examples

```bash
# Quick demo (no GPU required)
python -m ckks_torch.examples.encrypted_inference --demo conversion

# MNIST encrypted inference
python examples/mnist_encrypted.py --hidden 64 --samples 5
```

<details>
<summary><strong>CNN example</strong></summary>

```python
import torch.nn as nn
import ckks_torch

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 14 * 14, 10)
    
    def forward(self, x):
        return self.fc(self.flatten(self.pool1(self.act1(self.conv1(x)))))

model = MNISTCNN()
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

enc_input = ctx.encrypt(image)
prediction = ctx.decrypt(enc_model(enc_input)).argmax()
```

> **Note**: All operations in `forward()` must be layer attributes (e.g., `self.act1`), not inline operations like `x ** 2`.

</details>

<details>
<summary><strong>Batch processing</strong></summary>

```python
# Pack multiple samples into a single ciphertext (SIMD)
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)
enc_output = enc_model(enc_batch)
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

</details>

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `poly_mod_degree` (8192 instead of 16384) |
| Low Accuracy | Use `use_square_activation=True` or increase `activation_degree` |
| Slow Performance | Enable batch processing, reduce network depth |

## Documentation

- [API Reference](docs/api.md)
- [GPU Acceleration Guide](docs/gpu-acceleration.md)
- [CKKS Concepts](docs/concepts.md)

## License

Apache License 2.0

## Citation

```bibtex
@software{cukks,
  title = {CuKKS: PyTorch-compatible Encrypted Deep Learning},
  year = {2024},
  url = {https://github.com/devUuung/CuKKS}
}
```

## Related

### Libraries
- [OpenFHE](https://github.com/openfheorg/openfhe-development) — Underlying HE library
- [Microsoft SEAL](https://github.com/microsoft/SEAL) — Alternative HE library

### Papers
- [Homomorphic Encryption for Arithmetic of Approximate Numbers](https://eprint.iacr.org/2016/421) — Cheon et al. (CKKS)
- [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153) — Cheon et al.
- [Faster Homomorphic Linear Transformations in HElib](https://eprint.iacr.org/2018/244) — Halevi & Shoup (BSGS)
- [Tensor-Train Decomposition](https://epubs.siam.org/doi/10.1137/090752286) — Oseledets
- [Tensorizing Neural Networks](https://proceedings.neurips.cc/paper/5787-tensorizing-neural-networks) — Novikov et al.
- [Ultimate Tensorization: Compressing Convolutional and FC Layers Alike](https://arxiv.org/abs/1611.03214) — Garipov et al.
- [CNN Compression via Tensor-Train Decomposition on Permuted Weight Tensor](https://doi.org/10.1016/j.neunet.2020.07.028) — Gabor & Zdunek
