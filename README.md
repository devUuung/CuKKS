<p align="center">
  <a href="README.md">English</a> |
  <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">CuKKS</h1>

<p align="center">
  <strong>GPU-accelerated CKKS Homomorphic Encryption for PyTorch</strong>
</p>

<p align="center">
  <a href="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml"><img src="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml/badge.svg" alt="Build Wheels"></a>
  <a href="https://github.com/devUuung/CuKKS/actions/workflows/test-python.yml"><img src="https://github.com/devUuung/CuKKS/actions/workflows/test-python.yml/badge.svg" alt="Test Python"></a>
  <a href="https://github.com/devUuung/CuKKS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10--3.13-blue.svg" alt="Python 3.10-3.13"></a>
</p>

<p align="center">
  Run trained PyTorch models on <strong>encrypted data</strong> — no decryption needed, no privacy compromised.<br>
  Built on OpenFHE with CUDA acceleration.
</p>

---

## Why CuKKS?

Traditional machine learning requires access to raw input data — a privacy risk for sensitive domains like healthcare, finance, and biometrics. CuKKS lets you **deploy models that never see plaintext**:

```
User: encrypt(input) → [ciphertext] → Server: model([ciphertext]) → [encrypted output] → User: decrypt
```

The server performs full inference without ever decrypting the data. CuKKS makes this practical with:

- **One-line conversion** — `cukks.convert(model)` transforms any trained PyTorch model
- **GPU acceleration** — CUDA-accelerated HE operations via OpenFHE
- **37 layer types** — from Linear and Conv2d to Attention, GroupNorm, and ConvTranspose2d

## Quick Start

```python
import torch.nn as nn
import cukks

# 1. Train your model normally
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. Convert to encrypted model
enc_model, ctx = cukks.convert(model)

# 3. Run encrypted inference
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)
output = ctx.decrypt(enc_output)  # Same result, never decrypted on server
```

## Installation

```bash
pip install cukks[cu121]  # Match your PyTorch CUDA version: cu118, cu121, cu124, cu128
```

| Command | CUDA | Compute Capability |
|---------|------|--------------------|
| `pip install cukks[cu118]` | 11.8 | sm_50 – sm_90 (Maxwell ~ Hopper) |
| `pip install cukks[cu121]` | 12.1 | sm_50 – sm_90 (Maxwell ~ Hopper) |
| `pip install cukks[cu124]` | 12.4 | sm_50 – sm_90a (Maxwell ~ Hopper) |
| `pip install cukks[cu128]` | 12.8 | sm_50 – sm_100 (Maxwell ~ Blackwell) |

Not sure which CUDA version? Run `python -c "import torch; print(torch.version.cuda)"`.

<details>
<summary><strong>Docker, CLI tools, and building from source</strong></summary>

### Docker

```bash
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install cukks[cu121]
```

### Auto-install CLI

```bash
pip install cukks
cukks-install-backend  # Auto-detects PyTorch CUDA and installs matching backend
```

### Build from source

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

## Key Features

### Drop-in Model Conversion

No model rewriting. No custom HE code. Just call `cukks.convert(model)`:

```python
# Any PyTorch model — MLP, CNN, Transformer — converts automatically
enc_model, ctx = cukks.convert(model, activation_degree=4)
```

BatchNorm folding, BSGS matrix multiplication, and CNN optimizations are applied automatically.

### 37 Supported Layer Types

| Category | Layers |
|----------|--------|
| **Linear** | Linear, BlockDiagonalLinear, BlockDiagLowRankLinear |
| **Convolution** | Conv1d, Conv2d, ConvTranspose2d |
| **Pooling** | AvgPool2d, MaxPool2d, AdaptiveAvgPool2d |
| **Normalization** | LayerNorm, GroupNorm, InstanceNorm1d/2d, BatchNorm (folded) |
| **Activation** | ReLU, GELU, SiLU, Sigmoid, Tanh, Square |
| **Attention** | MultiheadAttention (seq_len ≤ 8) |
| **Embedding** | Embedding |
| **Spatial** | Upsample, PixelShuffle, PixelUnshuffle |
| **Padding** | ZeroPad2d, ConstantPad2d, ReflectionPad2d, ReplicationPad2d |
| **Other** | Flatten, Dropout, Sequential, ResidualBlock |

[Full layer table →](#supported-layers)

### GPU-Accelerated HE Operations

All core HE operations run on GPU:

| Operation | GPU |
|-----------|-----|
| Add / Sub / Mul / Square | ✅ |
| Rotate / Rescale | ✅ |
| Bootstrap | ✅ |
| Plaintext cache | ✅ |

### Polynomial Activation Approximation

CKKS only supports polynomial operations. CuKKS approximates non-polynomial activations (ReLU, GELU, SiLU, etc.) using Chebyshev polynomial fitting:

```python
# Default: degree-4 (good accuracy / depth balance)
enc_model, ctx = cukks.convert(model)

# Higher degree for better accuracy (costs more depth)
enc_model, ctx = cukks.convert(model, activation_degree=8)
```

### Packed Batch Inference

Process multiple samples in a single ciphertext:

```python
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)
enc_output = enc_model(enc_batch)
outputs = ctx.decrypt_batch(enc_output, sample_shape=(8,))
```

## Examples

```bash
# MNIST classification (MLP)
python examples/mnist_encrypted.py --hidden 64 --samples 5

# UNet-style segmentation (ConvTranspose2d, AdaptiveAvgPool2d, Upsample)
python examples/unet_encrypted.py --samples 2

# ResNet-style classification (GroupNorm, AdaptiveAvgPool2d)
python examples/resnet_encrypted.py --samples 1

# Transformer-style NLP (Embedding, LayerNorm)
python examples/transformer_encrypted.py --samples 2
```

See [examples/](examples/) for full scripts.

## Benchmarking

CuKKS includes a benchmark suite for measuring encrypted inference performance:

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Benchmark a specific model
python benchmarks/run_benchmarks.py --model mlp

# Save results to JSON
python benchmarks/run_benchmarks.py --output results.json
```

Supported models:

| Model | Params | Input | Architecture |
|-------|--------|-------|--------------|
| MLP | 50,890 | (1, 784) | Linear(784→64) → ReLU → Linear(64→10) |

Example output:

```
Model           Plain (ms)   Encrypted (ms)  Overhead   MAE       
--------------------------------------------------------------
mlp             0.01         82.12           6795      x 0.131713
```

> **Note:** Benchmarks require a GPU backend with OpenFHE. Run on a machine with CUDA support for accurate timing.

## Supported Layers

| PyTorch Layer | Encrypted Version | Notes |
|---------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS optimization |
| `nn.Conv1d` | `EncryptedConv1d` | 1D im2col |
| `nn.Conv2d` | `EncryptedConv2d` | im2col, BSGS |
| `nn.ConvTranspose2d` | `EncryptedConvTranspose2d` | Transposed convolution |
| `nn.ReLU` | `EncryptedReLU` | Polynomial approx |
| `nn.GELU` | `EncryptedGELU` | Polynomial approx |
| `nn.SiLU` | `EncryptedSiLU` | Polynomial approx |
| `nn.Sigmoid` | `EncryptedSigmoid` | Polynomial approx |
| `nn.Tanh` | `EncryptedTanh` | Polynomial approx |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | Rotation-based |
| `nn.MaxPool2d` | `EncryptedMaxPool2d` | Polynomial approx |
| `nn.AdaptiveAvgPool2d` | `EncryptedAdaptiveAvgPool2d` | Global-pool fast path |
| `nn.Flatten` | `EncryptedFlatten` | Logical reshape |
| `nn.BatchNorm1d/2d` | Folded | Merged into preceding layer |
| `nn.GroupNorm` | `EncryptedGroupNorm` | Per-group polynomial |
| `nn.InstanceNorm1d/2d` | `EncryptedInstanceNorm1d/2d` | Per-channel polynomial |
| `nn.LayerNorm` | `EncryptedLayerNorm` | Polynomial 1/sqrt |
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | seq_len ≤ 8 |
| `nn.Embedding` | `EncryptedEmbedding` | One-hot matmul |
| `nn.Upsample` | `EncryptedUpsample` | Nearest / bilinear |
| `nn.PixelShuffle` | `EncryptedPixelShuffle` | Channel-to-spatial |
| `nn.PixelUnshuffle` | `EncryptedPixelUnshuffle` | Spatial-to-channel |
| `nn.ZeroPad2d` | `EncryptedZeroPad2d` | Scatter matrix |
| `nn.ConstantPad2d` | `EncryptedConstantPad2d` | Scatter + constant |
| `nn.ReflectionPad2d` | `EncryptedReflectionPad2d` | Reflection mapping |
| `nn.ReplicationPad2d` | `EncryptedReplicationPad2d` | Replication mapping |
| `nn.Sequential` | `EncryptedSequential` | Full support |
| `nn.Dropout` | `EncryptedDropout` | No-op during inference |
| `nn.ResidualBlock` | `EncryptedResidualBlock` | Skip connection |

## Documentation

- [API Reference](docs/api.md)
- [CKKS Concepts](docs/concepts.md)
- [GPU Acceleration Guide](docs/gpu-acceleration.md)
- [STIP Packed Attention](docs/stip-attention.md)
- [Examples Overview](docs/examples/README.md)
- [한국어 문서](docs/ko/README.md)

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
- [GAZELLE: A Low Latency Framework for Secure Neural Network Inference](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar) — Juvekar et al. (Convolution)
- [PP-STAT: An Efficient Privacy-Preserving Statistical Analysis Framework using Homomorphic Encryption](https://doi.org/10.1145/3583780) — Choi (Encrypted Statistics)
- [STIP: Efficient and Secure Non-Interactive Transformer Inference via Compact Packing](https://doi.org/10.1145/3696410.3714779) — Wang et al. (Packed Attention)
- [Efficient Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2020/1203) — Bossuat et al. (Double-Hoisting)
- [On the Number of Nonscalar Multiplications Necessary to Evaluate Polynomials](https://doi.org/10.1137/0202007) — Paterson & Stockmeyer (Polynomial Evaluation)
