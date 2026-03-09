# STIP: Packed Attention for CKKS

This document describes the STIP (Secure Transformer Inference Protocol) implementation in CuKKS, including packing algorithms, inverse-free LayerNorm, and practical layer limits.

## Table of Contents

- [Overview](#overview)
- [Packing Approaches](#packing-approaches)
- [Implemented Algorithms](#implemented-algorithms)
- [Practical Attention Layer Limits](#practical-attention-layer-limits)
- [Configuration Guide](#configuration-guide)
- [Current Status](#current-status)

---

## Overview

CuKKS supports two data packing strategies for encrypted inference:

| Strategy | Slot Layout | Best For |
|----------|------------|----------|
| **Row packing** (default) | `CT[i] = token[i]` full hidden vector | CNN, MLP, short sequences, batch inference |
| **Column packing** (STIP) | `CT[j] = feature[j]` across all tokens | Long-sequence Transformers, rotation reduction |

Both paths are fully supported. The converter auto-detects which LayerNorm variant to use based on the model's activation functions.

## Packing Approaches

### Row Packing (Default)

Each ciphertext holds one token's complete hidden vector:

```
CT₀ = [token₀_feat₀, token₀_feat₁, ..., token₀_feat_d]
CT₁ = [token₁_feat₀, token₁_feat₁, ..., token₁_feat_d]
```

- Matrix multiplication via BSGS diagonal method: **O(2√d) rotations**
- Batch SIMD: pack N samples into one ciphertext
- QK^T requires cross-token rotations: **O(seq_len) rotations**

### Column Packing (STIP)

Each ciphertext holds one feature dimension across all tokens:

```
CT₀ = [token₀_feat₀, token₁_feat₀, ..., token_n_feat₀]
CT₁ = [token₀_feat₁, token₁_feat₁, ..., token_n_feat₁]
```

- Matrix multiplication via element-wise multiply + sum: **0 rotations**
- QK^T via RIHP halved inner product: **O(seq_len/2) rotations**
- Complex-slot utilization: RIHP packs 2 diagonals per ciphertext, DHP processes 2 heads simultaneously

## Implemented Algorithms

### RIHP (Real-Imaginary Hybrid Packing)

Packs two diagonals into one ciphertext using complex slots:

```
k_hybrid[j] = k[j] + i · RotL(k[j], m/2)
```

- Halves the number of ciphertexts needed for CCMM
- Unpacking: `P[r] = Re(C[r])`, `P[r+m/2] = Im(C[r])`
- Module: `cukks.batching.RIHPacker`

### DHP (Dual-Head Packing)

Packs adjacent attention heads into complex slots:

```
W̃ = W[h] + i·W[h+1]
```

- One PCMM processes two heads simultaneously
- After attention: repack for output projection
- Module: `cukks.batching.DHPacker`

### AMCP (Adaptive Multi-Column Packing)

Packs k feature columns into one ciphertext with interleaved batching:

```
k = max{κ ∈ Divisors(H) | κ·t·m ≤ S}
```

- Reduces ciphertext count by factor k
- Uses iterative group-wise rotation for alignment
- Module: `cukks.batching.AMCPacker`

### Inverse-Free LayerNorm

For models with homogeneous activations (ReLU), uses a depth-efficient LN:

```
ŷ_i = γ · z_i · √(nλ) + β · v    where v = √(λ · Σz_i²)
```

- **5 multiplicative levels** (vs 18 for standard LN)
- σ propagation through subsequent layers enables bias scaling
- Auto-detected via `torch.fx` graph tracing at conversion time
- Module: `cukks.nn.EncryptedInverseFreeLayerNorm`

**Eligibility**: A LayerNorm qualifies for inverse-free mode when the path to the next LayerNorm contains only linear operations and homogeneous activations (ReLU, Dropout, Identity). Non-homogeneous activations (GELU, SiLU, Sigmoid, Tanh) trigger fallback to standard LN.

## Practical Attention Layer Limits

> **Key insight**: Multiplicative depth is NOT the limiting factor — bootstrapping solves depth. **Precision (approximation error accumulation) is the real bottleneck.**

### Error Sources Per Block

| Component | Error Type | Magnitude |
|-----------|-----------|-----------|
| ReLU degree-4 | Absolute | ~0.23 |
| ReLU degree-8 | Absolute | ~0.13 |
| GELU degree-4 | Absolute | ~5×10⁻⁴ |
| GELU degree-8 | Absolute | ~2×10⁻⁴ |
| Softmax 1/x (degree-15) | Relative | ~6×10⁻¹⁰ |
| Inv-Free LN Taylor √ | Relative | ~10⁻³ |
| Standard LN 1/√x (degree-15) | Relative | ~0.28 (self-correcting) |

### Practical Layer Limits (cosine similarity > 0.99)

| Configuration | Activation | LayerNorm | Levels/Block | Practical Layers | Dominant Error |
|--------------|-----------|-----------|-------------|-----------------|----------------|
| ReLU deg-4 | `EncryptedReLU(degree=4)` | Inverse-Free (auto) | ~15 | **2–3** | ReLU approximation |
| ReLU deg-8 | `EncryptedReLU(degree=8)` | Inverse-Free (auto) | ~18 | **8–10** | ReLU approximation |
| GELU deg-4 | `EncryptedGELU(degree=4)` | Standard (auto) | ~28 | **12+** | Accumulated rounding |
| GELU deg-8 | `EncryptedGELU(degree=8)` | Standard (auto) | ~31 | **20+** | Accumulated rounding |

### Why ReLU Limits Are Lower

ReLU has a sharp kink at x=0 that polynomial approximation cannot capture well:
- Degree-4: large absolute error (~0.23) near the kink
- Degree-8: better but still ~0.13
- Error compounds multiplicatively through layers

GELU is inherently smooth, making polynomial approximation much more accurate (~5×10⁻⁴ even at degree-4).

### Recommendations

| Model Type | Recommended Config | Expected Layers |
|-----------|-------------------|----------------|
| Lightweight / edge | ReLU deg-8 | Up to 8 layers |
| BERT-base / GPT-small | GELU deg-4 | 12 layers |
| Large Transformers | GELU deg-8 | 20+ layers |

## Configuration Guide

### Basic Usage (Auto-Detection)

```python
import cukks

# GELU model → Standard LN auto-selected
model = MyGELUTransformer()
enc_model, ctx = cukks.convert(model, activation_degree=4)

# ReLU model → Inverse-Free LN auto-selected
model = MyReLUTransformer()
enc_model, ctx = cukks.convert(model, activation_degree=8)
```

### Column-Wise Attention (STIP Path)

```python
from cukks.batching import RIHPacker, DHPacker, AMCPacker

# RIHP: halved rotation count for attention
rihp = RIHPacker(half_seq_len=seq_len // 2)

# DHP: process 2 heads simultaneously
dhp = DHPacker(head_dim=64, num_heads=8)

# AMCP: pack multiple columns into one ciphertext
amcp = AMCPacker(
    num_slots=16384,
    seq_len=seq_len,
    hidden_dim=768,
    batch_size=1,
)
```

### Depth Budget Planning

```
Per Transformer block:
  QKV projection:        1 level  (matmul)
  Attention scores:      1 level  (Q·K^T)
  Softmax:              10 levels (degree-15 polynomial)
  Value weighting:       1 level  (weights·V)
  Output projection:     1 level  (matmul)
  FFN (2 Linear):        2 levels (matmul × 2)
  Activation:           2-4 levels (degree 4-8)
  LayerNorm (×2):       10-36 levels (standard) or 4-10 levels (inv-free)
  ─────────────────────
  Total:               ~15-31 levels per block (bootstrapping resets)
```

## Current Status

### Implemented ✅

| Component | Module | Tests |
|-----------|--------|-------|
| RIHP packer | `cukks.batching.RIHPacker` | 4 |
| DHP packer | `cukks.batching.DHPacker` | 7 |
| AMCP packer | `cukks.batching.AMCPacker` | 15 |
| Inverse-Free LN | `cukks.nn.EncryptedInverseFreeLayerNorm` | 33 |
| σ propagation (bias scaling) | `cukks.nn.EncryptedLinear` | 7 |
| Complex-slot ops | `cukks.tensor.EncryptedTensor` | — |
| Column-wise attention | `cukks.nn.EncryptedApproxAttention` | 7 |
| torch.fx LN detection | `cukks.analysis.inverse_free_detect` | — |
| **Total** | | **531 tests passing** |

### Planned Enhancements

| Enhancement | Impact | Status |
|------------|--------|--------|
| Hoisted automorphisms (BSGS) | 30–75× rotation speedup | Not started |
| GK constant normalization | Remove softmax 1/x error source | Not started |
| Composite polynomial ReLU | Extend ReLU model layer limit | Not started |
| AMCP → column-wise attention wiring | Reduce ciphertext count | Not started |
| Lazy rescaling in BSGS | Save 1 level per matmul | Not started |

## References

- Choi, H. et al. — STIP: Secure Transformer Inference Protocol (RIHP, DHP, AMCP algorithms)
- Halevi, S. & Shoup, V. — Faster Homomorphic Linear Transformations in HElib (BSGS, 2018)
- Bossuat, J.-P. et al. — Efficient Bootstrapping for Approximate HE (Double-hoisting, BMPH21)
