# CKKS Concepts

This guide explains the key concepts behind CKKS homomorphic encryption and how they apply to encrypted deep learning inference.

## Table of Contents

- [What is Homomorphic Encryption?](#what-is-homomorphic-encryption)
- [The CKKS Scheme](#the-ckks-scheme)
- [Key Parameters](#key-parameters)
- [Multiplicative Depth](#multiplicative-depth)
- [Rescaling](#rescaling)
- [Bootstrapping](#bootstrapping)
- [Polynomial Approximations](#polynomial-approximations)
- [SIMD and Slot Packing](#simd-and-slot-packing)
- [Practical Considerations](#practical-considerations)

---

## What is Homomorphic Encryption?

**Homomorphic Encryption (HE)** allows computation on encrypted data without decrypting it first. The result, when decrypted, matches what you would get from computing on the plaintext.

```
Encrypt(a) + Encrypt(b) = Encrypt(a + b)
Encrypt(a) × Encrypt(b) = Encrypt(a × b)
```

### Why Use HE for Machine Learning?

- **Privacy-Preserving Inference**: A client can send encrypted data to a server, which runs inference without ever seeing the raw data.
- **Secure Outsourcing**: Process sensitive data (medical, financial) on untrusted cloud infrastructure.
- **Regulatory Compliance**: Meet data protection requirements (GDPR, HIPAA) while still enabling ML services.

### Types of Homomorphic Encryption

| Type | Operations | Use Case |
|------|------------|----------|
| **Partially HE** | Only addition OR multiplication | Limited |
| **Somewhat HE** | Both, but limited depth | Simple circuits |
| **Leveled HE** | Both, up to a fixed depth | Deep learning (CKKS) |
| **Fully HE** | Unlimited operations via bootstrapping | Any computation |

CKKS is a **leveled** scheme that can become **fully** homomorphic with bootstrapping.

---

## The CKKS Scheme

**CKKS** (Cheon-Kim-Kim-Song, 2017) is designed specifically for approximate arithmetic on real/complex numbers—perfect for machine learning.

### Key Properties

1. **Approximate Arithmetic**: Unlike exact HE schemes, CKKS allows small errors in exchange for efficiency. This is acceptable for ML where small numerical errors don't affect predictions.

2. **SIMD Operations**: Pack many values into one ciphertext and operate on all of them in parallel (Single Instruction, Multiple Data).

3. **Efficient for Neural Networks**: Supports the core operations needed: addition, multiplication, and rotation (for matrix operations).

### How CKKS Works (Simplified)

```
Plaintext: [x₁, x₂, ..., xₙ]  (vector of real numbers)
    ↓ Encode
Polynomial: m(X) in ℤ[X]/(Xⁿ + 1)
    ↓ Encrypt (add noise)
Ciphertext: (c₀, c₁) - pair of polynomials
```

The security comes from the **Ring Learning with Errors (RLWE)** problem—it's computationally hard to recover the plaintext from the ciphertext without the secret key.

---

## Key Parameters

### Ring Dimension (poly_mod_degree)

The polynomial ring dimension `N` (must be a power of 2).

| N | Slots | Security | Speed | Memory |
|---|-------|----------|-------|--------|
| 8192 | 4096 | ~128-bit | Fast | Low |
| 16384 | 8192 | ~128-bit | Medium | Medium |
| 32768 | 16384 | ~128-bit | Slow | High |
| 65536 | 32768 | ~128-bit | Very Slow | Very High |

**Trade-offs:**
- Larger N → more slots for SIMD, deeper circuits possible
- Larger N → slower operations, more memory

**Recommendation:** Start with 16384 for most neural networks.

### Scale (scale_bits)

The precision of encoded values, typically 30-50 bits.

- **Higher scale** → more precision, but fewer multiplication levels
- **Lower scale** → less precision, but more levels available

**Recommendation:** Use 40 bits for a good balance.

### Security Level

Based on the hardness of the underlying RLWE problem.

| Level | Meaning | Use Case |
|-------|---------|----------|
| 128-bit | Standard security | Most applications |
| 192-bit | High security | Government, financial |
| 256-bit | Very high security | Long-term secrets |

The library automatically sets coefficient modulus sizes based on security level.

### Coefficient Modulus

A chain of prime moduli: `[q₀, q₁, ..., qₗ]`

- **First and last** primes: Larger (e.g., 60 bits) for key switching
- **Middle primes**: Match the scale (e.g., 40 bits each)
- **Number of primes** determines multiplicative depth

```python
# Example: depth=4 with 40-bit scale
coeff_mod_bits = (60, 40, 40, 40, 40, 60)
#                 ↑   ↑-----------↑   ↑
#                 key  4 mult levels  key
```

---

## Multiplicative Depth

The **multiplicative depth** is the maximum number of sequential multiplications a circuit can perform before running out of "budget."

### Why Depth Matters

Each multiplication:
1. Increases noise in the ciphertext
2. Squares the scale (scale → scale²)

After enough multiplications, noise overwhelms the signal, making decryption impossible.

### Depth Budget

```
Available depth = number of middle primes in coefficient modulus
```

Each multiplication consumes one level. When levels run out, you must:
- Reduce circuit depth
- Enable bootstrapping (expensive but refreshes levels)

### Depth in Neural Networks

| Operation | Depth Cost |
|-----------|------------|
| Linear (matmul) | 1 |
| ReLU (degree-4 poly) | 2-3 |
| Square activation (x²) | 1 |
| Convolution | 1 |
| Average pooling | 0-1 |

**Example: Simple MLP**
```
Input → Linear(1) → ReLU(2) → Linear(1) → ReLU(2) → Linear(1)
Total depth: 1 + 2 + 1 + 2 + 1 = 7
```

### Estimating Depth

```python
from ckks_torch import estimate_depth

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

depth = estimate_depth(model)  # Returns ~4-5
```

---

## Rescaling

After multiplication, the scale doubles (scale × scale = scale²). **Rescaling** reduces the scale back to the original level.

### How Rescaling Works

```
Before: ciphertext with scale = Δ²
After rescale: ciphertext with scale ≈ Δ

This consumes one level from the coefficient modulus chain.
```

### When to Rescale

- After every multiplication (cipher × cipher)
- After polynomial evaluation
- The library handles this automatically in most cases

### Example

```python
# Multiplication increases scale
result = enc_a.mul(enc_b)  # scale: Δ → Δ²

# Rescale to manage precision
result = result.rescale()  # scale: Δ² → Δ, consumes 1 level
```

---

## Bootstrapping

**Bootstrapping** is a technique to "refresh" a ciphertext, restoring its multiplicative depth. This enables unlimited computation depth.

### How It Works

Bootstrapping homomorphically evaluates the decryption function, essentially re-encrypting the ciphertext with fresh noise and full levels.

```
Low-level ciphertext → Bootstrap → Fresh ciphertext with full levels
```

### Cost of Bootstrapping

- **Very expensive**: 10-100x slower than regular operations
- **Consumes depth itself**: Requires ~10-15 levels
- **Approximation error**: Introduces additional noise

### When to Use Bootstrapping

| Network Depth | Recommendation |
|---------------|----------------|
| 1-4 layers | No bootstrapping needed |
| 5-8 layers | Consider larger ring dimension instead |
| 9+ layers | Bootstrapping may be necessary |

### Auto-Bootstrapping in CuKKS

```python
ctx = CKKSInferenceContext(
    config=config,
    auto_bootstrap=True,      # Enable automatic bootstrapping
    bootstrap_threshold=2,    # Bootstrap when 2 levels remain
)
```

---

## Polynomial Approximations

CKKS only supports **polynomial operations** (addition and multiplication). Non-polynomial functions like ReLU, sigmoid, and tanh must be approximated.

### Approximation Methods

#### 1. Chebyshev Polynomials

Best for smooth functions over a bounded interval.

```python
# ReLU approximation on [-1, 1] using degree-7 Chebyshev polynomial
from ckks_torch.utils.approximations import chebyshev_coefficients

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

coeffs = chebyshev_coefficients(relu, degree=7, domain=(-1, 1))
```

#### 2. Minimax Polynomials

Minimize the maximum error over the interval.

```python
from ckks_torch.nn import EncryptedReLU

relu = EncryptedReLU(degree=4, method="minimax")
```

#### 3. Taylor Series

Good for functions with known derivatives, but only accurate near expansion point.

### Polynomial Degree Trade-offs

| Degree | Accuracy | Depth Cost | Speed |
|--------|----------|------------|-------|
| 2 | Low | 1 | Fast |
| 4 | Medium | 2 | Medium |
| 7 | High | 3 | Slow |
| 15 | Very High | 4 | Very Slow |

### Square Activation

The simplest polynomial activation: `f(x) = x²`

**Advantages:**
- Exact in CKKS (no approximation error)
- Only 1 multiplication depth
- Very fast

**Disadvantages:**
- Different behavior than ReLU (always positive, quadratic growth)
- Model must be trained with square activation for best results

```python
# Use square activation
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)
```

### Input Normalization

Polynomial approximations are accurate only within their defined domain (typically [-1, 1] or [-4, 4]).

**Important:** Normalize your inputs before encryption!

```python
# Normalize inputs to [-1, 1]
normalized_input = (input - input.mean()) / input.std()
normalized_input = torch.clamp(normalized_input, -1, 1)
```

---

## SIMD and Slot Packing

CKKS supports **SIMD (Single Instruction, Multiple Data)** operations through slot packing.

### What Are Slots?

Each ciphertext has `N/2` slots (where N is the ring dimension). Each slot holds one complex number (or real number for our purposes).

```
Ring dimension N = 16384
Number of slots = 16384 / 2 = 8192
```

### Slot Packing

Pack multiple values into different slots of the same ciphertext:

```
Ciphertext slots: [x₀, x₁, x₂, ..., x₈₁₉₁]
```

Operations are applied **element-wise** across all slots simultaneously:

```
Add:      [a₀, a₁, ...] + [b₀, b₁, ...] = [a₀+b₀, a₁+b₁, ...]
Multiply: [a₀, a₁, ...] × [b₀, b₁, ...] = [a₀×b₀, a₁×b₁, ...]
```

### Rotation

Rotation shifts elements across slots:

```
rotate([x₀, x₁, x₂, x₃], steps=1) = [x₁, x₂, x₃, x₀]
```

Rotation is essential for:
- Matrix-vector multiplication
- Convolution
- Reduction operations (sum, max)

### Batch Processing

Pack multiple samples into one ciphertext for parallel inference:

```python
# Pack 8 samples into one ciphertext
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)

# All 8 inferences happen in parallel
enc_output = enc_model(enc_batch)

# Unpack results
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

---

## Practical Considerations

### Memory Requirements

CKKS operations are memory-intensive:

| Component | Approximate Size |
|-----------|------------------|
| One ciphertext (N=16384, L=6) | ~2 MB |
| Relinearization key | ~50-100 MB |
| Rotation keys (1000 rotations) | ~2-5 GB |
| Working memory | 2-3x ciphertext size |

**Tip:** Reduce rotation key count using Baby-step Giant-step algorithm (enabled by default).

### Performance Tips

1. **Minimize depth**: Use square activation when possible
2. **Use BSGS**: Reduces rotation keys from O(n) to O(√n)
3. **Batch samples**: Pack multiple inputs for SIMD parallelism
4. **Fold BatchNorm**: Merge into preceding linear/conv layers
5. **Choose right ring dimension**: Larger isn't always better

### Error Sources

1. **Encoding error**: Converting float to polynomial representation
2. **Encryption noise**: Random noise added for security
3. **Rescaling error**: Rounding during scale reduction
4. **Approximation error**: Polynomial approximation of activations

**Typical accuracy loss:** 0.1-2% compared to plaintext inference.

### Security Considerations

- **Never reuse keys**: Generate fresh keys for each deployment
- **Parameter selection**: Use recommended security levels (128-bit minimum)
- **Side channels**: Be aware of timing attacks in deployment
- **Key management**: Protect secret keys as you would passwords

---

## Further Reading

### Papers

1. **CKKS Original Paper**: [Homomorphic Encryption for Arithmetic of Approximate Numbers](https://eprint.iacr.org/2016/421) (Cheon et al., 2017)

2. **Bootstrapping for CKKS**: [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153)

3. **CryptoNets**: [CryptoNets: Applying Neural Networks to Encrypted Data](https://proceedings.mlr.press/v48/gilad-bachrach16.html)

### Libraries

- [OpenFHE](https://github.com/openfheorg/openfhe-development): The backend used by CuKKS
- [Microsoft SEAL](https://github.com/microsoft/SEAL): Alternative HE library
- [TenSEAL](https://github.com/OpenMined/TenSEAL): Python bindings for SEAL

### Tutorials

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [HomomorphicEncryption.org](https://homomorphicencryption.org/)
