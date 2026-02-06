# References

Academic papers, algorithms, and software referenced by CuKKS.

---

## CKKS Encryption Scheme

**[1]** Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017).
*Homomorphic Encryption for Arithmetic of Approximate Numbers.*
ASIACRYPT 2017.
IACR ePrint: [2016/421](https://eprint.iacr.org/2016/421)

> Foundation of the CKKS scheme — approximate HE supporting real/complex arithmetic. Used throughout the library.

**[2]** Cheon, J. H., Han, K., Kim, A., Kim, M., & Song, Y. (2018).
*Bootstrapping for Approximate Homomorphic Encryption.*
EUROCRYPT 2018.
IACR ePrint: [2018/153](https://eprint.iacr.org/2018/153)

> CKKS bootstrapping for refreshing ciphertext depth. Used in `crypto_inv_sqrt` and bootstrapping examples.

---

## Linear Transformations & Matrix-Vector Multiplication

**[3]** Halevi, S. & Shoup, V. (2018).
*Faster Homomorphic Linear Transformations in HElib.*
CRYPTO 2018.
IACR ePrint: [2018/244](https://eprint.iacr.org/2018/244)

> Baby-step Giant-step (BSGS) algorithm for encrypted matrix-vector multiplication, reducing rotation count from O(n) to O(√n). Used in `EncryptedLinear` and `EncryptedConv2d`.

---

## Convolution

**[4]** Juvekar, C., Vaikuntanathan, V., & Chandrakasan, A. (2018).
*GAZELLE: A Low Latency Framework for Secure Neural Network Inference.*
USENIX Security 2018.

> Rotation-based convolution for HE. The current implementation uses im2col + BSGS matmul; Gazelle-style rotation convolution is a planned optimization to avoid OOM on large inputs.

---

## Statistical Functions (PP-STAT)

**[5]** Choi, H. (2025).
*PP-STAT: An Efficient Privacy-Preserving Statistical Analysis Framework.*
CIKM'25.

> Privacy-preserving encrypted statistical operations: mean, variance, standard deviation, inverse square root, and reciprocal. Used in `ckks_torch.stats` (normalization, crypto_inv_sqrt, crypto_reciprocal).

---

## Polynomial Approximation Algorithms

**Chebyshev Interpolation.**
Used to approximate non-polynomial functions (ReLU, GELU, SiLU, Sigmoid, Tanh, 1/x, 1/√x) with polynomials evaluable in CKKS. Implemented in `ckks_torch.utils.approximations`, `activations.py`, `layernorm.py`, `crypto_inv_sqrt.py`, `crypto_reciprocal.py`.

**Paterson-Stockmeyer Algorithm.**
Paterson, M. S. & Stockmeyer, L. J. (1973).
*On the Number of Nonscalar Multiplications Necessary to Evaluate Polynomials.*
SIAM Journal on Computing, 2(1), 60–62.

> Efficient polynomial evaluation reducing multiplicative depth from O(n) to O(√n). Referenced in `converter.py` depth estimation.

**Newton-Raphson Method.**
Used for iterative refinement of 1/√x after initial Chebyshev approximation. Implemented in `crypto_inv_sqrt.py`.

**Horner's Method.**
Standard polynomial evaluation scheme. Used as fallback in `poly_eval()` when Paterson-Stockmeyer is not available.

**Remez Algorithm.**
Optimal minimax polynomial approximation. Referenced in `ckks_torch.utils.approximations` as the ideal approach; current implementation uses Chebyshev as a practical approximation to minimax.

---

## Attention Mechanisms

**Taylor Series Approximation of exp(x).**
Used to approximate softmax for single-token attention (seq_len=1). Implemented in `attention.py`.

**Power-Softmax.**
Polynomial alternative to standard softmax using x^p normalization (p=2) for multi-token attention (seq_len>1). Avoids the need for exp(x) approximation. Implemented in `attention.py` with `crypto_reciprocal` for normalization.

---

## Software

**[11]** Al Badawi, A., et al. (2022).
*OpenFHE: Open-Source Fully Homomorphic Encryption Library.*
IACR ePrint: [2022/915](https://eprint.iacr.org/2022/915)

> Underlying HE library providing CKKS implementation with GPU acceleration (FLEXIBLEAUTO scaling). CuKKS builds on OpenFHE's C++ backend via Python bindings.

---

## Related Frameworks

- **[CrypTen](https://github.com/facebookresearch/CrypTen)** — Facebook's privacy-preserving ML framework using secure multi-party computation.
- **[TenSEAL](https://github.com/OpenMined/TenSEAL)** — Python library for HE operations on tensors, built on Microsoft SEAL.
- **[HELayers](https://github.com/IBM/helayers)** — IBM's HE layer library for encrypted neural network inference.
- **[Microsoft SEAL](https://github.com/microsoft/SEAL)** — Alternative HE library supporting BFV and CKKS schemes.
- **[HEAR](https://github.com/safednn-group/HEAR)** — Homomorphic Encryption for Accurate and Robust neural network inference.
