# CKKS + Torch GPU backend

This package builds the `ckks_openfhe_backend` pybind11 extension against the local `openfhe-gpu-public` tree and exposes a small Torch-facing API in `ckks`.

## Building

```bash
# from repo root
export LD_LIBRARY_PATH="$PWD/openfhe-gpu-public/build/lib:$PWD/openfhe-gpu-public/build/_deps/rmm-build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install -e bindings/openfhe_backend
```

The CMake build links against CUDA (cublas/curand/cudart) plus the GPU OpenFHE libraries already built in `openfhe-gpu-public/build` (and the bundled `librmm`). RPATH is set so the extension can find those shared libs in-place.

## Quickstart

```python
import torch
from ckks import CKKSContext, CKKSConfig

config = CKKSConfig(
    poly_mod_degree=16384,  # set security_level="notset"/None to allow smaller rings
    coeff_mod_bits=(60, 40, 40, 60),
    scale_bits=40,
    security_level="128_classic",
    enable_bootstrap=False,
    # negative rotations are allowed (e.g., -1)
    rotations=[1, -1],
    relin=True,
)

ctx = CKKSContext(config)
plain = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")

cipher = ctx.encrypt(plain)
out = (cipher * 2.5).rotate(1).conjugate().rescale().decrypt(shape=plain.shape)

print(out)
```

Operations (`add`, `mul`, `rotate` (positive/negative), `conjugate`, `matmul_dense`, `matmul_diagonal`, `rescale`, `sum_slots`, `poly_eval`, and `bootstrap` when enabled) operate on encrypted data and keep the original tensor shape for easy round-trips.
`matmul_dense` uses the diagonal method internally; it supports rectangular matrices (m rows, n cols) as long as `n` does not exceed the slot count. You must provide rotation keys for shifts `1..n-1` in `CKKSConfig.rotations`.

### Bench

Run a quick GPU timing sweep (slots, depth) after setting `LD_LIBRARY_PATH`:

```bash
python bindings/openfhe_backend/examples/bench_gpu.py --slots 1024 --depth 3 --security notset --device cuda
```

### Encrypted MLP (toy)

End-to-end encrypted forward pass for a 1-layer MLP (square activation, square weight matrix):

```bash
python bindings/openfhe_backend/examples/encrypted_mlp.py --dim 4
```

It generates random weights/inputs, runs encrypted inference, and prints the decrypted output alongside a plaintext reference.

### Train then encrypt (iris/synthetic)

Train a tiny MLP with square activation on iris (or a synthetic fallback) and run encrypted inference on a few test samples:

```bash
python bindings/openfhe_backend/examples/encrypted_mlp_trained.py --samples 5 --device cpu
```

The script trains in plaintext, then reuses the learned weights in the CKKS pipeline (`matmul_dense` + square activation) and compares encrypted vs plaintext predictions.

Flags:
- `--activation {square,cheby_relu}` to switch to a Chebyshev ReLU approximation (requires numpy).
- `--cheby-degree` sets the polynomial degree.
