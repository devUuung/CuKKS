# CuKKS Benchmark Results

**Hardware**: NVIDIA GeForce RTX 5090 (32 GB VRAM)  
**Date**: 2026-04-04  
**Version**: v0.2.2

## Results

| Model | Params | Input Shape | Plain (ms) | Encrypted (ms) | Overhead | MAE |
|-------|--------|-------------|-----------|----------------|----------|-----|
| mlp | 50,890 | (1, 784) | 0.01 | 84.16 | 6,858x | 0.075594 |
| cnn | 15,770 | (1, 1, 28, 28) | 0.03 | 2,820.15 | 94,053x | 0.048294 |
| resnet | 1,300 | (1, 1, 8, 8) | 0.04 | 11,444.55 | 269,018x | 0.098026 |
| transformer | 212 | (1, 8) | 0.01 | 54.27 | 7,794x | 0.087978 |

## Model Architectures

| Model | Architecture |
|-------|-------------|
| **mlp** | Flatten → Linear(784→64) → ReLU → Linear(64→10) |
| **cnn** | Conv2d(1,8,3) → ReLU → AvgPool2d(2) → Flatten → Linear(1568→10) |
| **transformer** | Linear(8→16) → ReLU → Linear(16→4) — encrypted classifier stage |
| **resnet** | Conv2d(1→8, 3×3) → GroupNorm → ReLU → Conv2d(8→16, 1×1) → GroupNorm → ReLU → AdaptiveAvgPool2d(4×4) → Linear(256→4) |

## Notes

- **activation_degree=3** used for all models (Chebyshev polynomial approximation)
- `mlp`, `cnn`, and `transformer` were benchmarked at `poly_mod_degree=32768` (N=32768, 16384 slots)
- `resnet` was benchmarked at `poly_mod_degree=65536` (N=65536, 32768 slots) with `security_level=None` in the benchmark harness
- **resnet** is included as a baseline for GroupNorm and AdaptiveAvgPool2d performance on the currently supported packed-CNN benchmark architecture.

## Benchmark Commands

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Specific model
python benchmarks/run_benchmarks.py --model mlp

# Save to JSON
python benchmarks/run_benchmarks.py --output benchmarks/results.json

# List available models
python benchmarks/run_benchmarks.py --list
```

## GroupNorm Support

Models with `nn.GroupNorm` require deeper circuits and currently benchmark with
`poly_mod_degree=65536` in this suite. The benchmark harness constructs an
encrypted ResNet-style model that now runs end-to-end on the GPU path.
