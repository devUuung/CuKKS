#!/usr/bin/env python3
"""CuKKS Benchmark Suite.

Measures encrypted inference performance across different model architectures
and reports results in JSON format for CI integration.

Usage:
    python benchmarks/run_benchmarks.py --output results.json
    python benchmarks/run_benchmarks.py --model mlp --samples 10
    python benchmarks/run_benchmarks.py --list

Requires: GPU backend with OpenFHE. Falls back to mock mode for dry-run.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn

import cukks


@dataclass
class BenchmarkResult:
    model: str
    input_shape: List[int]
    num_params: int
    plaintext_ms: float
    encrypted_ms: float
    overhead_ratio: float
    accuracy_mae: float
    timestamp: str


def get_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class TinyMLP(nn.Module):
    """784 → 64 → 10 MLP for MNIST-style classification."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


class SmallCNN(nn.Module):
    """1×28×28 → Conv → Pool → FC for MNIST."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        return self.fc(self.flatten(x))


MODELS = {
    "mlp": lambda: TinyMLP(),
    "cnn": lambda: SmallCNN(),
}

INPUT_SHAPES = {
    "mlp": (1, 784),
    "cnn": (1, 1, 28, 28),
}


def benchmark_model(model_name: str, num_samples: int = 3) -> Optional[BenchmarkResult]:
    """Run plaintext + encrypted benchmark for a model."""
    model_fn = MODELS.get(model_name)
    if model_fn is None:
        print(f"Unknown model: {model_name}")
        return None

    model = model_fn()
    model.eval()
    input_shape = INPUT_SHAPES[model_name]
    num_params = sum(p.numel() for p in model.parameters())

    torch.manual_seed(42)
    sample = torch.randn(input_shape)

    with torch.no_grad():
        for _ in range(3):
            model(sample)
        t0 = time.perf_counter()
        for _ in range(num_samples):
            model(sample)
        plain_ms = (time.perf_counter() - t0) / num_samples * 1000

    backend_info = cukks.get_backend_info()
    if not backend_info["available"]:
        print(f"  [SKIP] Backend not available for {model_name}")
        return None

    try:
        enc_model, ctx = cukks.convert(model, activation_degree=3)
        enc_input = ctx.encrypt(sample.flatten())

        enc_model(enc_input)

        t0 = time.perf_counter()
        for _ in range(num_samples):
            enc_model(enc_input)
        enc_ms = (time.perf_counter() - t0) / num_samples * 1000

        plain_output = model(sample).flatten()
        enc_output = ctx.decrypt(enc_model(enc_input))
        mae = (plain_output - enc_output).abs().mean().item()

        overhead = enc_ms / plain_ms if plain_ms > 0 else float("inf")

        return BenchmarkResult(
            model=model_name,
            input_shape=list(input_shape),
            num_params=num_params,
            plaintext_ms=round(plain_ms, 2),
            encrypted_ms=round(enc_ms, 2),
            overhead_ratio=round(overhead, 1),
            accuracy_mae=round(mae, 6),
            timestamp=get_timestamp(),
        )

    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        return None


def run_all_benchmarks(num_samples: int = 3) -> List[BenchmarkResult]:
    """Run benchmarks for all models."""
    results = []
    for name in MODELS:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        result = benchmark_model(name, num_samples)
        if result:
            results.append(result)
            print(f"plain={result.plaintext_ms:.1f}ms, enc={result.encrypted_ms:.1f}ms, "
                  f"overhead={result.overhead_ratio:.0f}x, MAE={result.accuracy_mae:.6f}")
        else:
            print("skipped")
    return results


def main():
    parser = argparse.ArgumentParser(description="CuKKS Benchmark Suite")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Benchmark specific model")
    parser.add_argument("--samples", type=int, default=3, help="Number of inference runs to average")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, fn in MODELS.items():
            model = fn()
            params = sum(p.numel() for p in model.parameters())
            print(f"  {name:15s}  {params:>8,} params  input={INPUT_SHAPES[name]}")
        return

    if args.model:
        results = [benchmark_model(args.model, args.samples)]
        results = [r for r in results if r is not None]
    else:
        results = run_all_benchmarks(args.samples)

    if not results:
        print("No benchmarks completed.")
        return

    print(f"\n{'Model':<15} {'Plain (ms)':<12} {'Encrypted (ms)':<15} {'Overhead':<10} {'MAE':<10}")
    print("-" * 62)
    for r in results:
        print(f"{r.model:<15} {r.plaintext_ms:<12.2f} {r.encrypted_ms:<15.2f} "
              f"{r.overhead_ratio:<10.0f}x {r.accuracy_mae:<10.6f}")

    if args.output:
        data = [asdict(r) for r in results]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
