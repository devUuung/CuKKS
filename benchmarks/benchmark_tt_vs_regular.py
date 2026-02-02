#!/usr/bin/env python3
"""Benchmark: TT Decomposition vs Regular Linear Layer

Compares inference time between:
- Regular EncryptedLinear (full matrix multiply)
- EncryptedTTLinear (Tensor Train decomposed)

Usage:
    python benchmarks/benchmark_tt_vs_regular.py [--trials 5] [--real-backend]
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import ckks_torch
from ckks_torch import convert
from ckks_torch.nn import EncryptedTTLinear, EncryptedLinear


# =============================================================================
# Timing Utilities
# =============================================================================

def measure_time(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    gc.collect()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_inference(
    model: nn.Module,
    ctx: Any,
    input_data: torch.Tensor,
    trials: int = 5,
    warmup: int = 1,
) -> Dict[str, float]:
    """Benchmark encrypted inference."""
    
    # Encrypt input
    enc_input = ctx.encrypt(input_data)
    
    # Warmup
    for _ in range(warmup):
        _ = model(enc_input)
    
    # Benchmark
    times = []
    for _ in range(trials):
        gc.collect()
        start = time.perf_counter()
        _ = model(enc_input)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "trials": trials,
    }


# =============================================================================
# Benchmark Configurations
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for test model."""
    name: str
    in_features: int
    hidden_features: int
    out_features: int
    
    def create_model(self) -> nn.Module:
        """Create the PyTorch model."""
        return nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
        )


# Test configurations
CONFIGS = [
    ModelConfig("Small MLP", 256, 128, 64),
    ModelConfig("Medium MLP", 784, 256, 128),
    ModelConfig("Large MLP", 1024, 512, 256),
    ModelConfig("MNIST-like", 784, 128, 10),
]


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(
    config: ModelConfig,
    trials: int = 5,
    use_real_backend: bool = False,
) -> Dict[str, Any]:
    """Run TT vs Regular benchmark for a single configuration."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"  Input: {config.in_features} -> Hidden: {config.hidden_features} -> Output: {config.out_features}")
    print(f"{'='*60}")
    
    # Create base model
    model = config.create_model()
    model.eval()
    
    results: Dict[str, Any] = {
        "config": config.name,
        "in_features": config.in_features,
        "hidden_features": config.hidden_features,
        "out_features": config.out_features,
    }
    
    # Convert with tt=False (regular)
    print("\n[1] Converting with tt=False (Regular EncryptedLinear)...")
    enc_regular, ctx_regular = convert(model, tt=False, use_square_activation=True)
    
    # Get layer info
    layers_regular = list(enc_regular.children())
    regular_type = type(layers_regular[0]).__name__
    print(f"    First layer type: {regular_type}")
    
    # Convert with tt=True
    print("\n[2] Converting with tt=True (TT Decomposition)...")
    enc_tt, ctx_tt = convert(model, tt=True, use_square_activation=True)
    
    layers_tt = list(enc_tt.children())
    tt_layer = layers_tt[0]
    tt_type = type(tt_layer).__name__
    
    if isinstance(tt_layer, EncryptedTTLinear):
        num_cores = len(tt_layer.tt_cores)
        ranks = [c.shape[2] for c in tt_layer.tt_cores[:-1]]
        print(f"    First layer type: {tt_type}")
        print(f"    TT cores: {num_cores}, ranks: {ranks[:3]}..." if len(ranks) > 3 else f"    TT cores: {num_cores}, ranks: {ranks}")
        results["tt_num_cores"] = num_cores
        results["tt_max_rank"] = max(ranks) if ranks else 1
    else:
        print(f"    First layer type: {tt_type} (TT skipped - layer too small)")
        results["tt_num_cores"] = 0
        results["tt_max_rank"] = 0
    
    # Create test input
    input_data = torch.randn(config.in_features, dtype=torch.float64)
    
    # Benchmark Regular
    print(f"\n[3] Benchmarking Regular inference ({trials} trials)...")
    regular_results = benchmark_inference(enc_regular, ctx_regular, input_data, trials)
    print(f"    Mean: {regular_results['mean_ms']:.2f} ms")
    print(f"    Min:  {regular_results['min_ms']:.2f} ms")
    print(f"    Max:  {regular_results['max_ms']:.2f} ms")
    results["regular"] = regular_results
    
    # Benchmark TT
    print(f"\n[4] Benchmarking TT inference ({trials} trials)...")
    tt_results = benchmark_inference(enc_tt, ctx_tt, input_data, trials)
    print(f"    Mean: {tt_results['mean_ms']:.2f} ms")
    print(f"    Min:  {tt_results['min_ms']:.2f} ms")
    print(f"    Max:  {tt_results['max_ms']:.2f} ms")
    results["tt"] = tt_results
    
    # Calculate speedup
    speedup = regular_results['mean_ms'] / tt_results['mean_ms'] if tt_results['mean_ms'] > 0 else 0
    results["speedup"] = speedup
    
    print(f"\n[5] Comparison:")
    print(f"    Regular: {regular_results['mean_ms']:.2f} ms")
    print(f"    TT:      {tt_results['mean_ms']:.2f} ms")
    if speedup > 1:
        print(f"    TT is {speedup:.2f}x FASTER")
    elif speedup < 1 and speedup > 0:
        print(f"    TT is {1/speedup:.2f}x SLOWER")
    else:
        print(f"    Similar performance")
    
    return results


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print summary table of all results."""
    
    print("\n" + "="*80)
    print("SUMMARY: TT Decomposition vs Regular Linear")
    print("="*80)
    
    print(f"\n{'Config':<20} {'In->Hid->Out':<18} {'Regular (ms)':<14} {'TT (ms)':<14} {'Speedup':<10}")
    print("-"*80)
    
    for r in all_results:
        dims = f"{r['in_features']}->{r['hidden_features']}->{r['out_features']}"
        regular_ms = r['regular']['mean_ms']
        tt_ms = r['tt']['mean_ms']
        speedup = r['speedup']
        
        speedup_str = f"{speedup:.2f}x" if speedup != 0 else "N/A"
        if speedup > 1:
            speedup_str = f"+{speedup_str}"
        
        print(f"{r['config']:<20} {dims:<18} {regular_ms:<14.2f} {tt_ms:<14.2f} {speedup_str:<10}")
    
    print("-"*80)
    
    # Calculate average speedup
    speedups = [r['speedup'] for r in all_results if r['speedup'] > 0]
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        if avg_speedup > 1:
            print("TT decomposition is FASTER on average")
        else:
            print("Regular linear is FASTER on average")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TT vs Regular Linear")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per benchmark")
    parser.add_argument("--real-backend", action="store_true", help="Use real CKKS backend (requires GPU)")
    parser.add_argument("--config", type=str, default=None, help="Run specific config (Small, Medium, Large, MNIST)")
    args = parser.parse_args()
    
    print("="*80)
    print("TT Decomposition vs Regular Linear - Benchmark")
    print("="*80)
    print(f"Trials per test: {args.trials}")
    print(f"Backend: {'Real CKKS' if args.real_backend else 'Mock (CPU simulation)'}")
    
    # Select configs
    if args.config:
        configs = [c for c in CONFIGS if args.config.lower() in c.name.lower()]
        if not configs:
            print(f"Config '{args.config}' not found. Available: {[c.name for c in CONFIGS]}")
            return
    else:
        configs = CONFIGS
    
    # Run benchmarks
    all_results = []
    for config in configs:
        try:
            result = run_benchmark(config, args.trials, args.real_backend)
            all_results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {config.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
