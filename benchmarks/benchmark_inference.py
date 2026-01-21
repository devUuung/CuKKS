#!/usr/bin/env python3
"""Benchmark script for CKKS-Torch inference.

This script measures:
- Model conversion time
- Encryption/decryption time (mock and real)
- Layer-wise inference time
- Memory usage estimation

Usage:
    python benchmarks/benchmark_inference.py [--hidden 64] [--samples 10] [--real-backend]
"""

import argparse
import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import ckks_torch
from ckks_torch import convert, estimate_depth
from ckks_torch.nn import EncryptedSequential, EncryptedModule

# Try to import mock backend for testing without real HE
try:
    from tests.mocks.mock_backend import MockCKKSContext
    MOCK_AVAILABLE = True
except ImportError:
    try:
        from mocks.mock_backend import MockCKKSContext  # type: ignore[import-not-found]
        MOCK_AVAILABLE = True
    except ImportError:
        MockCKKSContext = None  # type: ignore[misc,assignment]
        MOCK_AVAILABLE = False


# =============================================================================
# Timing Utilities
# =============================================================================


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    result: Dict[str, float] = {}
    yield result
    result["elapsed"] = time.perf_counter() - start


def measure_time(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    gc.collect()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def warmup_func(func: Callable[..., Any], *args: Any, n: int = 2, **kwargs: Any) -> None:
    """Warm up a function before benchmarking."""
    for _ in range(n):
        func(*args, **kwargs)


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_usage_mb() -> Dict[str, float]:
    """Get current memory usage in MB."""
    result: Dict[str, float] = {}

    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        result["rss_mb"] = rusage.ru_maxrss / 1024
    except ImportError:
        pass

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    result["vm_rss_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("VmPeak:"):
                    result["vm_peak_mb"] = int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass

    if torch.cuda.is_available():
        try:
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        except Exception:
            pass

    return result


def estimate_ckks_memory_mb(
    poly_mod_degree: int,
    mult_depth: int,
    num_rotations: int,
) -> Dict[str, float]:
    """Estimate GPU memory for CKKS operations (rough heuristic).

    Based on typical CKKS memory patterns:
    - Each ciphertext: 2 polynomials * (mult_depth+2) limbs * poly_mod_degree * 8 bytes
    - Rotation keys: ciphertext size * limbs * num_rotations
    """
    limbs = mult_depth + 2
    ciphertext_mb = (2 * limbs * poly_mod_degree * 8) / (1024 * 1024)

    relin_key_mb = ciphertext_mb * limbs * 2
    rotation_key_mb = ciphertext_mb * limbs * num_rotations
    working_mb = ciphertext_mb * 10

    return {
        "ciphertext_mb": ciphertext_mb,
        "relin_key_mb": relin_key_mb,
        "rotation_keys_mb": rotation_key_mb,
        "working_mb": working_mb,
        "total_estimated_mb": ciphertext_mb + relin_key_mb + rotation_key_mb + working_mb,
    }


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    samples: int
    mean_ms: float
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    memory_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean_ms:.2f}ms, "
            f"std={self.std_ms:.2f}ms, "
            f"min={self.min_ms:.2f}ms, max={self.max_ms:.2f}ms"
        )


def compute_stats(times: List[float]) -> Tuple[float, float, float, float]:
    """Compute mean, std, min, max from a list of times in seconds."""
    if not times:
        return 0.0, 0.0, 0.0, 0.0

    times_ms = [t * 1000 for t in times]
    mean_ms = sum(times_ms) / len(times_ms)
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
    std_ms = variance ** 0.5
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return mean_ms, std_ms, min_ms, max_ms


# =============================================================================
# Mock Context for Benchmarking
# =============================================================================


class MockEncryptedContext:
    """Mock encrypted context for benchmarking without real HE backend."""

    def __init__(self) -> None:
        if not MOCK_AVAILABLE or MockCKKSContext is None:
            raise RuntimeError("Mock backend not available")
        self._ctx = MockCKKSContext()
        self.use_bsgs = False
        self._max_rotation_dim = 1024
        self._auto_bootstrap = False
        self._bootstrap_threshold = 2

    @property
    def auto_bootstrap(self) -> bool:
        return self._auto_bootstrap

    @property
    def bootstrap_threshold(self) -> int:
        return self._bootstrap_threshold

    def encrypt(self, tensor: torch.Tensor) -> Any:
        from ckks_torch.tensor import EncryptedTensor
        cipher = self._ctx.encrypt(tensor)
        return EncryptedTensor(cipher, tuple(tensor.shape), self)

    def decrypt(
        self, enc_tensor: Any, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        target_shape = shape if shape else enc_tensor.shape
        return self._ctx.decrypt(enc_tensor._cipher, shape=target_shape)


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_conversion(
    model: nn.Module,
    num_runs: int = 5,
) -> BenchmarkResult:
    """Benchmark model conversion time."""
    times: List[float] = []

    # Warmup
    _, _ = convert(model.eval(), use_square_activation=True)
    gc.collect()

    for _ in range(num_runs):
        gc.collect()
        start = time.perf_counter()
        _, _ = convert(model.eval(), use_square_activation=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_ms, std_ms, min_ms, max_ms = compute_stats(times)

    return BenchmarkResult(
        name="Model Conversion",
        samples=num_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
    )


def benchmark_encryption(
    ctx: Any,
    input_size: int,
    num_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark encryption time."""
    times: List[float] = []

    sample = torch.randn(input_size)

    # Warmup
    for _ in range(2):
        _ = ctx.encrypt(sample)
    gc.collect()

    for _ in range(num_runs):
        gc.collect()
        sample = torch.randn(input_size)
        start = time.perf_counter()
        _ = ctx.encrypt(sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_ms, std_ms, min_ms, max_ms = compute_stats(times)

    return BenchmarkResult(
        name=f"Encryption (size={input_size})",
        samples=num_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
    )


def benchmark_decryption(
    ctx: Any,
    input_size: int,
    num_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark decryption time."""
    times: List[float] = []

    sample = torch.randn(input_size)
    enc_sample = ctx.encrypt(sample)

    # Warmup
    for _ in range(2):
        _ = ctx.decrypt(enc_sample)
    gc.collect()

    for _ in range(num_runs):
        gc.collect()
        start = time.perf_counter()
        _ = ctx.decrypt(enc_sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_ms, std_ms, min_ms, max_ms = compute_stats(times)

    return BenchmarkResult(
        name=f"Decryption (size={input_size})",
        samples=num_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
    )


def benchmark_inference(
    enc_model: EncryptedModule,
    ctx: Any,
    input_size: int,
    num_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark full encrypted inference."""
    times: List[float] = []

    sample = torch.randn(input_size)

    # Warmup
    for _ in range(2):
        enc_input = ctx.encrypt(sample)
        _ = enc_model(enc_input)
    gc.collect()

    for _ in range(num_runs):
        gc.collect()
        sample = torch.randn(input_size)
        enc_input = ctx.encrypt(sample)

        start = time.perf_counter()
        enc_output = enc_model(enc_input)
        elapsed = time.perf_counter() - start

        _ = ctx.decrypt(enc_output)
        times.append(elapsed)

    mean_ms, std_ms, min_ms, max_ms = compute_stats(times)
    mem = get_memory_usage_mb()

    return BenchmarkResult(
        name=f"Full Inference (in={input_size})",
        samples=num_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        memory_mb=mem.get("vm_rss_mb", 0),
    )


def benchmark_layer_by_layer(
    enc_model: EncryptedModule,
    ctx: Any,
    input_size: int,
    num_runs: int = 5,
) -> List[BenchmarkResult]:
    """Benchmark each layer individually."""
    results: List[BenchmarkResult] = []

    if not isinstance(enc_model, EncryptedSequential):
        return results

    sample = torch.randn(input_size)
    enc_input = ctx.encrypt(sample)

    for idx, layer in enumerate(enc_model):
        layer_name = f"Layer {idx}: {layer.__class__.__name__}"
        times: List[float] = []

        current_input = ctx.encrypt(sample)

        for i, prev_layer in enumerate(enc_model):
            if i >= idx:
                break
            current_input = prev_layer(current_input)

        # Warmup
        for _ in range(2):
            _ = layer(current_input)

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = layer(current_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_ms, std_ms, min_ms, max_ms = compute_stats(times)
        results.append(BenchmarkResult(
            name=layer_name,
            samples=num_runs,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
        ))

    return results


# =============================================================================
# Main Benchmark Suite
# =============================================================================


def create_test_models(hidden_size: int) -> Dict[str, nn.Module]:
    """Create test models for benchmarking."""
    torch.manual_seed(42)

    models: Dict[str, nn.Module] = {}

    # Simple MLP (MNIST-like)
    models["Simple MLP (784->h->10)"] = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )

    # Deeper MLP
    models["Deep MLP (784->h->h->10)"] = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )

    # Very deep MLP
    models["Very Deep MLP (4 hidden)"] = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )

    return models


def run_benchmarks(
    hidden_size: int = 64,
    num_samples: int = 10,
    use_real_backend: bool = False,
) -> None:
    """Run the complete benchmark suite."""
    print("=" * 70)
    print("CKKS-Torch Inference Benchmarks")
    print("=" * 70)
    print()

    # System info
    print("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Backend available: {ckks_torch.is_available()}")
    print()

    # Memory info
    mem = get_memory_usage_mb()
    print("Initial Memory Usage:")
    for key, value in mem.items():
        print(f"  {key}: {value:.1f} MB")
    print()

    # Create context
    print("Creating benchmark context...")
    if use_real_backend and ckks_torch.is_available():
        print("  Using REAL CKKS backend")
        from ckks_torch import CKKSInferenceContext, InferenceConfig
        
        config = InferenceConfig(
            poly_mod_degree=32768,
            scale_bits=40,
            mult_depth=6,
        )
        ctx = CKKSInferenceContext(
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_rotation_dim=2048,
            use_bsgs=True,
        )
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    else:
        print("  Using MOCK backend (for timing comparison only)")
        if not MOCK_AVAILABLE:
            print("ERROR: Mock backend not available. Run from project root.")
            return
        ctx = MockEncryptedContext()
    print()

    # Create models
    models = create_test_models(hidden_size)
    print(f"Testing {len(models)} model architectures (hidden_size={hidden_size})")
    print()

    all_results: List[BenchmarkResult] = []

    for model_name, model in models.items():
        print("-" * 70)
        print(f"Model: {model_name}")
        print("-" * 70)

        model.eval()
        depth = estimate_depth(model)
        print(f"  Estimated multiplicative depth: {depth}")

        # Benchmark conversion
        print("\n  [1] Conversion Benchmark")
        conv_result = benchmark_conversion(model, num_runs=5)
        print(f"      {conv_result}")
        all_results.append(conv_result)

        # Convert model
        enc_model, _ = convert(model, use_square_activation=True)

        # Benchmark encryption
        print("\n  [2] Encryption Benchmark")
        enc_result = benchmark_encryption(ctx, input_size=784, num_runs=num_samples)
        print(f"      {enc_result}")
        all_results.append(enc_result)

        # Benchmark decryption
        print("\n  [3] Decryption Benchmark")
        dec_result = benchmark_decryption(ctx, input_size=10, num_runs=num_samples)
        print(f"      {dec_result}")
        all_results.append(dec_result)

        # Benchmark full inference
        print("\n  [4] Full Inference Benchmark")
        inf_result = benchmark_inference(enc_model, ctx, input_size=784, num_runs=num_samples)
        print(f"      {inf_result}")
        print(f"      Memory: {inf_result.memory_mb:.1f} MB")
        all_results.append(inf_result)

        # Layer-by-layer benchmark
        print("\n  [5] Layer-by-Layer Benchmark")
        layer_results = benchmark_layer_by_layer(enc_model, ctx, input_size=784, num_runs=5)
        for lr in layer_results:
            print(f"      {lr}")
            all_results.append(lr)

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print("Timing Overview (mock backend - relative comparison only):")
    print(f"  {'Benchmark':<40} {'Mean (ms)':<12} {'Std (ms)':<12}")
    print("  " + "-" * 64)
    for result in all_results:
        print(f"  {result.name:<40} {result.mean_ms:<12.2f} {result.std_ms:<12.2f}")

    # Memory estimation for real backend
    print()
    print("Memory Estimation (for real CKKS backend):")
    configs = [
        (8192, 4, 100),
        (16384, 6, 200),
        (32768, 10, 400),
    ]
    for poly_deg, depth, num_rots in configs:
        est = estimate_ckks_memory_mb(poly_deg, depth, num_rots)
        print(f"  poly_mod={poly_deg}, depth={depth}, rotations={num_rots}")
        print(f"    Estimated total: {est['total_estimated_mb']:.0f} MB")

    print()
    print("NOTE: Mock backend times are NOT representative of real CKKS performance.")
    print("      Use --real-backend flag to benchmark with actual homomorphic encryption.")


def main() -> None:
    """Entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="CKKS-Torch Inference Benchmarks")
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of benchmark samples (default: 10)",
    )
    parser.add_argument(
        "--real-backend",
        action="store_true",
        help="Use real CKKS backend (requires openfhe_backend)",
    )
    args = parser.parse_args()

    run_benchmarks(
        hidden_size=args.hidden,
        num_samples=args.samples,
        use_real_backend=args.real_backend,
    )


if __name__ == "__main__":
    main()
