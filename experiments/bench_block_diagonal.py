"""Benchmark encrypted inference for block-diagonal MNIST models.

Loads trained checkpoints, converts to encrypted models, runs inference,
and measures latency + correctness. Each block_size runs in a subprocess
to avoid GPU state corruption between CKKS contexts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_single_benchmark(
    hidden: int,
    block_size: int,
    num_samples: int,
    device: str,
) -> dict:
    """Run benchmark for one block_size in a subprocess."""
    worker = os.path.join(SCRIPT_DIR, "_bench_worker.py")
    cmd = [
        "/usr/bin/python", worker,
        "--hidden", str(hidden),
        "--block-size", str(block_size),
        "--num-samples", str(num_samples),
        "--device", device,
    ]
    env = os.environ.copy()
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/workspace/ckks-torch/openfhe-gpu-public/build/lib:"
        "/workspace/ckks-torch/openfhe-gpu-public/build/_deps/rmm-build",
    )

    print(f"\n--- block_size={block_size} (subprocess) ---")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        return {
            "block_size": block_size,
            "error": result.stderr.strip()[-500:],
        }

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])

    print(f"STDOUT:\n{result.stdout}")
    return {"block_size": block_size, "error": "no RESULT_JSON found in output"}


def main():
    parser = argparse.ArgumentParser(description="Benchmark encrypted block-diagonal inference")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = []
    for bs in args.block_sizes:
        if args.hidden % bs != 0:
            print(f"SKIP block_size={bs}: hidden={args.hidden} not divisible")
            continue
        r = run_single_benchmark(args.hidden, bs, args.num_samples, args.device)
        results.append(r)
        if "error" not in r:
            print(f"  latency={r.get('mean_latency_sec', '?')}s  cosine_sim={r.get('mean_cosine_similarity', '?')}")

    print(f"\n{'='*90}")
    print(f"{'block_size':>10} {'latency_s':>10} {'cosine_sim':>11} {'nz_diags':>9} {'rotations':>10} {'mults':>6}")
    print(f"{'='*90}")
    for r in results:
        if "error" in r:
            print(f"{r['block_size']:>10} ERROR: {r['error'][:60]}")
        else:
            print(
                f"{r['block_size']:>10} "
                f"{r.get('mean_latency_sec', 0):>10.3f} "
                f"{r.get('mean_cosine_similarity', 0):>11.6f} "
                f"{r.get('nonzero_diagonals', '?'):>9} "
                f"{r.get('total_rotations', '?'):>10} "
                f"{r.get('total_evalmults', '?'):>6}"
            )

    out_path = os.path.join(SCRIPT_DIR, "bench_block_diagonal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
