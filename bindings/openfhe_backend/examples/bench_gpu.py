"""
Quick GPU CKKS smoke/bench.

Usage:
  python examples/bench_gpu.py --slots 1024 --depth 3 --security notset
"""

from __future__ import annotations

import argparse
import time

import torch

from ckks import CKKSConfig, CKKSContext


def run_once(slots: int, depth: int, security: str | int | None, device: str) -> float:
    cfg = CKKSConfig(
        poly_mod_degree=max(16384, 2 * slots),
        coeff_mod_bits=(60,) + (40,) * depth + (60,),
        scale_bits=40,
        security_level=security,
        rotations=[1, -1],
        relin=True,
    )
    ctx = CKKSContext(cfg, device=device)
    torch_device = torch.device(device)

    plain = torch.randn(slots, device=device, dtype=torch.float64)
    weights = torch.randn(slots, device=device, dtype=torch.float64)

    cipher = ctx.encrypt(plain)
    if torch.cuda.is_available() and torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    t0 = time.time()

    # simple fused op chain
    out = cipher.mul(weights.cpu()).add(cipher).rotate(1).rescale()
    for _ in range(depth):
        out = out.mul(1.1).rescale()

    if torch.cuda.is_available() and torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.time() - t0
    # Force decrypt to validate plumbing but don't time it.
    _ = out.decrypt(shape=(slots,), device="cpu")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=1024, help="number of plaintext slots to use")
    parser.add_argument("--depth", type=int, default=3, help="number of mult/rescale steps")
    parser.add_argument(
        "--security",
        type=str,
        default="128_classic",
        help="CKKS security level (e.g., 128_classic, 192_classic, 256_classic, 128_quantum, 192_quantum, 256_quantum, notset)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    elapsed = run_once(args.slots, args.depth, args.security, args.device)
    print(f"slots={args.slots} depth={args.depth} device={args.device} time={elapsed:.3f}s")


if __name__ == "__main__":
    main()
