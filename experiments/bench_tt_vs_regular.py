#!/usr/bin/env python
"""Paper benchmark: TT-decomposed vs regular linear layers on GPU CKKS.

Each (dim, mode) configuration runs in an isolated subprocess to avoid
GPU state contamination between OpenFHE context teardown/creation.

Usage:
    python experiments/bench_tt_vs_regular.py              # full sweep
    python experiments/bench_tt_vs_regular.py --quick       # smoke test
    python experiments/bench_tt_vs_regular.py --dims 256 512 --layers 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

POLY_MOD = 32768
SCALE_BITS = 50
SECURITY = "none"
DEVICE = "cuda"
DEPTH_BY_LAYERS = {1: 8, 2: 8, 3: 8}

DIMS = [32, 64, 128, 256, 512]
LAYERS = [1]

PYTHON = sys.executable


@dataclass
class RunResult:
    dim: int
    num_hidden: int
    mode: str
    latency_median_ms: float = 0.0
    latency_q1_ms: float = 0.0
    latency_q3_ms: float = 0.0
    cosine_sim: float = 0.0
    max_error: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    depth_consumed: int = 0
    num_layers_enc: int = 0
    tt_cores: int = 0
    tt_max_rank: int = 0
    feasible: bool = True
    error_msg: str = ""


WORKER_CODE = textwrap.dedent(r'''
import sys, json, time
sys.path.insert(0, "{repo_root}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from ckks_torch import CKKSInferenceContext, InferenceConfig, convert
from ckks_torch.context import compute_bsgs_rotations
from ckks_torch.nn import EncryptedTTLinear

dim = {dim}
num_hidden = {num_hidden}
tt = {tt}
warmup_trials = {warmup}
measure_trials = {trials}
mult_depth = {mult_depth}

config = InferenceConfig(
    poly_mod_degree={poly_mod}, scale_bits={scale_bits},
    mult_depth=mult_depth, security_level="{security}",
)
rots = compute_bsgs_rotations(dim)
neg = [-r for r in rots if r > 0]
rots = sorted(set(rots + neg))

ctx = CKKSInferenceContext(
    config=config, device="{device}",
    rotations=rots, use_bsgs=True, max_rotation_dim=dim,
)

layers_list = []
for _ in range(num_hidden):
    layers_list.append(nn.Linear(dim, dim))
    layers_list.append(nn.ReLU())
layers_list.append(nn.Linear(dim, dim))
model = nn.Sequential(*layers_list).eval()

with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            m.weight.mul_(0.5)
            m.bias.zero_()

enc_model, ctx = convert(model, ctx=ctx, tt=tt, use_square_activation=True)

tt_cores_count = 0
tt_max_rank = 0
for m in enc_model.modules():
    if isinstance(m, EncryptedTTLinear):
        tt_cores_count = len(m.tt_cores)
        ranks = [c.shape[2] for c in m.tt_cores[:-1]]
        tt_max_rank = max(ranks) if ranks else 1
        break

num_enc_layers = sum(1 for _ in enc_model.children())

torch.manual_seed(42)
x = torch.randn(dim, dtype=torch.float64) * 0.3

with torch.no_grad():
    h = x.clone()
    for m in model:
        if isinstance(m, nn.Linear):
            h = m(h.float()).to(torch.float64)
        elif isinstance(m, nn.ReLU):
            h = h ** 2
    plain_out = h

torch.cuda.reset_peak_memory_stats()

enc_x = ctx.encrypt(x)
for _ in range(warmup_trials):
    _ = enc_model(enc_x)

times = []
for _ in range(measure_trials):
    enc_input = ctx.encrypt(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc_out = enc_model(enc_input)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

enc_out_final = enc_model(enc_x)
dec = ctx.decrypt(enc_out_final)
dec_cpu = dec.cpu()[:dim]

cos = F.cosine_similarity(
    plain_out.flatten().unsqueeze(0), dec_cpu.flatten().unsqueeze(0),
).item()
max_err = (plain_out.flatten() - dec_cpu.flatten()).abs().max().item()
depth = enc_out_final._depth
mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

s = sorted(times)
n = len(s)
med = s[n // 2]
q1 = s[n // 4]
q3 = s[3 * n // 4]

print(json.dumps({{
    "latency_median_ms": med, "latency_q1_ms": q1, "latency_q3_ms": q3,
    "cosine_sim": cos, "max_error": max_err, "depth_consumed": depth,
    "gpu_mem_peak_mb": mem_mb, "num_layers_enc": num_enc_layers,
    "tt_cores": tt_cores_count, "tt_max_rank": tt_max_rank,
}}))
''')


def run_one(dim: int, num_hidden: int, tt: bool,
            warmup: int, trials: int) -> RunResult:
    mode = "tt" if tt else "regular"
    res = RunResult(dim=dim, num_hidden=num_hidden, mode=mode)

    repo_root = str(Path(__file__).resolve().parent.parent)
    mult_depth = DEPTH_BY_LAYERS.get(num_hidden, 16)

    code = WORKER_CODE.format(
        repo_root=repo_root, dim=dim, num_hidden=num_hidden,
        tt=tt, warmup=warmup, trials=trials, mult_depth=mult_depth,
        poly_mod=POLY_MOD, scale_bits=SCALE_BITS,
        security=SECURITY, device=DEVICE,
    )

    print(f"  [{mode:>7}] dim={dim}, hidden={num_hidden} ... ", end="", flush=True)

    try:
        proc = subprocess.run(
            [PYTHON, "-c", code],
            capture_output=True, text=True, timeout=600,
            env=os.environ,
        )
    except subprocess.TimeoutExpired:
        res.feasible = False
        res.error_msg = "TIMEOUT"
        print("TIMEOUT")
        return res

    if proc.returncode != 0:
        err_lines = proc.stderr.strip().split("\n")
        last_err = ""
        for line in reversed(err_lines):
            line = line.strip()
            if line and not line.startswith("Traceback") and not line.startswith("File"):
                last_err = line[:120]
                break
        res.feasible = False
        res.error_msg = last_err
        print(f"FAIL ({last_err[:80]})")
        return res

    json_lines = [l for l in proc.stdout.strip().split("\n") if l.startswith("{")]
    if not json_lines:
        res.feasible = False
        res.error_msg = "no output"
        print("FAIL (no output)")
        return res

    data = json.loads(json_lines[-1])
    res.latency_median_ms = data["latency_median_ms"]
    res.latency_q1_ms = data["latency_q1_ms"]
    res.latency_q3_ms = data["latency_q3_ms"]
    res.cosine_sim = data["cosine_sim"]
    res.max_error = data["max_error"]
    res.depth_consumed = data["depth_consumed"]
    res.gpu_mem_peak_mb = data["gpu_mem_peak_mb"]
    res.num_layers_enc = data["num_layers_enc"]
    res.tt_cores = data["tt_cores"]
    res.tt_max_rank = data["tt_max_rank"]

    print(f"OK  lat={res.latency_median_ms:.1f}ms  cos={res.cosine_sim:.6f}  "
          f"depth={res.depth_consumed}  mem={res.gpu_mem_peak_mb:.0f}MB")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="TT vs Regular CKKS benchmark")
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default="experiments/results")
    args = parser.parse_args()

    warmup = 3
    trials = 15

    dims = args.dims or (DIMS if not args.quick else [64, 256])
    layers = args.layers or (LAYERS if not args.quick else [1])

    if args.quick:
        warmup = 1
        trials = 3
    if args.trials is not None:
        trials = args.trials

    os.makedirs(args.out, exist_ok=True)

    import torch
    print("=" * 70)
    print("TT vs Regular Linear - GPU CKKS Benchmark")
    print("=" * 70)
    print(f"Device:     {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"poly_mod:   {POLY_MOD}")
    print(f"scale_bits: {SCALE_BITS}")
    print(f"mult_depth: {DEPTH_BY_LAYERS}")
    print(f"trials:     {trials} (warmup={warmup})")
    print(f"dims:       {dims}")
    print(f"layers:     {layers}")
    print(f"isolation:  subprocess per config")
    print()

    all_results: List[RunResult] = []

    for num_hidden in layers:
        print(f"\n{'='*70}")
        print(f"  {num_hidden}-hidden layer model")
        print(f"{'='*70}")
        for dim in dims:
            res_regular = run_one(dim, num_hidden, tt=False,
                                  warmup=warmup, trials=trials)
            res_tt = run_one(dim, num_hidden, tt=True,
                             warmup=warmup, trials=trials)
            all_results.append(res_regular)
            all_results.append(res_tt)

    json_path = Path(args.out) / "tt_vs_regular.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nJSON saved to {json_path}")

    csv_path = Path(args.out) / "tt_vs_regular.csv"
    if all_results:
        keys = list(asdict(all_results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
    print(f"CSV saved to {csv_path}")

    print(f"\n{'='*95}")
    print("SUMMARY")
    print(f"{'='*95}")
    print(f"{'dim':>5} {'hid':>3} {'mode':>7} {'lat(ms)':>9} {'IQR':>14} "
          f"{'cosine':>8} {'depth':>5} {'mem(MB)':>8} {'tt_info':>16} {'ok':>4}")
    print("-" * 95)

    for r in all_results:
        iqr = f"[{r.latency_q1_ms:.1f}-{r.latency_q3_ms:.1f}]"
        tt_info = f"{r.tt_cores}c/r{r.tt_max_rank}" if r.tt_cores > 0 else "-"
        ok = "Y" if r.feasible else "N"
        print(f"{r.dim:>5} {r.num_hidden:>3} {r.mode:>7} {r.latency_median_ms:>9.1f} "
              f"{iqr:>14} {r.cosine_sim:>8.4f} {r.depth_consumed:>5} "
              f"{r.gpu_mem_peak_mb:>8.0f} {tt_info:>16} {ok:>4}")

    print(f"\n{'='*70}")
    print("SPEEDUP (Regular / TT)")
    print(f"{'='*70}")
    print(f"{'dim':>5} {'hid':>3} {'regular(ms)':>12} {'tt(ms)':>12} "
          f"{'speedup':>8} {'cos_reg':>8} {'cos_tt':>8}")
    print("-" * 70)

    for i in range(0, len(all_results), 2):
        reg = all_results[i]
        tt = all_results[i + 1]
        if reg.feasible and tt.feasible and tt.latency_median_ms > 0:
            speedup = reg.latency_median_ms / tt.latency_median_ms
            print(f"{reg.dim:>5} {reg.num_hidden:>3} {reg.latency_median_ms:>12.1f} "
                  f"{tt.latency_median_ms:>12.1f} {speedup:>7.2f}x "
                  f"{reg.cosine_sim:>8.4f} {tt.cosine_sim:>8.4f}")
        else:
            note = ""
            if not reg.feasible:
                note = f"reg: {reg.error_msg[:30]}"
            elif not tt.feasible:
                note = f"tt: {tt.error_msg[:30]}"
            print(f"{reg.dim:>5} {reg.num_hidden:>3} {'N/A':>12} {'N/A':>12} "
                  f"{'N/A':>8} {note}")


if __name__ == "__main__":
    main()
