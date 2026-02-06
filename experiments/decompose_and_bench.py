"""Post-training BD+LR decomposition: benchmark accuracy and encrypted latency.

Takes the pre-trained dense model (block_size=256), decomposes fc2 into
BD + low-rank with various (block_size, rank) combinations, and measures:
1. Plaintext accuracy (no fine-tuning)
2. Encrypted latency + cosine similarity (via subprocess)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ckks_torch.nn.block_diagonal import BlockDiagonalLinear
from ckks_torch.nn.block_diagonal_low_rank import BlockDiagLowRankLinear

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class BlockDiagMNIST(nn.Module):
    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.act1 = SquareActivation()
        self.fc2 = BlockDiagonalLinear(hidden, hidden, block_size=block_size)
        self.act2 = SquareActivation()
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def load_dense_model(hidden: int) -> BlockDiagMNIST:
    ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", f"block_diag_h{hidden}_bs{hidden}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = BlockDiagMNIST(hidden, ckpt["block_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def decompose_fc2(model: BlockDiagMNIST, block_size: int, rank: int) -> nn.Module:
    """Replace fc2 with BD+LR decomposition of the original dense weight."""
    dense_linear = model.fc2.to_linear()
    bd_lr = BlockDiagLowRankLinear.from_dense(dense_linear, block_size, rank)

    class DecomposedModel(nn.Module):
        def __init__(self, fc1, fc2_bdlr, fc3):
            super().__init__()
            self.fc1 = fc1
            self.act1 = SquareActivation()
            self.fc2 = fc2_bdlr
            self.act2 = SquareActivation()
            self.fc3 = fc3

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))

    return DecomposedModel(model.fc1, bd_lr, model.fc3)


def measure_accuracy(model: nn.Module, data_dir: str) -> float:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def approx_error(model: BlockDiagMNIST, bd_lr: BlockDiagLowRankLinear) -> float:
    """Frobenius-norm relative error of the BD+LR approximation."""
    W_orig = model.fc2.to_dense_weight().detach()
    W_approx = bd_lr.to_dense_weight().detach()
    err = torch.norm(W_orig - W_approx) / torch.norm(W_orig)
    return float(err)


def run_encrypted_bench(hidden: int, block_size: int, rank: int, num_samples: int, device: str) -> dict:
    worker = os.path.join(SCRIPT_DIR, "_bench_bdlr_worker.py")
    cmd = [
        "/usr/bin/python", worker,
        "--hidden", str(hidden),
        "--block-size", str(block_size),
        "--rank", str(rank),
        "--num-samples", str(num_samples),
        "--device", device,
    ]
    env = os.environ.copy()
    env.setdefault(
        "LD_LIBRARY_PATH",
        "/workspace/ckks-torch/openfhe-gpu-public/build/lib:"
        "/workspace/ckks-torch/openfhe-gpu-public/build/_deps/rmm-build",
    )
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-300:]}", file=sys.stderr)
        return {"block_size": block_size, "rank": rank, "error": result.stderr.strip()[-200:]}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"block_size": block_size, "rank": rank, "error": "no RESULT_JSON"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-encrypted", action="store_true")
    args = parser.parse_args()

    combos = [
        (32, 0), (32, 4), (32, 8), (32, 16), (32, 32),
        (64, 0), (64, 4), (64, 8), (64, 16),
        (128, 0), (128, 8),
        (256, 0),  # rank=0 with bs=256 = dense truncation (no-op)
    ]

    print("Loading dense model...")
    dense_model = load_dense_model(args.hidden)
    orig_acc = measure_accuracy(dense_model, args.data_dir)
    print(f"Original dense accuracy: {orig_acc:.2f}%\n")

    results = []
    for bs, rank in combos:
        if args.hidden % bs != 0:
            continue
        print(f"block_size={bs}, rank={rank}:")
        decomposed = decompose_fc2(dense_model, bs, rank)
        bd_lr = decomposed.fc2
        acc = measure_accuracy(decomposed, args.data_dir)
        err = approx_error(dense_model, bd_lr)
        nz_diags = 2 * bs - 1 if bs < args.hidden else args.hidden
        print(f"  accuracy={acc:.2f}%  rel_error={err:.4f}  nz_diags={nz_diags}")

        r = {
            "block_size": bs, "rank": rank,
            "accuracy": acc, "orig_accuracy": orig_acc,
            "rel_frobenius_error": round(err, 4),
            "nonzero_diags_bd": nz_diags,
            "evalmult_bd": nz_diags, "evalmult_lr": 2 * rank,
            "evalmult_total": nz_diags + 2 * rank,
            "evalmult_dense": args.hidden,
        }

        if not args.skip_encrypted:
            enc_r = run_encrypted_bench(args.hidden, bs, rank, args.num_samples, args.device)
            if "error" not in enc_r:
                r["latency"] = enc_r.get("mean_latency_sec")
                r["cosine_sim"] = enc_r.get("mean_cosine_similarity")
                print(f"  latency={r['latency']}s  cos_sim={r['cosine_sim']}")
            else:
                r["enc_error"] = enc_r["error"]
                print(f"  ENC ERROR: {enc_r['error'][:80]}")

        results.append(r)

    print(f"\n{'='*100}")
    print(f"{'bs':>4} {'rank':>5} {'acc%':>7} {'err':>7} {'mults':>6} {'dense':>6} {'latency':>9} {'cos_sim':>10}")
    print(f"{'='*100}")
    for r in results:
        lat = f"{r.get('latency', '?'):>9}" if isinstance(r.get('latency'), (int, float)) else f"{'?':>9}"
        cos = f"{r.get('cosine_sim', '?'):>10}" if isinstance(r.get('cosine_sim'), (int, float)) else f"{'?':>10}"
        print(f"{r['block_size']:>4} {r['rank']:>5} {r['accuracy']:>6.2f}% {r['rel_frobenius_error']:>7.4f} "
              f"{r['evalmult_total']:>6} {r['evalmult_dense']:>6} {lat} {cos}")

    out_path = os.path.join(SCRIPT_DIR, "decompose_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
