"""Subprocess worker for BD+LR encrypted benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cukks.nn.block_diagonal import BlockDiagonalLinear
from cukks.nn.block_diagonal_low_rank import BlockDiagLowRankLinear


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


class ConvertibleModel(nn.Module):
    def __init__(self, fc1, fc2, fc3):
        super().__init__()
        self.fc1 = fc1
        self.act1 = nn.ReLU()
        self.fc2 = fc2
        self.act2 = nn.ReLU()
        self.fc3 = fc3

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
        f"block_diag_h{args.hidden}_bs{args.hidden}.pt",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    orig_model = BlockDiagMNIST(ckpt["hidden"], ckpt["block_size"])
    orig_model.load_state_dict(ckpt["model_state_dict"])
    orig_model.eval()

    dense_fc2 = orig_model.fc2.to_linear()
    bd_lr = BlockDiagLowRankLinear.from_dense(dense_fc2, args.block_size, args.rank)
    conv_model = ConvertibleModel(orig_model.fc1, bd_lr, orig_model.fc3)
    conv_model.eval()

    import cukks
    from cukks.context import CKKSInferenceContext
    ctx = CKKSInferenceContext.for_model(
        conv_model, use_square_activation=True,
        enable_gpu=(args.device == "cuda"), security_level=None,
    )
    enc_model, ctx = cukks.convert(
        conv_model, ctx=ctx, use_square_activation=True,
        enable_gpu=(args.device == "cuda"),
    )

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    latencies = []
    cosine_sims = []
    for i in range(args.num_samples):
        sample, label = test_ds[i]
        x_plain = sample.view(-1).float()

        with torch.no_grad():
            h = orig_model.fc1(x_plain)
            h = h * h
            h = bd_lr(h)
            h = h * h
            y_ref = orig_model.fc3(h)

        enc_input = ctx.encrypt(x_plain)
        t0 = time.time()
        enc_output = enc_model(enc_input)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        y_enc_raw = ctx.decrypt(enc_output)
        y_enc = torch.as_tensor(y_enc_raw, dtype=torch.float32).cpu()[:10]
        y_ref_np = y_ref.detach().cpu().numpy()
        y_enc_np = y_enc.detach().cpu().numpy()
        cos_sim = float(np.dot(y_ref_np, y_enc_np) / (
            np.linalg.norm(y_ref_np) * np.linalg.norm(y_enc_np) + 1e-12))
        cosine_sims.append(cos_sim)
        print(f"  sample {i}: cos_sim={cos_sim:.6f} latency={elapsed:.3f}s", file=sys.stderr)

    result = {
        "block_size": args.block_size,
        "rank": args.rank,
        "hidden": args.hidden,
        "mean_latency_sec": round(float(np.mean(latencies)), 3),
        "std_latency_sec": round(float(np.std(latencies)), 3),
        "mean_cosine_similarity": round(float(np.mean(cosine_sims)), 6),
        "num_samples": args.num_samples,
    }
    print(f"RESULT_JSON:{json.dumps(result)}")


if __name__ == "__main__":
    main()
