"""Subprocess worker for single block-diagonal encrypted benchmark.

Prints RESULT_JSON:{...} on success so the parent can parse it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ckks_torch.nn.block_diagonal import BlockDiagonalLinear


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class BlockDiagMNIST(nn.Module):
    """Training model with x² activations."""

    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.act1 = SquareActivation()
        self.fc2 = BlockDiagonalLinear(hidden, hidden, block_size=block_size)
        self.act2 = SquareActivation()
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class BlockDiagMNISTConvertible(nn.Module):
    """Same architecture but with nn.ReLU so the converter can handle it."""

    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = BlockDiagonalLinear(hidden, hidden, block_size=block_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def count_nonzero_diagonals(model: BlockDiagMNIST) -> int:
    dense_w = model.fc2.to_dense_weight().detach()
    n = dense_w.shape[1]
    count = 0
    for d in range(n):
        diag_vals = torch.tensor([dense_w[i, (i + d) % n].item() for i in range(dense_w.shape[0])])
        if diag_vals.abs().max() > 0:
            count += 1
    return count


def estimate_bsgs_rotations(in_features: int, nonzero_diags: int) -> dict:
    n1 = math.ceil(math.sqrt(in_features))
    n2 = math.ceil(in_features / n1)
    baby_step_rots = n1 - 1
    nonempty_giant_steps = 0
    for k in range(n2):
        giant_step = k * n1
        has_nonzero = False
        for j in range(n1):
            d = giant_step + j
            if d >= in_features:
                break
            if d < nonzero_diags or (in_features - d) <= nonzero_diags:
                has_nonzero = True
                break
        if has_nonzero:
            nonempty_giant_steps += 1
    giant_step_rots = max(0, nonempty_giant_steps - 1)
    return {
        "total_rotations": baby_step_rots + giant_step_rots,
        "total_evalmults": nonzero_diags,
        "nonzero_diagonals": nonzero_diags,
        "dense_diagonals": in_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
        f"block_diag_h{args.hidden}_bs{args.block_size}.pt",
    )
    if not os.path.exists(ckpt_path):
        print(f"RESULT_JSON:" + json.dumps({
            "block_size": args.block_size,
            "error": f"checkpoint not found: {ckpt_path}",
        }))
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = BlockDiagMNIST(ckpt["hidden"], ckpt["block_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    nz_diags = count_nonzero_diagonals(model)
    rot_stats = estimate_bsgs_rotations(args.hidden, nz_diags)

    conv_model = BlockDiagMNISTConvertible(ckpt["hidden"], ckpt["block_size"])
    conv_model.load_state_dict(ckpt["model_state_dict"])
    conv_model.eval()

    import ckks_torch
    from ckks_torch.context import CKKSInferenceContext
    enable_gpu = args.device == "cuda"

    ctx = CKKSInferenceContext.for_model(
        conv_model,
        use_square_activation=True,
        enable_gpu=enable_gpu,
        security_level=None,
    )
    enc_model, ctx = ckks_torch.convert(
        conv_model,
        ctx=ctx,
        use_square_activation=True,
        enable_gpu=enable_gpu,
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
            y_plain = model(x_plain.unsqueeze(0)).squeeze(0)

        enc_input = ctx.encrypt(x_plain)

        t0 = time.time()
        enc_output = enc_model(enc_input)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        y_enc_raw = ctx.decrypt(enc_output)
        y_enc_t = torch.as_tensor(y_enc_raw, dtype=torch.float32).cpu()[:10]
        y_plain_np = y_plain.detach().cpu().numpy()
        y_enc_np = y_enc_t.detach().cpu().numpy()

        cos_sim = float(np.dot(y_plain_np, y_enc_np) / (np.linalg.norm(y_plain_np) * np.linalg.norm(y_enc_np) + 1e-12))
        cosine_sims.append(cos_sim)

        enc_pred = int(y_enc_t.argmax())
        plain_pred = int(y_plain.argmax())
        print(f"  sample {i}: label={label} plain_pred={plain_pred} enc_pred={enc_pred} "
              f"cos_sim={cos_sim:.6f} latency={elapsed:.3f}s", file=sys.stderr)

    result = {
        "block_size": args.block_size,
        "hidden": args.hidden,
        "test_accuracy": ckpt.get("test_accuracy", None),
        "mean_latency_sec": round(float(np.mean(latencies)), 3),
        "std_latency_sec": round(float(np.std(latencies)), 3),
        "mean_cosine_similarity": round(float(np.mean(cosine_sims)), 6),
        "min_cosine_similarity": round(float(np.min(cosine_sims)), 6),
        "num_samples": args.num_samples,
        **rot_stats,
    }
    print(f"RESULT_JSON:{json.dumps(result)}")


if __name__ == "__main__":
    main()
