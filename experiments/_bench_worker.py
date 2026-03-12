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
from experiments._block_diag_bench_common import (
    BlockDiagMNIST,
    BlockDiagMNISTConvertible,
    count_nonzero_diagonals,
    estimate_bsgs_rotations,
)


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

    nz_diags = count_nonzero_diagonals(model.fc2)
    rot_stats = estimate_bsgs_rotations(args.hidden, nz_diags)

    conv_model = BlockDiagMNISTConvertible(ckpt["hidden"], ckpt["block_size"])
    conv_model.load_state_dict(ckpt["model_state_dict"])
    conv_model.eval()

    import cukks
    from cukks.context import CKKSInferenceContext
    enable_gpu = args.device == "cuda"

    ctx = CKKSInferenceContext.for_model(
        conv_model,
        use_square_activation=True,
        enable_gpu=enable_gpu,
        security_level=None,
    )
    enc_model, ctx = cukks.convert(
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
