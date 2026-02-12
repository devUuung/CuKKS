"""Train MNIST models with block-diagonal weight constraint.

Architecture: Linear(784→H) → x² → BlockDiag(H,H,bs) → x² → Linear(H→10)

Trains one model per block_size and saves checkpoints + accuracy report.
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
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cukks.nn.block_diagonal import BlockDiagonalLinear


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
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def count_nonzero_diagonals(model: BlockDiagMNIST) -> int:
    """Count non-zero diagonals of the block-diagonal layer's dense weight."""
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
            diag_idx = d
            if diag_idx < nonzero_diags or (in_features - diag_idx) <= nonzero_diags:
                has_nonzero = True
                break
        if has_nonzero:
            nonempty_giant_steps += 1

    giant_step_rots = max(0, nonempty_giant_steps - 1)
    total_rots = baby_step_rots + giant_step_rots
    total_mults = nonzero_diags

    return {
        "n1": n1,
        "n2": n2,
        "baby_step_rotations": baby_step_rots,
        "giant_step_rotations": giant_step_rots,
        "total_rotations": total_rots,
        "total_evalmults": total_mults,
        "nonzero_diagonals": nonzero_diags,
        "dense_diagonals": in_features,
    }


def train_one(
    hidden: int,
    block_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    data_dir: str,
) -> dict:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)

    model = BlockDiagMNIST(hidden, block_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"block_size={block_size}, hidden={hidden}, params={total_params:,}")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = 100.0 * correct / total
            avg_loss = running_loss / len(train_loader)
            print(f"  epoch {epoch:3d}  loss={avg_loss:.4f}  test_acc={acc:.2f}%")

    train_time = time.time() - t0

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    final_acc = 100.0 * correct / total

    nz_diags = count_nonzero_diagonals(model)
    rot_stats = estimate_bsgs_rotations(hidden, nz_diags)

    result = {
        "block_size": block_size,
        "hidden": hidden,
        "total_params": total_params,
        "test_accuracy": final_acc,
        "train_time_sec": round(train_time, 1),
        "epochs": epochs,
        **rot_stats,
    }

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"block_diag_h{hidden}_bs{block_size}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden": hidden,
        "block_size": block_size,
        "test_accuracy": final_acc,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train MNIST with block-diagonal layers")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    results = []
    for bs in args.block_sizes:
        if args.hidden % bs != 0:
            print(f"SKIP block_size={bs}: hidden={args.hidden} not divisible")
            continue
        r = train_one(args.hidden, bs, args.epochs, args.batch_size, args.lr, args.device, args.data_dir)
        results.append(r)

    print(f"\n{'='*80}")
    print(f"{'block_size':>10} {'params':>8} {'accuracy':>9} {'nz_diags':>9} {'total_diags':>11} {'rotations':>10} {'mults':>6}")
    print(f"{'='*80}")
    for r in results:
        print(
            f"{r['block_size']:>10} {r['total_params']:>8,} "
            f"{r['test_accuracy']:>8.2f}% "
            f"{r['nonzero_diagonals']:>9} {r['dense_diagonals']:>11} "
            f"{r['total_rotations']:>10} {r['total_evalmults']:>6}"
        )

    out_path = os.path.join(os.path.dirname(__file__), "block_diagonal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
