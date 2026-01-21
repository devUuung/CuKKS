"""
Train a tiny MLP on a toy dataset, then run encrypted inference with CKKS.

The network uses a square activation (x^2) to stay polynomial-friendly. The first
layer is square (input_dim == hidden_dim) so we can reuse the CKKS dense matmul helper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.optim as optim

from ckks import CKKSConfig, CKKSContext
from ckks import CKKSTensor  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

DATASET = "breast_cancer"


def load_dataset(
    device: torch.device,
    dataset: Literal["breast_cancer", "iris", "synthetic"] = "breast_cancer",
    use_sklearn: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (features, labels) with features float64 and labels int64.
    Supports: breast_cancer (default), iris, or a synthetic 3-class blob fallback.
    """
    if use_sklearn:
        try:
            if dataset == "breast_cancer":
                from sklearn.datasets import load_breast_cancer  # type: ignore

                data = load_breast_cancer()
            elif dataset == "iris":
                from sklearn.datasets import load_iris  # type: ignore

                data = load_iris()
            else:
                data = None
            if data is not None:
                X = torch.tensor(data["data"], dtype=torch.float64, device=device)
                y = torch.tensor(data["target"], dtype=torch.int64, device=device)
                return X, y
        except Exception:
            pass  # fall back to synthetic

    torch.manual_seed(0)
    num_classes = 3
    points_per_class = 50
    centers = torch.tensor([[2.0, 0.0, 0.0, -2.0], [-2.0, 2.0, 0.0, 0.0], [0.0, -1.5, 2.0, 1.0]], dtype=torch.float64)
    X_list = []
    y_list = []
    for i in range(num_classes):
        cov = 0.5 * torch.eye(4, dtype=torch.float64)
        samples = torch.distributions.MultivariateNormal(centers[i], cov).sample((points_per_class,))
        X_list.append(samples)
        y_list.append(torch.full((points_per_class,), i, dtype=torch.int64))
    X = torch.cat(X_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)
    return X, y


class SquareMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.W1.t()) + self.b1
        z_sq = z * z  # square activation
        logits = torch.matmul(z_sq, self.W2.t()) + self.b2
        return logits


@dataclass
class TrainResult:
    model: SquareMLP
    X_test: torch.Tensor
    y_test: torch.Tensor


def train_model(device: torch.device, epochs: int = 50, lr: float = 0.05, hidden_dim: int | None = None) -> TrainResult:
    X, y = load_dataset(device, dataset=DATASET)
    # standardize
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)

    num_classes = int(y.max().item() + 1)
    input_dim = X.shape[1]
    hid = hidden_dim or input_dim
    model = SquareMLP(input_dim=input_dim, hidden_dim=hid, num_classes=num_classes).to(device)

    # train/test split
    perm = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float64)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train_onehot)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        print(f"[plain] test accuracy: {acc:.3f}")
    return TrainResult(model=model.cpu(), X_test=X_test.cpu(), y_test=y_test.cpu())


def encrypt_forward(ctx: CKKSContext, sample: torch.Tensor, model: SquareMLP) -> torch.Tensor:
    """
    Encrypted forward pass for a single sample (vector length n).
    Returns logits tensor (num_classes,).
    """
    sample = sample.detach().to(dtype=torch.float64, device=torch.device("cpu"))
    n = sample.numel()
    if model.W1.shape[1] != n:
        raise ValueError("Input dimension mismatch for encrypted path")

    cipher = ctx.encrypt(sample)
    h = cipher.matmul_dense(model.W1.detach().cpu().numpy().tolist()).rescale().add(
        model.b1.detach().cpu().numpy().tolist()
    )
    h_act = (h * h).rescale()

    logits = []
    for i in range(model.W2.shape[0]):
        cls_w = model.W2[i].detach().cpu().numpy().tolist()
        if len(cls_w) < h_act.size:
            cls_w = cls_w + [0.0] * (h_act.size - len(cls_w))
        li = h_act.mul(cls_w).rescale().sum_slots()
        li = li.add([float(model.b2[i].item())])
        logits.append(li.decrypt(shape=(1,)).item())
    return torch.tensor(logits, dtype=torch.float64)


def cheby_relu_coeffs(degree: int) -> list[float]:
    if np is None:
        raise RuntimeError("numpy is required to fit Chebyshev coefficients for ReLU")
    xs = np.linspace(-1, 1, 512)
    ys = np.maximum(xs, 0.0)
    coeffs = np.polynomial.chebyshev.chebfit(xs, ys, degree)
    return coeffs.tolist()


def encrypt_forward_poly(
    ctx: CKKSContext,
    sample: torch.Tensor,
    model: SquareMLP,
    activation: str = "square",
    cheby_degree: int = 5,
    bsgs_group: int | None = None,
) -> torch.Tensor:
    """
    Encrypted forward with selectable activation:
      - square: x^2
      - cheby_relu: Chebyshev approximation of ReLU on [-1, 1]
    """
    sample = sample.detach().to(dtype=torch.float64, device=torch.device("cpu"))
    if activation == "cheby_relu":
        # keep inputs in the approximation interval
        sample = torch.clamp(sample, -1.0, 1.0)
    n = sample.numel()
    slot_count = n
    cipher = ctx.encrypt(sample)
    if bsgs_group:
        # Use BSGS diagonal multiply by converting the matrix to diagonals (pad rows to slot count).
        w = model.W1.detach().cpu().numpy()
        m, n_cols = w.shape
        slot_count = cipher.size
        diagonals: list[list[float]] = []
        for k in range(n_cols):
            diag = [0.0] * slot_count
            for i in range(m):
                diag[i] = float(w[i, (i + k) % n_cols])
            diagonals.append(diag)
        bias = model.b1.detach().cpu().numpy().tolist()
        bias += [0.0] * (slot_count - len(bias))
        h = matmul_diagonal_bsgs(cipher, diagonals, bsgs_group).rescale().add(bias)
    else:
        h = cipher.matmul_dense(model.W1.detach().cpu().numpy().tolist()).rescale().add(
            model.b1.detach().cpu().numpy().tolist()
        )

    if activation == "square":
        h_act = (h * h).rescale()
    elif activation == "cheby_relu":
        coeffs = cheby_relu_coeffs(cheby_degree)
        h_act = h.poly_eval(coeffs).rescale()
    else:
        raise ValueError(f"Unknown activation {activation}")

    logits = []
    for i in range(model.W2.shape[0]):
        cls_w = model.W2[i].detach().cpu().numpy().tolist()
        target_len = h_act.size if hasattr(h_act, "size") else len(cls_w)
        if len(cls_w) < target_len:
            cls_w = cls_w + [0.0] * (target_len - len(cls_w))
        li = h_act.mul(cls_w).rescale().sum_slots()
        li = li.add([float(model.b2[i].item())])
        logits.append(li.decrypt(shape=(1,)).item())
    return torch.tensor(logits, dtype=torch.float64)


# --------------------
# Rotation-key sharing + BSGS diagonal matmul
# --------------------
def matmul_diagonal_bsgs(cipher: CKKSTensor, diagonals: list[list[float]], g: int) -> CKKSTensor:
    """
    Baby-step giant-step diagonal matmul to reduce rotation keys.
    Rotations needed: baby steps 1..(g-1) and giant steps in multiples of g.
    """
    n = len(diagonals)
    if n == 0:
        raise ValueError("diagonals must not be empty")
    if g <= 0:
        raise ValueError("group size g must be positive")

    baby = []
    max_baby = min(g, n)
    for j in range(max_baby):
        baby_cipher = cipher.rotate(j) if j > 0 else cipher
        baby.append(baby_cipher)

    acc = None
    num_giant = (n + g - 1) // g
    for k in range(num_giant):
        block = None
        for j in range(max_baby):
            idx = k * g + j
            if idx >= n:
                break
            term = baby[j].mul(diagonals[idx])
            block = term if block is None else block.add(term)
        if block is None:
            continue
        if k > 0:
            block = block.rotate(k * g)
        acc = block if acc is None else acc.add(block)
    return acc if acc is not None else cipher


def shared_rotation_keys(n: int, g: int) -> list[int]:
    """
    Rotation set that supports BSGS with baby size g for dimension n.
    """
    baby = list(range(1, min(g, n)))
    giant_indices = [k * g for k in range(1, (n + g - 1) // g)]
    return sorted(set(baby + giant_indices))


def run_encrypted_demo(
    sample_count: int = 5,
    device: str = "cpu",
    activation: str = "square",
    cheby_degree: int = 5,
    hidden_dim: int | None = None,
):
    train_res = train_model(torch.device(device), hidden_dim=hidden_dim)
    model = train_res.model
    X_test, y_test = train_res.X_test, train_res.y_test

    n = X_test.shape[1]
    coeffs = (60, 40, 40, 60) if activation == "square" else (60, 40, 40, 40, 40, 40, 40, 40, 60)
    # choose rotation set for BSGS based on column count
    use_bsgs = True
    bsgs_g = 4
    rotation_set = shared_rotation_keys(n, bsgs_g) if use_bsgs else list(range(1, n))

    cfg = CKKSConfig(
        poly_mod_degree=32768,
        coeff_mod_bits=coeffs,
        scale_bits=40,
        security_level="none",
        rotations=rotation_set,
        relin=True,
        generate_conjugate_keys=False,
    )
    ctx = CKKSContext(cfg)

    correct = 0
    total = X_test.shape[0] if sample_count <= 0 else min(sample_count, X_test.shape[0])
    for i in range(total):
        sample = X_test[i]
        plain_logits = model(sample.unsqueeze(0)).squeeze(0)
        enc_logits = encrypt_forward_poly(
            ctx,
            sample,
            model,
            activation=activation,
            cheby_degree=cheby_degree,
            bsgs_group=bsgs_g if use_bsgs else None,
        )
        plain_pred = int(torch.argmax(plain_logits))
        enc_pred = int(torch.argmax(enc_logits))
        correct += int(plain_pred == y_test[i].item())
        print(f"[#{i}] plain_pred={plain_pred} enc_pred={enc_pred} label={y_test[i].item()} logits(enc)={enc_logits.tolist()}")

    print(f"[encrypted] accuracy on {total} samples: {correct/total:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=0, help="number of test samples to run encrypted (0=all)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--activation", type=str, default="square", choices=["square", "cheby_relu"])
    parser.add_argument("--cheby-degree", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="breast_cancer", choices=["breast_cancer", "iris", "synthetic"])
    parser.add_argument("--hidden-dim", type=int, default=None, help="hidden width (default: input dimension)")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    # propagate dataset selection globally
    global DATASET
    DATASET = args.dataset
    run_encrypted_demo(
        sample_count=args.samples,
        device=args.device,
        activation=args.activation,
        cheby_degree=args.cheby_degree,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
