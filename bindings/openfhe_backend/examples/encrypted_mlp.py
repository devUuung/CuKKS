"""
Tiny encrypted MLP inference demo (1 hidden layer, square activation).

The matrices are square so they fit the current dense CKKS matmul helper.
All math stays on ciphertext; activation is x^2.
"""

from __future__ import annotations

import argparse
import torch

from ckks import CKKSConfig, CKKSContext


def encrypted_mlp(x: torch.Tensor, hidden_w: torch.Tensor, hidden_b: torch.Tensor, out_w: torch.Tensor) -> torch.Tensor:
    """
    Runs y = out_w^T * ( (W1 x + b1) ⊙ (W1 x + b1) ) with CKKS ciphertexts.
    Assumes square matrices and len(rotations) covers 1..n-1.
    """
    n = x.numel()
    cfg = CKKSConfig(
        poly_mod_degree=max(16384, 2 * n),
        coeff_mod_bits=(60, 40, 40, 60),
        scale_bits=40,
        security_level="128_classic",
        rotations=list(range(1, n)),  # needed for dense matmul
        relin=True,
        generate_conjugate_keys=False,
    )
    ctx = CKKSContext(cfg)

    cipher_x = ctx.encrypt(x)
    z = cipher_x.matmul_dense(hidden_w).rescale().add(hidden_b)
    z_sq = (z * z).rescale()  # square activation
    y = z_sq.mul(out_w).rescale().sum_slots().decrypt(shape=(1,))
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4, help="width of square layer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    n = args.dim
    scale = 0.1
    x = scale * torch.randn(n, dtype=torch.float64, device="cpu")
    hidden_w = scale * torch.randn(n, n, dtype=torch.float64, device="cpu")
    hidden_b = scale * torch.randn(n, dtype=torch.float64, device="cpu")
    out_w = scale * torch.randn(n, dtype=torch.float64, device="cpu")

    y_enc = encrypted_mlp(x, hidden_w, hidden_b, out_w)

    # Plain reference for comparison
    z_plain = hidden_w @ x + hidden_b
    y_plain = (z_plain * z_plain).dot(out_w)

    diff = abs(y_enc.item() - y_plain.item())
    print(f"encrypted output    {y_enc.item():.6f}")
    print(f"plaintext reference {y_plain.item():.6f}")
    print(f"abs error           {diff:.6f}")


if __name__ == "__main__":
    main()
