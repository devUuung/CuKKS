#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cukks


class SimpleResNet(nn.Module):

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


def make_class_templates() -> torch.Tensor:
    templates = torch.zeros(4, 1, 8, 8)
    templates[0, 0, :4, :4] = 1.0
    templates[1, 0, :4, 4:] = 1.0
    templates[2, 0, 4:, :4] = 1.0
    templates[3, 0, 4:, 4:] = 1.0

    templates[:, 0, 3:5, :] += 0.15
    templates[:, 0, :, 3:5] += 0.15
    return templates


def generate_synthetic_data(
    num_train: int = 256,
    num_test: int = 64,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    templates = make_class_templates()

    def build_split(num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.arange(num_samples) % 4
        images = templates[labels].clone()
        images += 0.20 * torch.randn_like(images)
        images += 0.05 * torch.randn(num_samples, 1, 1, 1)
        images.clamp_(-1.0, 1.5)

        perm = torch.randperm(num_samples)
        return images[perm], labels[perm]

    X_train, y_train = build_split(num_train)
    X_test, y_test = build_split(num_test)
    return X_train, y_train, X_test, y_test


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 8,
    verbose: bool = False,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0

        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            batch_x, batch_y = X[idx], y[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if verbose or epoch == epochs - 1:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={total_loss / len(X):.4f}")


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def run_encrypted_inference(
    enc_model: cukks.nn.EncryptedModule,
    ctx: cukks.CKKSInferenceContext,
    model: SimpleResNet,
    sample: torch.Tensor,
) -> torch.Tensor:
    conv = model.features[0]
    conv_params = [{
        "kernel_size": conv.kernel_size,
        "stride": conv.stride,
        "padding": conv.padding,
        "out_channels": conv.out_channels,
    }]
    enc_input = ctx.encrypt_cnn_input(sample.to(torch.float64), conv_params)
    enc_output = enc_model(enc_input)
    return ctx.decrypt(enc_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small ResNet-style encrypted classification example")
    parser.add_argument("--samples", type=int, default=2, help="Number of test samples to compare")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch and per-sample details")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    args = parser.parse_args()

    torch.manual_seed(42)

    print("=" * 60)
    print("CuKKS: ResNet-Style Encrypted Classification Example")
    print("=" * 60)

    print("\n[1] Generating synthetic 8x8 data...")
    X_train, y_train, X_test, y_test = generate_synthetic_data()
    print(f"  Train: {tuple(X_train.shape)}, Test: {tuple(X_test.shape)}")

    print("\n[2] Training plaintext model...")
    model = SimpleResNet()
    train_model(model, X_train, y_train, epochs=args.epochs, verbose=args.verbose)
    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")

    print("\n[3] Converting to encrypted model...")
    enc_model, ctx = cukks.convert(
        model.eval(),
        input_shape=(1, 8, 8),
        activation_degree=3,
        use_square_activation=False,
    )
    print(f"  Context: slots={ctx.num_slots}, depth={ctx.config.mult_depth}")

    backend_info = cukks.get_backend_info()
    print(f"  Backend available: {backend_info['available']}")

    print(f"\n[4] Comparing plaintext vs encrypted predictions ({args.samples} samples)...")
    model.eval()

    with torch.no_grad():
        if not backend_info["available"]:
            print("  CKKS backend not available; showing plaintext predictions only.")
            for i in range(min(args.samples, len(X_test))):
                plain_output = model(X_test[i:i + 1]).squeeze(0)
                pred = plain_output.argmax().item()
                print(f"  Sample {i}: plain={pred}, true={y_test[i].item()}")
            return

        for i in range(min(args.samples, len(X_test))):
            sample = X_test[i]
            true_label = y_test[i].item()

            plain_output = model(sample.unsqueeze(0)).squeeze(0)
            plain_pred = plain_output.argmax().item()

            try:
                start = time.time()
                enc_output = run_encrypted_inference(enc_model, ctx, model, sample)
                elapsed = time.time() - start
                enc_pred = enc_output.argmax().item()

                mae = (plain_output.cpu() - enc_output.cpu()).abs().mean().item()
                print(
                    f"  Sample {i}: plain={plain_pred}, enc={enc_pred}, true={true_label}, "
                    f"mae={mae:.6f}, time={elapsed:.2f}s"
                )
                if args.verbose:
                    print(f"    plaintext logits : {plain_output.tolist()}")
                    print(f"    encrypted logits : {enc_output.tolist()}")
            except Exception as exc:
                print(
                    f"  Sample {i}: plain={plain_pred}, true={true_label}, "
                    f"encrypted inference unavailable ({type(exc).__name__}: {exc})"
                )


if __name__ == "__main__":
    main()
