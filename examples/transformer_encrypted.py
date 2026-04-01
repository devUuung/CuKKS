#!/usr/bin/env python3
"""Tiny transformer-style encrypted NLP example for CuKKS.

This example keeps token lookup in plaintext, then encrypts the prepared
embedding representation for private inference.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import cukks


VOCAB_SIZE = 8
EMBED_DIM = 4
SEQ_LEN = 2
HIDDEN_DIM = 8
NUM_CLASSES = 2


class SimpleTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(SEQ_LEN * EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def prepare_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.norm(x)
        return x.reshape(x.shape[0], -1)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.prepare_features(token_ids))


def generate_synthetic_text_data(
    num_samples: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN), generator=generator)
    labels = (token_ids.sum(dim=1) >= VOCAB_SIZE).long()
    return token_ids, labels


def train_model(
    model: nn.Module,
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 8,
    verbose: bool = False,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if verbose:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"  epoch {epoch + 1:02d}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}")


def accuracy(model: nn.Module, token_ids: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(token_ids).argmax(dim=1)
    return (pred == labels).float().mean().item()


def select_demo_indices(labels: torch.Tensor, count: int) -> list[int]:
    chosen: list[int] = []
    seen_labels: set[int] = set()

    for index, label in enumerate(labels.tolist()):
        if label not in seen_labels:
            chosen.append(index)
            seen_labels.add(label)
        if len(chosen) == count:
            return chosen

    for index in range(len(labels)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == count:
            break

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny encrypted transformer-style NLP example")
    parser.add_argument("--samples", type=int, default=2, help="Number of test samples to compare")
    parser.add_argument("--verbose", action="store_true", help="Print training details")
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_printoptions(precision=4, sci_mode=False)

    print("=" * 60)
    print("CuKKS: Tiny Transformer-Style Encrypted NLP Example")
    print("=" * 60)
    print(f"vocab_size={VOCAB_SIZE}, embed_dim={EMBED_DIM}, seq_len={SEQ_LEN}, classes={NUM_CLASSES}")

    train_tokens, train_labels = generate_synthetic_text_data(96, seed=7)
    test_tokens, test_labels = generate_synthetic_text_data(16, seed=13)

    print("\n[1] Training tiny plaintext model on synthetic token IDs...")
    model = SimpleTransformer()
    train_model(model, train_tokens, train_labels, epochs=8, verbose=args.verbose)
    train_acc = accuracy(model, train_tokens, train_labels)
    test_acc = accuracy(model, test_tokens, test_labels)
    print(f"  train accuracy: {train_acc:.3f}")
    print(f"  test accuracy:  {test_acc:.3f}")

    backend_info = cukks.get_backend_info()

    print("\n[2] Converting only the post-embedding classifier to encrypted form...")
    print(f"  backend available: {backend_info['available']}")
    print("  note: embedding + LayerNorm stay plaintext; prepared embeddings are encrypted.")

    if backend_info["available"]:
        model.eval()
        enc_model, ctx = cukks.convert(
            model.classifier,
            activation_degree=3,
            use_square_activation=False,
        )
        print(f"  encrypted head: {enc_model}")
        print(f"  context depth: {ctx.config.mult_depth}")
    else:
        enc_model = None
        ctx = None
        print("  conversion skipped because the CKKS backend is not installed.")

    demo_indices = select_demo_indices(test_labels, min(args.samples, len(test_tokens)))
    print(f"\n[3] Plaintext vs encrypted predictions on {len(demo_indices)} test samples...")
    if not backend_info["available"]:
        print("  CKKS backend not available, so encrypted inference is skipped.")
        with torch.no_grad():
            for display_index, sample_index in enumerate(demo_indices):
                token_ids = test_tokens[sample_index : sample_index + 1]
                logits = model(token_ids).squeeze(0)
                pred = logits.argmax().item()
                print(
                    f"  sample {display_index}: tokens={token_ids.squeeze(0).tolist()}, "
                    f"plain_pred={pred}, true={test_labels[sample_index].item()}"
                )
        return

    assert enc_model is not None and ctx is not None

    with torch.no_grad():
        for display_index, sample_index in enumerate(demo_indices):
            token_ids = test_tokens[sample_index : sample_index + 1]
            features = model.prepare_features(token_ids).squeeze(0)
            plain_logits = model(token_ids).squeeze(0)

            enc_input = ctx.encrypt(features)
            enc_output = enc_model(enc_input)
            dec_logits = ctx.decrypt(enc_output)

            plain_pred = plain_logits.argmax().item()
            enc_pred = dec_logits.argmax().item()
            true_label = test_labels[sample_index].item()
            mae = (plain_logits.cpu() - dec_logits.cpu()).abs().mean().item()

            print(f"  sample {display_index}")
            print(f"    tokens:         {token_ids.squeeze(0).tolist()}")
            print(f"    plain logits:   {plain_logits.tolist()}")
            print(f"    decrypt logits: {dec_logits.tolist()}")
            print(f"    prediction:     plain={plain_pred}, enc={enc_pred}, true={true_label}, mae={mae:.6f}")


if __name__ == "__main__":
    main()
