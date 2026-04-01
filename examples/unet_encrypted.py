#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cukks


IMAGE_SIZE = 8
NUM_CLASSES = 2
TRAIN_SAMPLES = 96
TEST_SAMPLES = 12
EPOCHS = 8


class SimpleUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.enc_relu1 = nn.ReLU()

        self.bottleneck_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.bottleneck_conv = nn.Conv2d(8, 16, kernel_size=1)
        self.bottleneck_relu = nn.ReLU()

        self.dec_deconv = nn.ConvTranspose2d(
            16,
            8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.dec_relu = nn.ReLU()
        self.dec_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.out_pad = nn.ZeroPad2d(1)
        self.out_conv = nn.Conv2d(8, NUM_CLASSES, kernel_size=1)
        self.out_pool = nn.AdaptiveAvgPool2d((IMAGE_SIZE, IMAGE_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_relu1(self.enc_conv1(x))
        x = self.bottleneck_pool(x)
        x = self.bottleneck_relu(self.bottleneck_conv(x))
        x = self.dec_relu(self.dec_deconv(x))
        x = self.dec_upsample(x)
        x = self.out_pad(x)
        x = self.out_conv(x)
        x = self.out_pool(x)
        return x


def make_mask(image_size: int = IMAGE_SIZE) -> torch.Tensor:
    ys = torch.arange(image_size).view(-1, 1)
    xs = torch.arange(image_size).view(1, -1)

    if torch.rand(1).item() < 0.5:
        height = int(torch.randint(3, 5, (1,)).item())
        width = int(torch.randint(3, 5, (1,)).item())
        top = int(torch.randint(0, image_size - height + 1, (1,)).item())
        left = int(torch.randint(0, image_size - width + 1, (1,)).item())
        mask = (
            (ys >= top)
            & (ys < top + height)
            & (xs >= left)
            & (xs < left + width)
        )
    else:
        radius = int(torch.randint(2, 3, (1,)).item())
        cy = int(torch.randint(radius, image_size - radius, (1,)).item())
        cx = int(torch.randint(radius, image_size - radius, (1,)).item())
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2

    return mask.long()


def generate_synthetic_segmentation(
    num_samples: int,
    image_size: int = IMAGE_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros(num_samples, 1, image_size, image_size)
    masks = torch.zeros(num_samples, image_size, image_size, dtype=torch.long)

    for idx in range(num_samples):
        mask = make_mask(image_size)
        image = 0.10 * torch.randn(image_size, image_size)
        image += mask.float() * 0.9
        image += torch.linspace(0.0, 0.2, image_size).view(1, -1)
        image = image.clamp(0.0, 1.0)

        images[idx, 0] = image
        masks[idx] = mask

    return images, masks


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def binary_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    pred_fg = pred == 1
    target_fg = target == 1
    intersection = (pred_fg & target_fg).sum().item()
    union = (pred_fg | target_fg).sum().item()
    return 1.0 if union == 0 else intersection / union


def train_model(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    epochs: int = EPOCHS,
    verbose: bool = False,
) -> None:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    batch_size = 16

    for epoch in range(epochs):
        perm = torch.randperm(len(images))
        total_loss = 0.0

        for start in range(0, len(images), batch_size):
            idx = perm[start:start + batch_size]
            batch_x = images[idx]
            batch_y = masks[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            avg_loss = total_loss / max(1, (len(images) + batch_size - 1) // batch_size)
            model.eval()
            with torch.no_grad():
                train_acc = pixel_accuracy(model(images), masks)
            model.train()
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, pixel_acc={train_acc:.3f}")


def decrypt_segmentation_logits(ctx: cukks.CKKSInferenceContext, enc_output: Any) -> torch.Tensor:
    decrypted = ctx.decrypt(enc_output)
    layout = getattr(enc_output, "_cnn_layout", None)
    if layout is None:
        raise RuntimeError("Encrypted output is missing CNN layout metadata")

    height = layout["height"]
    width = layout["width"]
    channels = layout["patch_features"]
    return decrypted.view(height, width, channels).permute(2, 0, 1).unsqueeze(0).to(torch.float32)


def run_encrypted_inference(
    enc_model: cukks.nn.EncryptedModule,
    ctx: cukks.CKKSInferenceContext,
    sample: torch.Tensor,
) -> torch.Tensor:
    conv_params = [{
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "out_channels": 8,
    }]
    enc_input = ctx.encrypt_cnn_input(sample.to(torch.float64), conv_params)
    enc_output = enc_model(enc_input)
    return decrypt_segmentation_logits(ctx, enc_output)


def summarize_model(model: nn.Module, images: torch.Tensor, masks: torch.Tensor, label: str) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(images)
    print(
        f"  {label}: pixel_acc={pixel_accuracy(logits, masks):.3f}, "
        f"fg_iou={binary_iou(logits, masks):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small UNet-style encrypted inference example for CuKKS")
    parser.add_argument("--samples", type=int, default=2, help="Number of test samples for comparison")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch training progress")
    args = parser.parse_args()

    torch.manual_seed(7)

    print("=" * 60)
    print("CuKKS: Small UNet-Style Encrypted Segmentation Example")
    print("=" * 60)

    print("\n[1] Generating synthetic segmentation data...")
    train_images, train_masks = generate_synthetic_segmentation(TRAIN_SAMPLES)
    test_images, test_masks = generate_synthetic_segmentation(TEST_SAMPLES)
    print(f"  Train: images={tuple(train_images.shape)}, masks={tuple(train_masks.shape)}")
    print(f"  Test:  images={tuple(test_images.shape)}, masks={tuple(test_masks.shape)}")

    print("\n[2] Training small UNet-style model...")
    model = SimpleUNet()
    train_model(model, train_images, train_masks, verbose=args.verbose)
    summarize_model(model, train_images, train_masks, "train")
    summarize_model(model, test_images, test_masks, "test")

    backend_info = cukks.get_backend_info()
    config = cukks.InferenceConfig(
        poly_mod_degree=32768,
        scale_bits=40,
        mult_depth=14,
        security_level=None,
    )
    ctx = cukks.CKKSInferenceContext(
        config=config,
        max_rotation_dim=1024,
        use_bsgs=True,
        cnn_config={
            "image_height": IMAGE_SIZE,
            "image_width": IMAGE_SIZE,
            "channels": 16,
            "pool_size": 2,
            "pool_stride": 2,
        },
        enable_gpu=False,
        device="cpu",
    )

    print("\n[3] Converting to encrypted model...")
    print(f"  Backend available: {backend_info['available']}")
    enc_model, ctx = cukks.convert(
        model.eval(),
        ctx=ctx,
        input_shape=(1, IMAGE_SIZE, IMAGE_SIZE),
        activation_degree=3,
        use_square_activation=False,
        optimize_cnn=True,
    )
    print(f"  Context: slots={ctx.num_slots}, depth={ctx.config.mult_depth}, device={ctx.device}")

    if not backend_info["available"]:
        print("\n[4] CKKS backend not available here. Plaintext training/conversion succeeded;")
        print("    run this script in a CuKKS backend-enabled environment to execute encrypted inference.")
        return

    print(f"\n[4] Comparing plaintext vs encrypted inference ({min(args.samples, len(test_images))} samples)...")
    model.eval()

    pixel_matches = 0
    total_mae = 0.0
    total_plain_iou = 0.0
    total_enc_iou = 0.0
    num_samples = min(args.samples, len(test_images))

    with torch.no_grad():
        for idx in range(num_samples):
            sample = test_images[idx]
            target = test_masks[idx:idx + 1]

            plain_logits = model(sample.unsqueeze(0))
            enc_logits = run_encrypted_inference(enc_model, ctx, sample)

            mae = (plain_logits - enc_logits).abs().mean().item()
            plain_acc = pixel_accuracy(plain_logits, target)
            enc_acc = pixel_accuracy(enc_logits, target)
            plain_iou = binary_iou(plain_logits, target)
            enc_iou = binary_iou(enc_logits, target)

            plain_pred = plain_logits.argmax(dim=1)
            enc_pred = enc_logits.argmax(dim=1)
            match = (plain_pred == enc_pred).float().mean().item()

            pixel_matches += int(torch.equal(plain_pred, enc_pred))
            total_mae += mae
            total_plain_iou += plain_iou
            total_enc_iou += enc_iou

            print(
                f"  Sample {idx}: plain_acc={plain_acc:.3f}, enc_acc={enc_acc:.3f}, "
                f"plain_iou={plain_iou:.3f}, enc_iou={enc_iou:.3f}, "
                f"logit_mae={mae:.5f}, pixel_match={match:.3f}"
            )

    print("\n[Summary]")
    print(f"  Exact mask matches: {pixel_matches}/{num_samples}")
    print(f"  Avg plaintext IoU:  {total_plain_iou / num_samples:.3f}")
    print(f"  Avg encrypted IoU:  {total_enc_iou / num_samples:.3f}")
    print(f"  Avg logit MAE:      {total_mae / num_samples:.5f}")


if __name__ == "__main__":
    main()
