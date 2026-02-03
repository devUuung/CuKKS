#!/usr/bin/env python3
"""
Complete Example: Encrypted Deep Learning Inference with ckks_torch.

This example demonstrates the end-to-end workflow:
1. Define and train a PyTorch model
2. Convert it to an encrypted model
3. Run encrypted inference
4. Compare with plaintext results

Usage:
    python -m ckks_torch.examples.encrypted_inference
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def create_sample_model(input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 4) -> nn.Module:
    """Create a simple MLP for demonstration.
    
    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (number of classes).
        
    Returns:
        A PyTorch Sequential model.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def create_sample_data(
    num_samples: int = 100,
    input_dim: int = 16,
    num_classes: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data.
    
    Args:
        num_samples: Number of samples to generate.
        input_dim: Input feature dimension.
        num_classes: Number of classes.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (features, labels).
    """
    torch.manual_seed(seed)
    
    # Generate random class centers
    centers = torch.randn(num_classes, input_dim) * 2
    
    # Generate samples around each center
    samples_per_class = num_samples // num_classes
    X_list = []
    y_list = []
    
    for i in range(num_classes):
        noise = torch.randn(samples_per_class, input_dim) * 0.5
        X_list.append(centers[i] + noise)
        y_list.append(torch.full((samples_per_class,), i, dtype=torch.long))
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
) -> float:
    """Train the model.
    
    Args:
        model: The model to train.
        X: Training features.
        y: Training labels.
        epochs: Number of training epochs.
        lr: Learning rate.
        
    Returns:
        Final training accuracy.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}")
    
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def demo_encrypted_inference(
    use_square_activation: bool = True,
    input_dim: int = 16,
    hidden_dim: int = 32,
    output_dim: int = 4,
    num_test_samples: int = 5,
):
    """Demonstrate encrypted inference workflow.
    
    Args:
        use_square_activation: If True, use x^2 activation instead of ReLU.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Number of output classes.
        num_test_samples: Number of samples to test.
    """
    import ckks_torch
    
    print("=" * 60)
    print("CuKKS Encrypted Inference Demo")
    print("=" * 60)
    
    # Check backend availability
    backend_info = ckks_torch.get_backend_info()
    print(f"\nBackend: {backend_info}")
    
    if not backend_info["available"]:
        print("\nWARNING: CKKS backend not available!")
        print("This demo will show the conversion process but cannot run encrypted inference.")
        print("Please build the ckks_openfhe_backend extension first.\n")
    
    # Step 1: Create and train model
    print("\n[Step 1] Creating and training model...")
    
    # Use appropriate model based on activation choice
    if use_square_activation:
        # For encrypted inference, x^2 activation is exact
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Will be replaced with x^2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # Will be replaced with x^2
            nn.Linear(hidden_dim, output_dim),
        )
    else:
        model = create_sample_model(input_dim, hidden_dim, output_dim)
    
    X_train, y_train = create_sample_data(100, input_dim, output_dim)
    X_test, y_test = create_sample_data(20, input_dim, output_dim, seed=123)
    
    train_acc = train_model(model, X_train, y_train)
    print(f"  Training accuracy: {train_acc:.3f}")
    
    # Step 2: Convert to encrypted model
    print("\n[Step 2] Converting to encrypted model...")
    
    enc_model, ctx = ckks_torch.convert(
        model,
        use_square_activation=use_square_activation,
        activation_degree=4,
    )
    
    print(f"  Encrypted model structure:")
    print(f"    {enc_model}")
    print(f"\n  Context: {ctx}")
    
    # Step 3: Run inference comparison
    print(f"\n[Step 3] Running inference on {num_test_samples} test samples...")
    
    if not backend_info["available"]:
        print("  [Skipping encrypted inference - backend not available]")
        
        # Just show plaintext results
        model.eval()
        with torch.no_grad():
            for i in range(min(num_test_samples, len(X_test))):
                sample = X_test[i]
                plain_output = model(sample.unsqueeze(0)).squeeze(0)
                plain_pred = plain_output.argmax().item()
                true_label = y_test[i].item()
                
                print(f"\n  Sample {i}:")
                print(f"    Plaintext prediction: {plain_pred}")
                print(f"    True label: {true_label}")
                print(f"    Correct: {plain_pred == true_label}")
        return
    
    # Run both plaintext and encrypted inference
    model.eval()
    correct_plain = 0
    correct_enc = 0
    total_time = 0
    
    with torch.no_grad():
        for i in range(min(num_test_samples, len(X_test))):
            sample = X_test[i]
            true_label = y_test[i].item()
            
            # Plaintext inference
            plain_output = model(sample.unsqueeze(0)).squeeze(0)
            plain_pred = plain_output.argmax().item()
            
            # Encrypted inference
            t0 = time.time()
            enc_input = ctx.encrypt(sample)
            enc_output = enc_model(enc_input)
            dec_output = ctx.decrypt(enc_output)
            enc_time = time.time() - t0
            total_time += enc_time
            
            enc_pred = dec_output.argmax().item()
            
            correct_plain += int(plain_pred == true_label)
            correct_enc += int(enc_pred == true_label)
            
            # Compute error
            error = (plain_output - dec_output).abs().mean().item()
            
            print(f"\n  Sample {i}:")
            print(f"    Plaintext pred: {plain_pred}, Encrypted pred: {enc_pred}, True: {true_label}")
            print(f"    Mean abs error: {error:.6f}")
            print(f"    Encrypted time: {enc_time:.3f}s")
    
    print(f"\n[Summary]")
    print(f"  Plaintext accuracy: {correct_plain}/{num_test_samples}")
    print(f"  Encrypted accuracy: {correct_enc}/{num_test_samples}")
    print(f"  Avg encrypted inference time: {total_time/num_test_samples:.3f}s")


def demo_model_conversion():
    """Demonstrate model conversion without running encryption.
    
    This is useful for understanding how the library works
    even without the CKKS backend installed.
    """
    import ckks_torch
    from ckks_torch.converter import ModelConverter, ConversionOptions
    
    print("=" * 60)
    print("Model Conversion Demo (No Encryption)")
    print("=" * 60)
    
    # Create a more complex model
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.eval()
    
    print("\nOriginal PyTorch model:")
    print(model)
    
    # Estimate depth
    depth = ckks_torch.estimate_depth(model)
    print(f"\nEstimated multiplicative depth: {depth}")
    
    # Convert with different options
    print("\n[Option 1] Convert with ReLU approximation (degree 4):")
    converter1 = ModelConverter(ConversionOptions(
        fold_batchnorm=True,
        activation_degree=4,
        use_square_activation=False,
    ))
    
    try:
        enc_model1 = converter1.convert(model)
        print(enc_model1)
    except NotImplementedError as e:
        print(f"  Note: {e}")
        print("  (Some layers may need manual handling for complex architectures)")
    
    print("\n[Option 2] Convert with square activation (exact):")
    converter2 = ModelConverter(ConversionOptions(
        fold_batchnorm=True,
        use_square_activation=True,
    ))
    
    try:
        enc_model2 = converter2.convert(model)
        print(enc_model2)
    except NotImplementedError as e:
        print(f"  Note: {e}")
    
    # Simple MLP conversion (fully supported)
    print("\n[Option 3] Simple MLP (fully supported):")
    simple_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    simple_model.eval()
    
    enc_simple = converter2.convert(simple_model)
    print(enc_simple)


def main():
    parser = argparse.ArgumentParser(description="CuKKS Examples")
    parser.add_argument(
        "--demo",
        choices=["inference", "conversion"],
        default="conversion",
        help="Which demo to run",
    )
    parser.add_argument(
        "--use-relu",
        action="store_true",
        help="Use polynomial ReLU approximation instead of x^2",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of test samples for inference demo",
    )
    args = parser.parse_args()
    
    if args.demo == "inference":
        demo_encrypted_inference(
            use_square_activation=not args.use_relu,
            num_test_samples=args.samples,
        )
    else:
        demo_model_conversion()


if __name__ == "__main__":
    main()
