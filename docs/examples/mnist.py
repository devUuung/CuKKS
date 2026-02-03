#!/usr/bin/env python3
"""
MNIST Encrypted Inference Example

This example demonstrates how to:
1. Train a simple MLP on MNIST
2. Convert it to an encrypted model
3. Run inference on encrypted data
4. Verify results match plaintext inference

Requirements:
    pip install torch torchvision
    # Plus CKKS backend for actual encryption
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Import CuKKS
import ckks_torch
from ckks_torch import CKKSInferenceContext
from ckks_torch.nn import EncryptedModule


# =============================================================================
# Step 1: Define and Train a Simple Model
# =============================================================================

class SimpleMNISTModel(nn.Module):
    """Simple MLP for MNIST classification."""
    
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


def load_mnist():
    """Load MNIST dataset."""
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        
        return train_data, test_data
    except ImportError:
        print("torchvision not available. Using synthetic data.")
        return None, None


def train_model(model: nn.Module, train_data: Any, epochs: int = 5) -> nn.Module:
    """Train the model on MNIST."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    model.eval()
    return model


# =============================================================================
# Step 2: Convert to Encrypted Model
# =============================================================================

def convert_to_encrypted(model: nn.Module):
    """Convert PyTorch model to encrypted version."""
    
    # Option A: Use square activation (exact, recommended)
    enc_model, ctx = ckks_torch.convert(
        model,
        use_square_activation=True,  # x^2 is exact in CKKS
    )
    
    # Option B: Use polynomial approximation of ReLU
    # enc_model, ctx = ckks_torch.convert(
    #     model,
    #     use_square_activation=False,
    #     activation_degree=4,  # Higher = more accurate but deeper circuit
    # )
    
    print(f"Converted model: {enc_model}")
    print(f"Context: slots={ctx.num_slots}, depth={ctx.config.mult_depth}")
    
    return enc_model, ctx


# =============================================================================
# Step 3: Run Encrypted Inference
# =============================================================================

def run_encrypted_inference(
    enc_model: EncryptedModule, 
    ctx: CKKSInferenceContext, 
    sample: torch.Tensor
) -> torch.Tensor:
    """Run inference on a single encrypted sample."""
    
    # Flatten the input (28x28 -> 784)
    flat_input = sample.flatten()
    
    # Encrypt the input
    enc_input = ctx.encrypt(flat_input)
    print(f"Encrypted input: {enc_input}")
    
    # Run forward pass on encrypted data
    enc_output = enc_model(enc_input)
    print(f"Encrypted output: {enc_output}")
    
    # Decrypt the result
    output = ctx.decrypt(enc_output)
    
    return output


def run_batch_inference(
    enc_model: EncryptedModule, 
    ctx: CKKSInferenceContext, 
    samples: list
) -> list:
    """Run inference on multiple samples packed in one ciphertext."""
    
    # Pack samples into a single ciphertext (SIMD parallelism)
    flat_samples = [s.flatten() for s in samples]
    enc_batch = ctx.encrypt_batch(flat_samples)
    
    # Run inference on all samples at once
    enc_output = enc_model(enc_batch)
    
    # Decrypt individual results
    outputs = ctx.decrypt_batch(enc_output, num_samples=len(samples))
    
    return outputs


# =============================================================================
# Step 4: Compare with Plaintext Inference
# =============================================================================

def compare_results(
    model: nn.Module, 
    enc_model: EncryptedModule, 
    ctx: CKKSInferenceContext, 
    test_data: Any, 
    num_samples: int = 5
) -> None:
    """Compare plaintext and encrypted inference results."""
    
    model.eval()
    
    print("\nComparing plaintext vs encrypted inference:")
    print("-" * 60)
    
    correct_plain = 0
    correct_enc = 0
    
    for i in range(num_samples):
        sample, label = test_data[i]
        
        # Plaintext inference
        with torch.no_grad():
            plain_output = model(sample.unsqueeze(0)).squeeze()
        plain_pred = plain_output.argmax().item()
        
        # Encrypted inference
        enc_output = run_encrypted_inference(enc_model, ctx, sample)
        enc_pred = enc_output.argmax().item()
        
        # Compare
        error = (plain_output - enc_output).abs().mean().item()
        
        print(f"Sample {i}: true={label}, plain={plain_pred}, enc={enc_pred}, error={error:.6f}")
        
        if plain_pred == label:
            correct_plain += 1
        if enc_pred == label:
            correct_enc += 1
    
    print("-" * 60)
    print(f"Plaintext accuracy: {correct_plain}/{num_samples}")
    print(f"Encrypted accuracy: {correct_enc}/{num_samples}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("=" * 60)
    print("CuKKS: MNIST Encrypted Inference Example")
    print("=" * 60)
    
    # Check backend availability
    backend_info = ckks_torch.get_backend_info()
    print(f"\nBackend: {backend_info}")
    
    # Load data
    print("\n[1] Loading MNIST data...")
    train_data, test_data = load_mnist()
    
    if train_data is None or test_data is None:
        print("Cannot load data. Exiting.")
        return
    
    # Create and train model
    print("\n[2] Training model...")
    model = SimpleMNISTModel(hidden_size=128)
    model = train_model(model, train_data, epochs=5)
    
    # Evaluate plaintext accuracy
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)  # type: ignore
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            acc = (model(x).argmax(dim=1) == y).float().mean().item()
            print(f"Test accuracy: {acc:.4f}")
            break
    
    # Convert to encrypted model
    print("\n[3] Converting to encrypted model...")
    enc_model, ctx = convert_to_encrypted(model)
    
    # Run encrypted inference (if backend available)
    if backend_info['available']:
        print("\n[4] Running encrypted inference...")
        compare_results(model, enc_model, ctx, test_data, num_samples=5)
    else:
        print("\n[4] CKKS backend not available.")
        print("    Install openfhe_backend to run actual encrypted inference.")
        print("    The model conversion was successful - encrypted inference")
        print("    would work with the same API once backend is installed.")


if __name__ == "__main__":
    main()
