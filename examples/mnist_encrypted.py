#!/usr/bin/env python3
import argparse
import atexit
import os
import signal
import sys
import tempfile
import threading
import time
import resource
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ckks_torch
from ckks_torch import CKKSInferenceContext, InferenceConfig

MEMORY_LOG_FILE = Path(tempfile.gettempdir()) / "ckks_memory_monitor.log"
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()


def get_memory_usage_mb() -> dict:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"rss_mb": rusage.ru_maxrss / 1024}
    
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    result["vm_rss_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("VmPeak:"):
                    result["vm_peak_mb"] = int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    result["available_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("MemTotal:"):
                    result["total_mb"] = int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    
    if torch.cuda.is_available():
        try:
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            props = torch.cuda.get_device_properties(0)
            result["gpu_total_mb"] = props.total_memory / 1024 / 1024
        except Exception:
            pass
    
    return result


def format_memory(mem: dict, label: str = "") -> str:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    parts = [f"[{ts}]"]
    if label:
        parts.append(f"[{label}]")
    
    if "vm_rss_mb" in mem:
        parts.append(f"RSS={mem['vm_rss_mb']:.0f}MB")
    if "vm_peak_mb" in mem:
        parts.append(f"Peak={mem['vm_peak_mb']:.0f}MB")
    if "available_mb" in mem:
        parts.append(f"SysAvail={mem['available_mb']:.0f}MB")
    if "gpu_allocated_mb" in mem:
        parts.append(f"GPU={mem['gpu_allocated_mb']:.0f}/{mem.get('gpu_total_mb', 0):.0f}MB")
    
    return " | ".join(parts)


def check_memory_pressure(threshold_mb: float = 500) -> Optional[str]:
    mem = get_memory_usage_mb()
    available = mem.get("available_mb", float("inf"))
    if available < threshold_mb:
        return f"LOW MEMORY: {available:.0f}MB available (threshold: {threshold_mb}MB)"
    return None


def log_memory(label: str) -> None:
    mem = get_memory_usage_mb()
    print("  " + format_memory(mem, label))


def _memory_monitor_loop(interval: float = 0.1):
    """Background thread: writes memory snapshots to file until stopped."""
    with open(MEMORY_LOG_FILE, "w") as f:
        f.write(f"=== Memory Monitor Started (PID={os.getpid()}) ===\n")
        prev_gpu = 0
        while not _monitor_stop.is_set():
            try:
                mem = get_memory_usage_mb()
                curr_gpu = mem.get("gpu_allocated_mb", 0)
                if abs(curr_gpu - prev_gpu) > 100 or curr_gpu > 1000:
                    line = format_memory(mem) + " [GPU SPIKE]\n"
                else:
                    line = format_memory(mem) + "\n"
                f.write(line)
                f.flush()
                prev_gpu = curr_gpu
            except Exception as e:
                f.write(f"[ERROR] {e}\n")
                f.flush()
            _monitor_stop.wait(interval)
        f.write("=== Monitor Stopped (normal exit) ===\n")


def start_memory_monitor(interval: float = 0.1):
    global _monitor_thread
    if _monitor_thread is not None:
        return
    _monitor_stop.clear()
    _monitor_thread = threading.Thread(target=_memory_monitor_loop, args=(interval,), daemon=True)
    _monitor_thread.start()
    print(f"  [Monitor] Logging to {MEMORY_LOG_FILE} (interval={interval}s)")


def stop_memory_monitor():
    global _monitor_thread
    if _monitor_thread is None:
        return
    _monitor_stop.set()
    _monitor_thread.join(timeout=2)
    _monitor_thread = None


class OOMError(Exception):
    pass


def setup_signal_handlers():
    def handler(signum, frame):
        sig_name = signal.Signals(signum).name
        mem = get_memory_usage_mb()
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"TERMINATED by signal: {sig_name} ({signum})", file=sys.stderr)
        print(f"Memory at termination:", file=sys.stderr)
        for k, v in mem.items():
            print(f"  {k}: {v:.0f} MB", file=sys.stderr)
        if signum == signal.SIGTERM:
            print("Likely cause: OOM killer (out of memory)", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"\nCheck {MEMORY_LOG_FILE} for memory history before kill", file=sys.stderr)
        sys.exit(137 if signum == signal.SIGTERM else 128 + signum)
    
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def print_crash_analysis():
    print(f"\n  [!] If process is killed, check: {MEMORY_LOG_FILE}")
    print(f"      Run: tail -20 {MEMORY_LOG_FILE}")


def estimate_ckks_memory_mb(poly_mod_degree: int, mult_depth: int, num_rotations: int) -> dict:
    """Estimate GPU memory for CKKS operations (rough heuristic)."""
    # Each ciphertext: 2 polynomials * (mult_depth+2) limbs * poly_mod_degree * 8 bytes
    limbs = mult_depth + 2
    ciphertext_mb = (2 * limbs * poly_mod_degree * 8) / (1024 * 1024)
    
    # Keys: relinearization + rotation keys
    relin_key_mb = ciphertext_mb * limbs * 2
    rotation_key_mb = ciphertext_mb * limbs * num_rotations
    
    # Working memory during ops (rough 3x multiplier)
    working_mb = ciphertext_mb * 10
    
    return {
        "ciphertext_mb": ciphertext_mb,
        "relin_key_mb": relin_key_mb,
        "rotation_keys_mb": rotation_key_mb,
        "working_mb": working_mb,
        "total_estimated_mb": ciphertext_mb + relin_key_mb + rotation_key_mb + working_mb,
        "num_rotations": num_rotations,
    }


def check_gpu_memory_sufficient(ctx: CKKSInferenceContext) -> Tuple[bool, str]:
    """Check if GPU has enough memory for the CKKS context."""
    if not torch.cuda.is_available():
        return True, "No GPU (CPU mode)"
    
    props = torch.cuda.get_device_properties(0)
    gpu_total_mb = props.total_memory / 1024 / 1024
    gpu_free_mb = (props.total_memory - torch.cuda.memory_allocated()) / 1024 / 1024
    
    num_rots = len(ctx._rotations) if hasattr(ctx, '_rotations') else 8000
    est = estimate_ckks_memory_mb(
        ctx.config.poly_mod_degree,
        ctx.config.mult_depth,
        num_rots
    )
    
    msg = (
        f"GPU Memory Check:\n"
        f"  GPU Total: {gpu_total_mb:.0f}MB, Free: {gpu_free_mb:.0f}MB\n"
        f"  Estimated CKKS requirement: {est['total_estimated_mb']:.0f}MB\n"
        f"    - Rotation keys ({est['num_rotations']}): {est['rotation_keys_mb']:.0f}MB\n"
        f"    - Relin keys: {est['relin_key_mb']:.0f}MB\n"
        f"    - Working memory: {est['working_mb']:.0f}MB"
    )
    
    if est['total_estimated_mb'] > gpu_free_mb:
        return False, msg + f"\n  [FATAL] Estimated {est['total_estimated_mb']:.0f}MB > Available {gpu_free_mb:.0f}MB"
    
    if est['total_estimated_mb'] > gpu_free_mb * 0.8:
        return True, msg + f"\n  [WARNING] May be tight on memory"
    
    return True, msg + f"\n  [OK] Should fit"


class SimpleMLP(nn.Module):
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_synthetic_mnist(num_train: int = 1000, num_test: int = 100) -> Tuple:
    """Generate synthetic MNIST-like data for testing."""
    torch.manual_seed(42)
    
    X_train = torch.randn(num_train, 784) * 0.3
    y_train = torch.randint(0, 10, (num_train,))
    
    for i in range(num_train):
        label = y_train[i].item()
        X_train[i, label * 78:(label + 1) * 78] += 1.0
    
    X_test = torch.randn(num_test, 784) * 0.3
    y_test = torch.randint(0, 10, (num_test,))
    
    for i in range(num_test):
        label = y_test[i].item()
        X_test[i, label * 78:(label + 1) * 78] += 1.0
    
    return X_train, y_train, X_test, y_test


def load_mnist() -> Tuple:
    """Load real MNIST dataset if available."""
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        X_train = train_dataset.data.float().view(-1, 784) / 255.0 * 2.0 - 1.0
        y_train = train_dataset.targets
        X_test = test_dataset.data.float().view(-1, 784) / 255.0 * 2.0 - 1.0
        y_test = test_dataset.targets
        
        return X_train, y_train, X_test, y_test
    except ImportError:
        print("torchvision not available, using synthetic data")
        return generate_synthetic_mnist()


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 20) -> None:
    """Train the model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 64
    num_batches = len(X) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(len(X))
        
        for i in range(num_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            batch_x, batch_y = X[idx], y[idx]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        output = model(X)
        pred = output.argmax(dim=1)
        accuracy = (pred == y).float().mean().item()
    return accuracy


def run_encrypted_inference(
    enc_model: ckks_torch.nn.EncryptedModule,
    ctx: CKKSInferenceContext,
    sample: torch.Tensor,
) -> torch.Tensor:
    warning = check_memory_pressure(threshold_mb=1000)
    if warning:
        print(f"  [WARNING] {warning}", file=sys.stderr)
    
    try:
        enc_input = ctx.encrypt(sample.flatten())
    except MemoryError as e:
        raise OOMError(f"Failed to encrypt input: {e}") from e
    
    try:
        enc_output = enc_model(enc_input)
    except MemoryError as e:
        raise OOMError(f"Failed during encrypted forward pass: {e}") from e
    
    try:
        output = ctx.decrypt(enc_output)
    except MemoryError as e:
        raise OOMError(f"Failed to decrypt output: {e}") from e
    
    return output


def main():
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mnist", action="store_true", help="Use real MNIST dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--verbose-memory", action="store_true", help="Log memory usage")
    parser.add_argument("--activation", choices=["square", "cheby_relu"], default="square", help="Activation type")
    parser.add_argument("--cheby-degree", type=int, default=5, help="Chebyshev polynomial degree")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CuKKS: MNIST Encrypted Inference Example")
    print("=" * 60)
    
    log_memory("startup")
    start_memory_monitor(interval=0.5)
    print_crash_analysis()
    
    backend_info = ckks_torch.get_backend_info()
    print(f"\nBackend available: {backend_info['available']}")
    
    # Load data
    print("\n[1] Loading data...")
    if args.use_mnist:
        X_train, y_train, X_test, y_test = load_mnist()
        X_train, y_train = X_train[:5000], y_train[:5000]
    else:
        X_train, y_train, X_test, y_test = generate_synthetic_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create and train model
    print(f"\n[2] Training model (hidden_size={args.hidden})...")
    model = SimpleMLP(input_size=784, hidden_size=args.hidden, num_classes=10)
    train(model, X_train, y_train, epochs=args.epochs)
    
    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    
    print("\\n[3] Converting to encrypted model...")
    log_memory("before conversion")
    
    use_square = args.activation == "square"
    
    try:
        enc_model, ctx = ckks_torch.convert(
            model,
            use_square_activation=use_square,
            activation_degree=args.cheby_degree,
        )
    except MemoryError as e:
        print(f"\n[FATAL] Out of memory during model conversion: {e}", file=sys.stderr)
        log_memory("OOM during conversion")
        sys.exit(137)
    
    print(f"  {enc_model}")
    print(f"  Context: slots={ctx.num_slots}, depth={ctx.config.mult_depth}")
    log_memory("after conversion")
    
    ok, gpu_msg = check_gpu_memory_sufficient(ctx)
    print(f"\n{gpu_msg}")
    if not ok:
        print("\n[FATAL] Insufficient GPU memory. Options:", file=sys.stderr)
        print("  1. Reduce poly_mod_degree (e.g., 8192 instead of 16384)", file=sys.stderr)
        print("  2. Reduce mult_depth", file=sys.stderr)
        print("  3. Use fewer rotation keys", file=sys.stderr)
        sys.exit(137)
    
    print(f"\n[4] Comparing plaintext vs encrypted inference ({args.samples} samples)...")
    
    if not backend_info['available']:
        print("\n  [!] CKKS backend not available - showing plaintext only")
        model.eval()
        with torch.no_grad():
            for i in range(min(args.samples, len(X_test))):
                sample = X_test[i:i+1]
                output = model(sample)
                pred = output.argmax().item()
                true_label = y_test[i].item()
                print(f"  Sample {i}: pred={pred}, true={true_label}, {'OK' if pred == true_label else 'WRONG'}")
        return
    
    model.eval()
    results = []
    
    log_memory("before inference loop")
    
    for i in range(min(args.samples, len(X_test))):
        sample = X_test[i]
        true_label = y_test[i].item()
        
        with torch.no_grad():
            plain_output = model(sample.unsqueeze(0)).squeeze()
        plain_pred = plain_output.argmax().item()
        
        try:
            t0 = time.time()
            enc_output = run_encrypted_inference(enc_model, ctx, sample)
            enc_time = time.time() - t0
            enc_pred = enc_output.argmax().item()
        except OOMError as e:
            print(f"\n[FATAL] {e}", file=sys.stderr)
            log_memory("OOM during inference")
            sys.exit(137)
        except Exception as e:
            print(f"\n[ERROR] Inference failed for sample {i}: {type(e).__name__}: {e}", file=sys.stderr)
            log_memory("error during inference")
            raise
        
        error = (plain_output.cpu() - enc_output.cpu()).abs().mean().item()
        
        results.append({
            'plain_pred': plain_pred,
            'enc_pred': enc_pred,
            'true': true_label,
            'error': error,
            'time': enc_time,
        })
        
        status = "OK" if enc_pred == true_label else "WRONG"
        print(f"  Sample {i}: plain={plain_pred}, enc={enc_pred}, true={true_label}, "
              f"error={error:.6f}, time={enc_time:.2f}s [{status}]")
        
        if args.verbose_memory:
            log_memory(f"after sample {i}")
    
    # Summary
    enc_correct = sum(1 for r in results if r['enc_pred'] == r['true'])
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_error = sum(r['error'] for r in results) / len(results)
    
    print(f"\n[Summary]")
    print(f"  Encrypted accuracy: {enc_correct}/{len(results)}")
    print(f"  Avg inference time: {avg_time:.2f}s")
    print(f"  Avg output error: {avg_error:.6f}")
    
    stop_memory_monitor()


if __name__ == "__main__":
    atexit.register(stop_memory_monitor)
    main()
