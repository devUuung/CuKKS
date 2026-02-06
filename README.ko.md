<p align="center">
  <a href="README.md">English</a> |
  <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">CuKKS</h1>

<p align="center">
  <strong>PyTorch를 위한 GPU 가속 CKKS 동형암호</strong>
</p>

<p align="center">
  <a href="https://github.com/devUuung/CuKKS/actions"><img src="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/devUuung/CuKKS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
</p>

<p align="center">
  학습된 PyTorch 모델을 <strong>암호화된 데이터</strong>에서 실행 — 정확도를 유지하면서 프라이버시 보호.<br>
  OpenFHE 기반, CUDA 가속 지원.
</p>

---

## 빠른 시작

```python
import torch.nn as nn
import ckks_torch

# 1. 모델 정의 및 학습 (일반 PyTorch)
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. 암호화 모델로 변환
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 3. 암호화 추론 실행
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)
output = ctx.decrypt(enc_output)
```

## 설치

```bash
pip install cukks-cu121  # CUDA 12.1 (가장 일반적)
```

<details>
<summary><strong>다른 CUDA 버전</strong></summary>

| 패키지 | CUDA | 지원 GPU |
|---------|------|----------------|
| `cukks-cu118` | 11.8 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu121` | 12.1 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu124` | 12.4 | V100, T4, RTX 20/30/40xx, A100, H100 |
| `cukks-cu128` | 12.8 | 위 모두 + **RTX 50xx** |
| `cukks` | - | CPU 전용 |

```bash
pip install cukks-cu118  # CUDA 11.8
pip install cukks-cu124  # CUDA 12.4
pip install cukks-cu128  # CUDA 12.8 (RTX 50xx)
pip install cukks        # CPU 전용
```

</details>

<details>
<summary><strong>Docker 이미지</strong></summary>

| CUDA | 호환 Docker 이미지 |
|------|-------------------------|
| 11.8 | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` |
| 12.1 | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| 12.4 | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` |
| 12.8 | `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04` |

```bash
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install cukks-cu121
```

</details>

<details>
<summary><strong>소스에서 빌드</strong></summary>

```bash
git clone https://github.com/devUuung/CuKKS.git && cd CuKKS
pip install -e .

# OpenFHE 백엔드 빌드
cd openfhe-gpu-public && mkdir build && cd build
cmake .. -DWITH_CUDA=ON && make -j$(nproc)

cd ../../bindings/openfhe_backend
pip install -e .
```

</details>

## 주요 기능

| 기능 | 설명 |
|---------|-------------|
| **PyTorch API** | 익숙한 인터페이스 — `ckks_torch.convert(model)` 호출만으로 변환 |
| **GPU 가속** | OpenFHE 기반 CUDA 가속 HE 연산 |
| **자동 최적화** | BatchNorm 폴딩, BSGS 행렬 곱셈 |
| **다양한 레이어** | Linear, Conv2d, ReLU/GELU/SiLU, Pool, LayerNorm, Attention |

## 지원 레이어

| 레이어 | 암호화 버전 | 비고 |
|-------|------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS 최적화 |
| `nn.Conv2d` | `EncryptedConv2d` | im2col 방식 |
| `nn.ReLU/GELU/SiLU` | 다항식 근사 | 또는 정확한 `x²` 사용 |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | 회전 기반 |
| `nn.BatchNorm` | Folded | 이전 레이어에 병합 |
| `nn.LayerNorm` | `EncryptedLayerNorm` | 다항식 근사 |
| `nn.Attention` | `EncryptedApproxAttention` | seq_len=1 |

<details>
<summary><strong>전체 레이어 지원 표</strong></summary>

| PyTorch 레이어 | 암호화 버전 | 비고 |
|--------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS 최적화로 완전 지원 |
| `nn.Conv2d` | `EncryptedConv2d` | im2col 방식 |
| `nn.ReLU` | `EncryptedReLU` | 다항식 근사 |
| `nn.GELU` | `EncryptedGELU` | 다항식 근사 |
| `nn.SiLU` | `EncryptedSiLU` | 다항식 근사 |
| `nn.Sigmoid` | `EncryptedSigmoid` | 다항식 근사 |
| `nn.Tanh` | `EncryptedTanh` | 다항식 근사 |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | 완전 지원 |
| `nn.MaxPool2d` | `EncryptedMaxPool2d` | 다항식으로 근사 |
| `nn.Flatten` | `EncryptedFlatten` | 논리적 reshape |
| `nn.BatchNorm1d/2d` | Folded | 이전 레이어에 병합 |
| `nn.Sequential` | `EncryptedSequential` | 완전 지원 |
| `nn.Dropout` | `EncryptedDropout` | 추론 시 no-op |
| `nn.LayerNorm` | `EncryptedLayerNorm` | 순수 HE 다항식 근사 |
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | 다항식 softmax (seq_len=1) |

</details>

## 활성화 함수

CKKS는 다항식 연산만 지원합니다. 다음 중 선택:

```python
# 옵션 1: 제곱 활성화 (권장 - 정확, 오차 없음)
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 옵션 2: 다항식 근사 (원래 ReLU/GELU에 더 가까움)
enc_model, ctx = ckks_torch.convert(model, use_square_activation=False, activation_degree=4)
```

## GPU 가속

| 연산 | 가속 여부 |
|-----------|-------------|
| Add/Sub/Mul/Square | ✅ GPU |
| Rotate/Rescale | ✅ GPU |
| Bootstrap | ✅ GPU |
| Encrypt/Decrypt | CPU |

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(poly_mod_degree=8192, scale_bits=40)
ctx = CKKSContext(config, enable_gpu=True)  # 기본적으로 GPU 활성화
```

## 예제

```bash
# 빠른 데모 (GPU 불필요)
python -m ckks_torch.examples.encrypted_inference --demo conversion

# MNIST 암호화 추론
python examples/mnist_encrypted.py --hidden 64 --samples 5
```

<details>
<summary><strong>CNN 예제</strong></summary>

```python
import torch.nn as nn
import ckks_torch

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 14 * 14, 10)
    
    def forward(self, x):
        return self.fc(self.flatten(self.pool1(self.act1(self.conv1(x)))))

model = MNISTCNN()
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

enc_input = ctx.encrypt(image)
prediction = ctx.decrypt(enc_model(enc_input)).argmax()
```

> **참고**: `forward()`의 모든 연산은 레이어 속성(예: `self.act1`)이어야 하며, `x ** 2` 같은 인라인 연산은 사용할 수 없습니다.

</details>

<details>
<summary><strong>배치 처리</strong></summary>

```python
# 여러 샘플을 단일 암호문에 패킹 (SIMD)
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)
enc_output = enc_model(enc_batch)
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

</details>

## 문제 해결

| 문제 | 해결책 |
|-------|----------|
| 메모리 부족 | `poly_mod_degree` 감소 (16384 대신 8192) |
| 낮은 정확도 | `use_square_activation=True` 사용 또는 `activation_degree` 증가 |
| 느린 성능 | 배치 처리 활성화, 네트워크 깊이 감소 |

## 문서

- [API 레퍼런스](docs/api.md)
- [GPU 가속 가이드](docs/gpu-acceleration.md)
- [CKKS 개념](docs/concepts.md)

## 라이선스

Apache License 2.0

## 인용

```bibtex
@software{cukks,
  title = {CuKKS: PyTorch-compatible Encrypted Deep Learning},
  year = {2024},
  url = {https://github.com/devUuung/CuKKS}
}
```

## 관련 리소스

### 라이브러리
- [OpenFHE](https://github.com/openfheorg/openfhe-development) — 기반 HE 라이브러리
- [Microsoft SEAL](https://github.com/microsoft/SEAL) — 대안 HE 라이브러리

### 논문
- [Homomorphic Encryption for Arithmetic of Approximate Numbers](https://eprint.iacr.org/2016/421) — Cheon et al. (CKKS)
- [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153) — Cheon et al.
- [Faster Homomorphic Linear Transformations in HElib](https://eprint.iacr.org/2018/244) — Halevi & Shoup (BSGS)
