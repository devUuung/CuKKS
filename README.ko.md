<p align="center">
  <a href="README.md">English</a> |
  <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">CuKKS</h1>

<p align="center">
  <strong>PyTorch를 위한 GPU 가속 CKKS 동형암호</strong>
</p>

<p align="center">
  <a href="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml"><img src="https://github.com/devUuung/CuKKS/actions/workflows/build-wheels.yml/badge.svg" alt="Build Wheels"></a>
  <a href="https://github.com/devUuung/CuKKS/actions/workflows/test-python.yml"><img src="https://github.com/devUuung/CuKKS/actions/workflows/test-python.yml/badge.svg" alt="Test Python"></a>
  <a href="https://github.com/devUuung/CuKKS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10--3.13-blue.svg" alt="Python 3.10-3.13"></a>
</p>

<p align="center">
  학습된 PyTorch 모델을 <strong>암호화된 데이터</strong>에서 실행 — 복호화 없이, 프라이버시를 유지하면서.<br>
  OpenFHE 기반, CUDA 가속 지원.
</p>

---

## 왜 CuKKS인가?

기존 머신러닝은 원본 입력 데이터에 접근해야 합니다 — 의료, 금융, 생체 인증 같은 민감한 분야에서는 프라이버시 위험이 됩니다. CuKKS는 **평문을 전혀 보지 않는 모델**을 배포할 수 있게 해줍니다:

```
사용자: 암호화(입력) → [암호문] → 서버: 모델([암호문]) → [암호화 출력] → 사용자: 복호화
```

서버는 데이터를 복호화하지 않고도 완전한 추론을 수행합니다. CuKKS는 이를 실용적으로 만듭니다:

- **한 줄 변환** — `cukks.convert(model)`로 학습된 PyTorch 모델을 즉시 변환
- **GPU 가속** — OpenFHE 기반 CUDA 가속 HE 연산
- **37개 레이어** — Linear, Conv2d부터 Attention, GroupNorm, ConvTranspose2d까지

## 빠른 시작

```python
import torch.nn as nn
import cukks

# 1. 모델을 평소대로 학습
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. 암호화 모델로 변환
enc_model, ctx = cukks.convert(model)

# 3. 암호화 추론 실행
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)
output = ctx.decrypt(enc_output)  # 같은 결과, 서버에서는 절대 복호화되지 않음
```

## 설치

```bash
pip install cukks[cu121]  # PyTorch CUDA 버전에 맞춰 선택: cu118, cu121, cu124, cu128
```

| 명령어 | CUDA | Compute Capability |
|--------|------|--------------------|
| `pip install cukks[cu118]` | 11.8 | sm_50 – sm_90 (Maxwell ~ Hopper) |
| `pip install cukks[cu121]` | 12.1 | sm_50 – sm_90 (Maxwell ~ Hopper) |
| `pip install cukks[cu124]` | 12.4 | sm_50 – sm_90a (Maxwell ~ Hopper) |
| `pip install cukks[cu128]` | 12.8 | sm_50 – sm_100 (Maxwell ~ Blackwell) |

CUDA 버신이 헷갈리나요? `python -c "import torch; print(torch.version.cuda)"`를 실행하세요.

<details>
<summary><strong>Docker, CLI 도구, 소스 빌드</strong></summary>

### Docker

```bash
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install cukks[cu121]
```

### 자동 설치 CLI

```bash
pip install cukks
cukks-install-backend  # PyTorch CUDA 버전 자동 감지 및 설치
```

### 소스에서 빌드

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

### 모델 변환 한 줄로 끝내기

모델 재작성 없음. 커스텀 HE 코드 없음. `cukks.convert(model)` 한 줄이면 됩니다:

```python
# MLP, CNN, Transformer — 어떤 PyTorch 모델이든 자동 변환
enc_model, ctx = cukks.convert(model, activation_degree=4)
```

BatchNorm 폴딩, BSGS 행렬 곱셈, CNN 최적화가 자동으로 적용됩니다.

### 37개 레이어 지원

| 카테고리 | 레이어 |
|----------|--------|
| **선형** | Linear, BlockDiagonalLinear, BlockDiagLowRankLinear |
| **합성곱** | Conv1d, Conv2d, ConvTranspose2d |
| **풀링** | AvgPool2d, MaxPool2d, AdaptiveAvgPool2d |
| **정규화** | LayerNorm, GroupNorm, InstanceNorm1d/2d, BatchNorm (folded) |
| **활성화** | ReLU, GELU, SiLU, Sigmoid, Tanh, Square |
| **어텐션** | MultiheadAttention (seq_len ≤ 8) |
| **임베딩** | Embedding |
| **공간 연산** | Upsample, PixelShuffle, PixelUnshuffle |
| **패딩** | ZeroPad2d, ConstantPad2d, ReflectionPad2d, ReplicationPad2d |
| **기타** | Flatten, Dropout, Sequential, ResidualBlock |

[전체 레이어 표 →](#지원-레이어)

### GPU 가속 HE 연산

모든 핵심 HE 연산이 GPU에서 실행됩니다:

| 연산 | GPU |
|------|-----|
| Add / Sub / Mul / Square | ✅ |
| Rotate / Rescale | ✅ |
| Bootstrap | ✅ |
| Plaintext 캐시 | ✅ |

### 다항식 활성화 함수 근사

CKKS는 다항식 연산만 지원합니다. CuKKS는 비다항식 활성화 함수(ReLU, GELU, SiLU 등)를 Chebyshev 다항식 피팅으로 근사합니다:

```python
# 기본: 4차 다항식 (정확도/깊이 균형)
enc_model, ctx = cukks.convert(model)

# 더 높은 차수로 정확도 향상 (깊이 소비 증가)
enc_model, ctx = cukks.convert(model, activation_degree=8)
```

### 패킹 배치 추론

단일 암호문에서 여러 샘플을 한 번에 처리:

```python
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)
enc_output = enc_model(enc_batch)
outputs = ctx.decrypt_batch(enc_output, sample_shape=(8,))
```

## 예제

```bash
# MNIST 분류 (MLP)
python examples/mnist_encrypted.py --hidden 64 --samples 5

# UNet 스타일 세그멘테이션 (ConvTranspose2d, AdaptiveAvgPool2d, Upsample)
python examples/unet_encrypted.py --samples 2

# ResNet 스타일 분류 (GroupNorm, AdaptiveAvgPool2d)
python examples/resnet_encrypted.py --samples 1

# Transformer 스타일 NLP (Embedding, LayerNorm)
python examples/transformer_encrypted.py --samples 2
```

전체 스크립트는 [examples/](examples/)에서 확인하세요.

## 벤치마크

CuKKS는 암호화 추론 성능 측정을 위한 벤치마크 스위트가 포함되어 있습니다:

```bash
# 전체 벤치마크 실행
python benchmarks/run_benchmarks.py

# 특정 모델 벤치마크
python benchmarks/run_benchmarks.py --model mlp

# 결과를 JSON으로 저장
python benchmarks/run_benchmarks.py --output results.json
```

지원 모델:

| 모델 | 파라미터 | 입력 | 아키텍처 |
|------|----------|------|----------|
| MLP | 50,890 | (1, 784) | Linear(784→64) → ReLU → Linear(64→10) |
| CNN | 15,770 | (1, 1, 28, 28) | Conv2d(1→8, 3×3) → ReLU → AvgPool2d(2) → Linear(1568→10) |
| ResNet | 1,300 | (1, 1, 8, 8) | Conv2d(1→8, 3×3) → GroupNorm → ReLU → Conv2d(8→16, 1×1) → GroupNorm → ReLU → AdaptiveAvgPool2d(4×4) → Linear(256→4) |
| Transformer | 212 | (1, 8) | Linear(8→16) → ReLU → Linear(16→4) |

실행 예시:

```
Model           Plain (ms)   Encrypted (ms)  Overhead   MAE       
--------------------------------------------------------------
mlp             0.01         84.16           6858      x 0.075594
cnn             0.03         2820.15         94053     x 0.048294
resnet          0.04         11444.55        269018    x 0.098026
transformer     0.01         54.27           7794      x 0.087978
```

> **참고:** 벤치마크는 OpenFHE GPU 백엔드가 필요합니다. CUDA 지원 머신에서 실행하세요.

## 지원 레이어

| PyTorch 레이어 | 암호화 버전 | 비고 |
|----------------|-------------|------|
| `nn.Linear` | `EncryptedLinear` | BSGS 최적화 |
| `nn.Conv1d` | `EncryptedConv1d` | 1D im2col |
| `nn.Conv2d` | `EncryptedConv2d` | im2col, BSGS |
| `nn.ConvTranspose2d` | `EncryptedConvTranspose2d` | 전치 합성곱 |
| `nn.ReLU` | `EncryptedReLU` | 다항식 근사 |
| `nn.GELU` | `EncryptedGELU` | 다항식 근사 |
| `nn.SiLU` | `EncryptedSiLU` | 다항식 근사 |
| `nn.Sigmoid` | `EncryptedSigmoid` | 다항식 근사 |
| `nn.Tanh` | `EncryptedTanh` | 다항식 근사 |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | 회전 기반 |
| `nn.MaxPool2d` | `EncryptedMaxPool2d` | 다항식 근사 |
| `nn.AdaptiveAvgPool2d` | `EncryptedAdaptiveAvgPool2d` | 글로벌 풀링 고속 경로 |
| `nn.Flatten` | `EncryptedFlatten` | 논리적 reshape |
| `nn.BatchNorm1d/2d` | Folded | 이전 레이어에 병합 |
| `nn.GroupNorm` | `EncryptedGroupNorm` | 그룹별 다항식 |
| `nn.InstanceNorm1d/2d` | `EncryptedInstanceNorm1d/2d` | 채널별 다항식 |
| `nn.LayerNorm` | `EncryptedLayerNorm` | 다항식 1/sqrt |
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | seq_len ≤ 8 |
| `nn.Embedding` | `EncryptedEmbedding` | one-hot 행렬 곱 |
| `nn.Upsample` | `EncryptedUpsample` | 최근접 / 쌍선형 |
| `nn.PixelShuffle` | `EncryptedPixelShuffle` | 채널→공간 |
| `nn.PixelUnshuffle` | `EncryptedPixelUnshuffle` | 공간→채널 |
| `nn.ZeroPad2d` | `EncryptedZeroPad2d` | 산란 행렬 |
| `nn.ConstantPad2d` | `EncryptedConstantPad2d` | 산란 + 상수 |
| `nn.ReflectionPad2d` | `EncryptedReflectionPad2d` | 반사 매핑 |
| `nn.ReplicationPad2d` | `EncryptedReplicationPad2d` | 복제 매핑 |
| `nn.Sequential` | `EncryptedSequential` | 완전 지원 |
| `nn.Dropout` | `EncryptedDropout` | 추론 시 no-op |
| `nn.ResidualBlock` | `EncryptedResidualBlock` | 스킵 연결 |

## 문서

- [API 레퍼런스](docs/api.md)
- [CKKS 개념](docs/concepts.md)
- [GPU 가속 가이드](docs/gpu-acceleration.md)
- [STIP 패킹 어텐션](docs/stip-attention.md)
- [예제 개요](docs/examples/README.md)

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
- [GAZELLE: A Low Latency Framework for Secure Neural Network Inference](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar) — Juvekar et al. (합성곱)
- [PP-STAT: An Efficient Privacy-Preserving Statistical Analysis Framework using Homomorphic Encryption](https://doi.org/10.1145/3583780) — Choi (암호화 통계)
- [STIP: Efficient and Secure Non-Interactive Transformer Inference via Compact Packing](https://doi.org/10.1145/3696410.3714779) — Wang et al. (패킹 어텐션)
- [Efficient Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2020/1203) — Bossuat et al. (Double-Hoisting)
- [On the Number of Nonscalar Multiplications Necessary to Evaluate Polynomials](https://doi.org/10.1137/0202007) — Paterson & Stockmeyer (다항식 평가)
