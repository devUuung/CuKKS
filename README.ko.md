<p align="center">
  <a href="README.md">English</a> |
  <a href="README.ko.md">한국어</a>
</p>

# CuKKS

**CKKS 동형암호를 활용한 PyTorch 호환 암호화 딥러닝 추론 라이브러리 (GPU 가속 지원)**

CuKKS는 암호화된 데이터에서 학습된 PyTorch 모델을 실행할 수 있게 해주며, 모델 정확도를 유지하면서 프라이버시를 보호합니다. OpenFHE 기반으로 CUDA 가속을 지원합니다.

## 주요 기능

- **PyTorch 스타일 API**: 딥러닝 개발자에게 익숙한 인터페이스
- **자동 모델 변환**: 한 줄의 함수 호출로 학습된 PyTorch 모델 변환
- **GPU 가속**: OpenFHE 기반 CUDA 가속 CKKS 연산
- **다항식 활성화 함수**: ReLU, GELU, SiLU 등의 사전 계산된 근사치
- **BatchNorm 폴딩**: 효율적인 추론을 위한 자동 최적화
- **배치 처리**: SIMD 병렬 처리를 위해 여러 샘플을 단일 암호문에 패킹
- **유연한 설정**: 보안/성능 트레이드오프에 맞는 쉬운 파라미터 튜닝

## 빠른 시작

### MLP 예제

```python
import torch.nn as nn
import ckks_torch

# 1. PyTorch에서 일반적으로 모델 학습
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
train(model, data)

# 2. 암호화 모델로 변환
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 3. 입력 암호화 및 추론 실행
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)

# 4. 출력 복호화
output = ctx.decrypt(enc_output)
```

### CNN 예제

```python
import torch.nn as nn
import ckks_torch

# 모든 연산을 레이어 속성으로 정의한 CNN
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()           # x^2로 대체됨
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()     # 반드시 레이어 속성이어야 함
        self.fc = nn.Linear(8 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MNISTCNN()
model.eval()

# 암호화 모델로 변환
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)

# 암호화 및 추론 실행
enc_input = ctx.encrypt(image)  # shape: (1, 1, 28, 28)
enc_output = enc_model(enc_input)
prediction = ctx.decrypt(enc_output).argmax()
```

> **중요**: `forward()`의 모든 연산은 레이어 속성(예: `self.act1`, `self.flatten`)이어야 하며, `x ** 2`나 `x.flatten(1)` 같은 인라인 연산은 사용할 수 없습니다.

## 설치

### 빠른 설치 (권장)

PyPI에서 사전 빌드된 wheel 설치:

```bash
# CPU 전용
pip install cukks

# GPU (CUDA 버전 선택)
pip install cukks-cu118  # CUDA 11.8
pip install cukks-cu121  # CUDA 12.1
pip install cukks-cu124  # CUDA 12.4
pip install cukks-cu128  # CUDA 12.8 (RTX 50xx 지원)
```

### CUDA 버전 호환성

| 패키지 | CUDA | PyTorch | 호환 Docker 이미지 |
|---------|------|---------|-------------------------|
| `cukks-cu118` | 11.8 | 2.0-2.2 | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`<br>`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| `cukks-cu121` | 12.1 | 2.1-2.3 | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`<br>`nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| `cukks-cu124` | 12.4 | 2.4+ | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`<br>`nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04` |
| `cukks-cu128` | 12.8 | 2.6+ | `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04` |
| `cukks` | - | 2.0+ | 모두 (CPU 전용) |

#### 지원 GPU (SM 아키텍처)

| 아키텍처 | GPU | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | CUDA 12.8 |
|--------------|------|-----------|-----------|-----------|-----------|
| sm_70 | V100 | ✅ | ✅ | ✅ | ✅ |
| sm_75 | T4, RTX 20xx | ✅ | ✅ | ✅ | ✅ |
| sm_80 | A100, A30 | ✅ | ✅ | ✅ | ✅ |
| sm_86 | RTX 30xx, A40 | ✅ | ✅ | ✅ | ✅ |
| sm_89 | RTX 40xx, L40 | ✅ | ✅ | ✅ | ✅ |
| sm_90 | H100 | ✅ | ✅ | ✅ | ✅ |
| sm_120 | **RTX 50xx** | ❌ | ❌ | ❌ | ✅ |

#### Docker 사용 예제

```bash
# PyTorch 공식 이미지 (CUDA 12.1)
docker run --gpus all -it pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
pip install cukks-cu121

# NVIDIA CUDA 이미지 (CUDA 11.8)
docker run --gpus all -it nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 bash
apt update && apt install -y python3-pip
pip install cukks-cu118 torch --index-url https://download.pytorch.org/whl/cu118
```

### 소스에서 빌드

#### 요구사항

- Python 3.11
- PyTorch 2.0+
- CUDA Toolkit (cukks-cuXXX 버전과 일치)
- CMake 3.18+
- GCC 9+ 또는 Clang 10+

#### 1단계: Python 패키지 설치 (High-Level API)

```bash
git clone https://github.com/devUuung/CuKKS.git
cd CuKKS
pip install -e .
```

### 2단계: OpenFHE 백엔드 빌드 (암호화에 필요)

OpenFHE GPU 소스는 `openfhe-gpu-public/` 아래에 포함되어 있습니다. 로컬에서 빌드:

```bash
# GPU 지원으로 OpenFHE 빌드
cd openfhe-gpu-public
mkdir build && cd build
cmake .. -DWITH_CUDA=ON
make -j$(nproc)

# Python 바인딩 빌드
cd ../../bindings/openfhe_backend
export LD_LIBRARY_PATH="$PWD/../../openfhe-gpu-public/build/lib:$LD_LIBRARY_PATH"
pip install -e .
```

### 설치 확인

```python
import ckks_torch

# 백엔드 사용 가능 여부 확인
print(ckks_torch.is_available())  # OpenFHE 백엔드 설치 시 True
print(ckks_torch.get_backend_info())
```

## 문서

- **[API 레퍼런스](docs/api.md)**: 전체 API 문서
- **[GPU 가속](docs/gpu-acceleration.md)**: GPU 설정 및 성능 튜닝
- **[CKKS 개념](docs/concepts.md)**: 동형암호 이해하기
- **[예제](docs/examples/)**: 작동하는 코드 예제

## 지원 레이어

| PyTorch 레이어 | 암호화 버전 | 비고 |
|--------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS 최적화로 완전 지원 |
| `nn.Linear` | `EncryptedTTLinear` | 대형 레이어용 TT 분해 (메모리 효율) |
| `nn.Conv2d` | `EncryptedConv2d` | im2col 방식 |
| `nn.Conv2d` | `EncryptedTTConv2d` | 대형 커널용 TT 분해 (메모리 효율) |
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
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | 다항식 softmax 근사 (seq_len=1) |

## 활성화 함수

CKKS는 다항식 연산만 지원하므로 비선형 활성화 함수는 근사해야 합니다:

### 옵션 1: 제곱 활성화 (최대 정확도를 위해 권장)

```python
# x^2 활성화 사용 - CKKS에서 정확, 근사 오차 없음
enc_model, ctx = ckks_torch.convert(model, use_square_activation=True)
```

### 옵션 2: 다항식 근사

```python
# ReLU의 체비쇼프 다항식 근사 사용
enc_model, ctx = ckks_torch.convert(
    model,
    use_square_activation=False,
    activation_degree=4  # 높을수록 정확하지만 회로 깊이 증가
)
```

## 설정

### 추론 컨텍스트

```python
from ckks_torch import CKKSInferenceContext, InferenceConfig

# 모델 기반 자동 설정
ctx = CKKSInferenceContext.for_model(model)

# 또는 수동 설정
config = InferenceConfig(
    poly_mod_degree=16384,      # 링 차원 (2의 거듭제곱)
    scale_bits=40,              # 정밀도 비트
    security_level="128_classic",
    mult_depth=6,               # 곱셈 깊이
    enable_bootstrap=False,     # 매우 깊은 네트워크용
)
ctx = CKKSInferenceContext(config)
```

### 변환 옵션

```python
from ckks_torch.converter import ModelConverter, ConversionOptions

options = ConversionOptions(
    fold_batchnorm=True,        # BN을 이전 레이어에 폴딩
    activation_degree=4,        # 활성화 함수의 다항식 차수
    use_square_activation=False # 근사 대신 x^2 사용
)

converter = ModelConverter(options)
enc_model = converter.convert(model)
```

## 예제

### 데모 실행

```bash
# 모델 변환 데모 (GPU 불필요)
python -m ckks_torch.examples.encrypted_inference --demo conversion

# 전체 암호화 추론 (CKKS 백엔드 필요)
python -m ckks_torch.examples.encrypted_inference --demo inference
```

### MNIST 암호화 추론

```bash
# 합성 데이터로 MNIST 예제 실행
python examples/mnist_encrypted.py --hidden 64 --samples 5

# 실제 MNIST 데이터셋 사용
python examples/mnist_encrypted.py --use-mnist --samples 10
```

간단한 예제는 [docs/examples/mnist.py](docs/examples/mnist.py)를 참조하세요.

### 커스텀 다항식 활성화

```python
from ckks_torch.utils.approximations import chebyshev_coefficients

# 커스텀 ReLU 근사 계산
def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

coeffs = chebyshev_coefficients(relu, degree=7, domain=(-1, 1))

# 모델에서 사용
from ckks_torch.nn import EncryptedPolynomial
custom_activation = EncryptedPolynomial(coeffs)
```

### 배치 처리

```python
# 여러 샘플을 단일 암호문에 패킹
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)

# 8개 샘플 모두 한 번에 추론 실행
enc_output = enc_model(enc_batch)

# 개별 결과 복호화
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

## 아키텍처

```
ckks_torch/
├── __init__.py          # 메인 exports
├── context.py           # CKKSInferenceContext
├── tensor.py            # EncryptedTensor
├── converter.py         # PyTorch → Encrypted 변환
├── batching/            # 배치 처리 유틸리티
│   └── packing.py       # SIMD 배치용 SlotPacker
├── nn/                  # 암호화 신경망 레이어
│   ├── module.py        # 기본 클래스
│   ├── linear.py        # EncryptedLinear
│   ├── conv.py          # EncryptedConv2d (groups, dilation 지원)
│   ├── activations.py   # 다항식 활성화
│   ├── pooling.py       # EncryptedAvgPool2d, EncryptedMaxPool2d
│   ├── batchnorm.py     # BatchNorm (폴딩용)
│   ├── layernorm.py     # EncryptedLayerNorm
│   ├── dropout.py       # EncryptedDropout
│   ├── residual.py      # EncryptedResidualBlock
│   └── attention.py     # EncryptedApproxAttention
└── utils/
    └── approximations.py # 다항식 피팅 유틸리티
```

## GPU 가속

CuKKS는 동형암호 연산에 대한 GPU 가속을 지원합니다:

### GPU 활성화

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(
    poly_mod_degree=8192,
    scale_bits=40,
    rotations=list(range(-16, 17)),
    coeff_mod_bits=[60, 40, 40, 40, 40, 60],
)

# 기본적으로 GPU 활성화
ctx = CKKSContext(config, enable_gpu=True)
print(f"GPU 활성화: {ctx.gpu_enabled}")

# GPU 비활성화
ctx = CKKSContext(config, enable_gpu=False)
```

### GPU 가속 연산

| 연산 | 가속 |
|-----------|-------------|
| Add/Sub | GPU |
| Mul/Square | GPU |
| Rotate | GPU |
| Rescale | GPU |
| Bootstrap | GPU |
| Encrypt/Decrypt | CPU |
| BSGS MatMul | CPU (자동 GPU reload) |

### 지연 동기화

GPU 결과는 필요할 때만 (예: 복호화 시) CPU로 동기화되어, 연쇄 연산에서 ~2배 효율 향상을 제공합니다:

```python
# 효율적: 연산 체인, 한 번만 복호화
enc_result = enc_x.add(enc_x).add(enc_x).add(enc_x)
result = ctx.decrypt(enc_result)  # 여기서 동기화 발생
```

자세한 문서는 [GPU 가속 가이드](docs/gpu-acceleration.md)를 참조하세요.

## 성능 팁

1. **가능하면 제곱 활성화 사용** - CKKS에서 정확함
2. **곱셈 깊이 최소화** - 각 곱셈은 정밀도를 소모
3. **변환 전 BatchNorm 폴딩** (자동으로 수행됨)
4. **적절한 링 차원 선택** - 클수록 슬롯이 많지만 느려짐
5. **배치 처리 사용** - SIMD 병렬 처리를 위해 여러 샘플 패킹
6. **매우 깊은 네트워크 (10+ 레이어)에는 부트스트래핑 고려**
7. **행렬-벡터 곱에 BSGS 최적화 사용** (기본 활성화)

### CNN 전용 최적화

CuKKS는 자동으로 적용되는 여러 CNN 전용 최적화를 포함합니다:

#### Flatten 흡수
`Flatten`의 순열 연산이 다음 `Linear` 레이어의 가중치에 흡수되어, 런타임에서 비용이 큰 matmul 연산을 제거합니다.

```python
# CNN 모델 변환 시 자동 최적화
enc_model, ctx = ckks_torch.convert(cnn_model, optimize_cnn=True)
```

#### Pool 회전 최적화
2x2 평균 풀링에서 희소 행렬 곱셈 대신 회전 기반 합산을 사용하여 HE 연산 수를 줄입니다.

#### 지연 Rescale
Rescale 연산이 필요할 때까지 (예: 다음 곱셈 전) 지연되어, 불필요한 rescale 호출을 줄이고 정밀도 레벨을 보존합니다.

### CNN 성능 벤치마크

8×8 다운샘플링된 MNIST에서 Conv(8ch) → Square → AvgPool(2) → FC(10)로 테스트:

| 최적화 | 시간 | 개선율 |
|--------------|------|-------------|
| 베이스라인 | 3.12s | - |
| + Flatten 흡수 | 2.81s | 10% |
| + Pool 회전 + 지연 Rescale | 2.74s | **12%** |

### 진정한 HE CNN 추론

CuKKS는 모든 연산이 복호화 없이 암호화된 데이터에서 실행되는 *진정한* 동형 CNN 추론을 구현합니다:

```python
# 수동 HE CNN 파이프라인 (고급 사용자용)
ctx = ckks_torch.CKKSInferenceContext(max_rotation_dim=576, use_bsgs=True)

# 암호화 전 im2col 사전 적용
conv_params = [{'kernel_size': (3,3), 'stride': (1,1), 'padding': (1,1), 'out_channels': 8}]
enc_x = ctx.encrypt_cnn_input(image, conv_params)

# 모든 연산이 암호화된 데이터에서 실행
enc_out = EncryptedConv2d.from_torch(conv)(enc_x)   # HE matmul
enc_out = EncryptedSquare()(enc_out)                 # HE square
enc_out = EncryptedAvgPool2d.from_torch(pool)(enc_out)  # HE 회전 기반 pool
enc_out = EncryptedFlatten(absorb_permutation=True)(enc_out)  # No-op
enc_out = EncryptedLinear.from_torch_cnn(fc, cnn_layout)(enc_out)  # 순열된 가중치로 HE matmul

result = ctx.decrypt(enc_out)
```

## 제한사항

- **추론 전용**: 암호화된 학습 미지원
- **근사 활성화**: ReLU/GELU는 다항식 근사
- **고정 정밀도**: 누적 오차가 네트워크 깊이와 함께 증가
- **메모리 집약적**: CKKS 연산은 상당한 GPU 메모리 필요

## 문제 해결

### 일반적인 문제

**메모리 부족 (OOM)**:
- `poly_mod_degree` 감소 (예: 16384 대신 8192)
- 가능하면 `mult_depth` 감소
- 회전 키 수 줄이기

**낮은 정확도**:
- 더 나은 근사를 위해 `activation_degree` 증가
- `use_square_activation=True` 사용 고려
- 입력을 [-1, 1] 범위로 정규화

**느린 성능**:
- BSGS 최적화 활성화 (기본값)
- 여러 샘플에 배치 처리 사용
- 가능하면 네트워크 깊이 감소

## 라이선스

Apache License 2.0

## 인용

연구에서 CuKKS를 사용하시면 다음과 같이 인용해 주세요:

```bibtex
@software{cukks,
  title = {CuKKS: PyTorch-compatible Encrypted Deep Learning},
  year = {2024},
  url = {https://github.com/devUuung/CuKKS}
}
```

## 관련 리소스

- [OpenFHE](https://github.com/openfheorg/openfhe-development): 기반 HE 라이브러리
- [CKKS 논문](https://eprint.iacr.org/2016/421): 원본 CKKS 스킴
- [Microsoft SEAL](https://github.com/microsoft/SEAL): 대안 HE 라이브러리
