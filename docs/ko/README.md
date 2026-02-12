# CuKKS

**GPU 가속을 활용한 CKKS 동형암호 기반 PyTorch 호환 암호화 딥러닝 추론 라이브러리**

CuKKS는 학습된 PyTorch 모델을 암호화된 데이터에서 실행할 수 있게 해주며, 모델 정확도를 유지하면서 프라이버시를 보호합니다. CUDA 가속을 지원하는 OpenFHE 기반으로 구축되었습니다.

## 주요 기능

- **PyTorch 호환 API**: 딥러닝 개발자에게 익숙한 인터페이스
- **자동 모델 변환**: 한 줄의 함수 호출로 학습된 PyTorch 모델 변환
- **GPU 가속**: OpenFHE를 통한 CUDA 가속 CKKS 연산
- **다항식 활성화 함수**: ReLU, GELU, SiLU 등의 사전 계산된 근사값
- **BatchNorm 폴딩**: 효율적인 추론을 위한 자동 최적화
- **배치 처리**: SIMD 병렬 처리를 위해 여러 샘플을 하나의 암호문에 패킹
- **유연한 설정**: 보안/성능 트레이드오프에 따른 쉬운 파라미터 조정

## 빠른 시작

### MLP 예제

```python
import torch.nn as nn
import cukks

# 1. PyTorch에서 정상적으로 모델 학습
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
train(model, data)

# 2. 암호화 모델로 변환
enc_model, ctx = cukks.convert(model, use_square_activation=True)

# 3. 입력 암호화 및 추론 실행
enc_input = ctx.encrypt(test_input)
enc_output = enc_model(enc_input)

# 4. 출력 복호화
output = ctx.decrypt(enc_output)
```

### CNN 예제

```python
import torch.nn as nn
import cukks

# 모든 연산을 레이어 속성으로 정의
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()           # x^2로 대체됨
        self.pool1 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()     # 레이어 속성이어야 함
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
enc_model, ctx = cukks.convert(model, use_square_activation=True)

# 암호화 및 추론 실행
enc_input = ctx.encrypt(image)  # shape: (1, 1, 28, 28)
enc_output = enc_model(enc_input)
prediction = ctx.decrypt(enc_output).argmax()
```

> **중요**: `forward()`의 모든 연산은 레이어 속성이어야 합니다 (예: `self.act1`, `self.flatten`). `x ** 2`나 `x.flatten(1)` 같은 인라인 연산은 변환되지 않습니다.

## 설치

### 요구 사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU 가속용)
- CMake 3.16+ (OpenFHE 백엔드 빌드용)
- GCC 9+ 또는 Clang 10+

### 1단계: Python 패키지 설치 (고수준 API)

```bash
git clone https://github.com/devUuung/CuKKS.git
cd CuKKS
pip install -e .
```

### 2단계: OpenFHE 백엔드 빌드 (암호화에 필요)

OpenFHE GPU 소스는 `openfhe-gpu-public/`에 포함되어 있으며, 로컬 빌드가 필요합니다:

```bash
# GPU 지원과 함께 OpenFHE 빌드
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
import cukks

# 백엔드 사용 가능 여부 확인
print(cukks.is_available())  # OpenFHE 백엔드 설치 시 True
print(cukks.get_backend_info())
```

## 문서

- **[API 레퍼런스](api.md)**: 완전한 API 문서
- **[CKKS 개념](concepts.md)**: 동형암호 이해하기
- **[예제](../examples/)**: 동작하는 코드 예제

## 지원 레이어

| PyTorch 레이어 | 암호화 버전 | 비고 |
|--------------|-------------------|-------|
| `nn.Linear` | `EncryptedLinear` | BSGS 최적화와 함께 전체 지원 |
| `nn.Conv2d` | `EncryptedConv2d` | im2col 방식 사용 |
| `nn.ReLU` | `EncryptedReLU` | 다항식 근사 |
| `nn.GELU` | `EncryptedGELU` | 다항식 근사 |
| `nn.SiLU` | `EncryptedSiLU` | 다항식 근사 |
| `nn.Sigmoid` | `EncryptedSigmoid` | 다항식 근사 |
| `nn.Tanh` | `EncryptedTanh` | 다항식 근사 |
| `nn.AvgPool2d` | `EncryptedAvgPool2d` | 전체 지원 |
| `nn.MaxPool2d` | `EncryptedMaxPool2d` | 다항식 근사 |
| `nn.Flatten` | `EncryptedFlatten` | 논리적 reshape |
| `nn.BatchNorm1d/2d` | 폴딩됨 | 이전 레이어에 병합 |
| `nn.Sequential` | `EncryptedSequential` | 전체 지원 |
| `nn.Dropout` | `EncryptedDropout` | 추론 시 no-op |
| `nn.LayerNorm` | `EncryptedLayerNorm` | 복호화-재암호화 방식 |
| `nn.MultiheadAttention` | `EncryptedApproxAttention` | 다항식 소프트맥스 근사 |

## 활성화 함수

CKKS는 다항식 연산만 지원하므로 비선형 활성화 함수는 근사가 필요합니다:

### 옵션 1: 제곱 활성화 (최대 정확도 권장)

```python
# x^2 활성화 사용 - CKKS에서 정확함, 근사 오차 없음
enc_model, ctx = cukks.convert(model, use_square_activation=True)
```

### 옵션 2: 다항식 근사

```python
# ReLU의 체비쇼프 다항식 근사 사용
enc_model, ctx = cukks.convert(
    model,
    use_square_activation=False,
    activation_degree=4  # 높을수록 정확하지만 회로 깊이 증가
)
```

## 설정

### 추론 컨텍스트

```python
from cukks import CKKSInferenceContext, InferenceConfig

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
from cukks.converter import ModelConverter, ConversionOptions

options = ConversionOptions(
    fold_batchnorm=True,        # BN을 이전 레이어에 폴딩
    activation_degree=4,        # 활성화 다항식 차수
    use_square_activation=False # 근사 대신 x^2 사용
)

converter = ModelConverter(options)
enc_model = converter.convert(model)
```

## 예제

### 데모 실행

```bash
# 모델 변환 데모 (GPU 불필요)
python -m cukks.examples.encrypted_inference --demo conversion

# 전체 암호화 추론 (CKKS 백엔드 필요)
python -m cukks.examples.encrypted_inference --demo inference
```

### MNIST 암호화 추론

```bash
# 합성 데이터로 MNIST 예제 실행
python examples/mnist_encrypted.py --hidden 64 --samples 5

# 실제 MNIST 데이터셋 사용
python examples/mnist_encrypted.py --use-mnist --samples 10
```

간소화된 예제는 [docs/examples/mnist.py](../examples/mnist.py)를 참조하세요.

### 사용자 정의 다항식 활성화

```python
from cukks.utils.approximations import chebyshev_coefficients

# 커스텀 ReLU 근사 계산
def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

coeffs = chebyshev_coefficients(relu, degree=7, domain=(-1, 1))

# 모델에서 사용
from cukks.nn import EncryptedPolynomial
custom_activation = EncryptedPolynomial(coeffs)
```

### 배치 처리

```python
# 여러 샘플을 하나의 암호문에 패킹
samples = [torch.randn(784) for _ in range(8)]
enc_batch = ctx.encrypt_batch(samples)

# 8개 샘플에 대해 동시에 추론 실행
enc_output = enc_model(enc_batch)

# 개별 결과 복호화
outputs = ctx.decrypt_batch(enc_output, num_samples=8)
```

## 아키텍처

```
cukks/
├── __init__.py          # 메인 익스포트
├── context.py           # CKKSInferenceContext
├── tensor.py            # EncryptedTensor
├── converter.py         # PyTorch → 암호화 변환
├── batching/            # 배치 처리 유틸리티
│   └── packing.py       # SIMD 배칭용 SlotPacker
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

## 성능 팁

1. **가능하면 제곱 활성화 사용** - CKKS에서 정확함
2. **곱셈 깊이 최소화** - 각 곱셈이 정밀도를 소비
3. **BatchNorm 폴딩** 변환 전 (자동으로 수행됨)
4. **적절한 링 차원 선택** - 크다고 항상 좋은 것은 아님
5. **배치 처리 사용** - SIMD 병렬 처리를 위해 여러 샘플 패킹
6. **매우 깊은 네트워크(>10 레이어)에는 부트스트래핑 고려**
7. **행렬-벡터 곱에 BSGS 최적화 사용** (기본 활성화)

## 제한 사항

- **추론 전용**: 암호화 학습 미지원
- **근사 활성화**: ReLU/GELU는 다항식 근사
- **고정 정밀도**: 누적 오차가 네트워크 깊이에 따라 증가
- **메모리 집약적**: CKKS 연산에 상당한 GPU 메모리 필요

## 문제 해결

### 일반적인 이슈

**메모리 부족 (OOM)**:
- `poly_mod_degree` 줄이기 (예: 16384 대신 8192)
- 가능하면 `mult_depth` 줄이기
- 회전 키 수 줄이기

**낮은 정확도**:
- 더 나은 근사를 위해 `activation_degree` 증가
- `use_square_activation=True` 사용 고려
- 입력을 [-1, 1] 범위로 정규화

**느린 성능**:
- BSGS 최적화 활성화 (기본값)
- 여러 샘플에 대해 배치 처리 사용
- 가능하면 네트워크 깊이 줄이기

## 라이선스

Apache License 2.0

## 인용

연구에 CuKKS를 사용하신다면 인용해 주세요:

```bibtex
@software{cukks,
  title = {CuKKS: PyTorch-compatible Encrypted Deep Learning},
  year = {2024},
  url = {https://github.com/devUuung/CuKKS}
}
```

## 관련 자료

- [OpenFHE](https://github.com/openfheorg/openfhe-development): 기반이 되는 HE 라이브러리
- [CKKS 논문](https://eprint.iacr.org/2016/421): 원본 CKKS 스킴
- [Microsoft SEAL](https://github.com/microsoft/SEAL): 대안 HE 라이브러리
