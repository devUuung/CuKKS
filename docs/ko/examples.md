# CuKKS 예제 가이드

이 문서는 CuKKS 라이브러리의 예제들을 한국어로 설명합니다.

## 사전 요구사항

```bash
# CuKKS 설치
pip install -e .

# GPU 예제를 위해 OpenFHE GPU 백엔드 빌드 필요
# 자세한 빌드 방법은 메인 README 참조
```

## 예제 개요

| 예제 | 설명 | 난이도 |
|------|------|--------|
| [01_basic_encryption.py](../examples/01_basic_encryption.py) | 기본 암호화/복호화 | 초급 |
| [02_mlp_inference.py](../examples/02_mlp_inference.py) | MLP 신경망 추론 | 초급 |
| [03_cnn_inference.py](../examples/03_cnn_inference.py) | CNN 이미지 분류 | 중급 |
| [04_statistics.py](../examples/04_statistics.py) | PP-STAT 통계 함수 | 중급 |
| [05_bootstrapping.py](../examples/05_bootstrapping.py) | GPU 가속 부트스트랩 | 고급 |

---

## 01. 기본 암호화

**파일:** `01_basic_encryption.py`

CKKS 동형암호의 기본 사용법을 배웁니다:
- `CKKSConfig`로 CKKS 컨텍스트 생성
- 벡터 암호화 및 복호화
- 기본 연산: 덧셈, 곱셈, 스칼라 곱셈
- 오차 분석

```python
from ckks.torch_api import CKKSContext, CKKSConfig

config = CKKSConfig(
    poly_mod_degree=8192,      # 링 차원 (슬롯 수 = 4096)
    coeff_mod_bits=[60, 40, 40, 60],  # 계수 모듈러스
    scale_bits=40,             # 인코딩 스케일
)
ctx = CKKSContext(config, enable_gpu=True)

# 암호화
enc_x = ctx.encrypt(x)
enc_y = ctx.encrypt(y)

# 동형 연산
enc_add = enc_x + enc_y       # 덧셈
enc_mul = enc_x * enc_y       # 곱셈
enc_scale = enc_x * 2.5       # 스칼라 곱셈

# 복호화
result = ctx.decrypt(enc_add)
```

### 주요 개념

- **슬롯 (Slot)**: CKKS는 벡터 연산을 지원합니다. `poly_mod_degree // 2`개의 슬롯에 값을 저장할 수 있습니다.
- **스케일 (Scale)**: 실수를 정수로 근사하기 위한 스케일링 팩터입니다.
- **레벨 (Level)**: 남은 연산 가능 깊이입니다. 곱셈마다 레벨이 감소합니다.

---

## 02. MLP 추론

**파일:** `02_mlp_inference.py`

다층 퍼셉트론(MLP)에서 암호화 추론을 수행합니다:
- PyTorch 레이어를 암호화 레이어로 변환
- BSGS 최적화로 빠른 행렬 곱셈
- 제곱 활성화 (CKKS 친화적인 ReLU 대안)

```python
from ckks_torch import CKKSInferenceContext
from ckks_torch.nn import EncryptedLinear, EncryptedSquare

# 컨텍스트 생성
ctx = CKKSInferenceContext(
    max_rotation_dim=64,  # 최대 회전 차원
    use_bsgs=True,        # Baby-step Giant-step 최적화
)

# PyTorch 레이어 → 암호화 레이어 변환
enc_fc1 = EncryptedLinear.from_torch(model.fc1)
enc_square = EncryptedSquare()
enc_fc2 = EncryptedLinear.from_torch(model.fc2)

# 암호화 추론
enc_x = ctx.encrypt(x)
enc_h = enc_fc1(enc_x)      # 첫 번째 FC
enc_h = enc_square(enc_h)   # 제곱 활성화
enc_out = enc_fc2(enc_h)    # 두 번째 FC
```

### 왜 제곱 활성화인가?

CKKS에서 ReLU를 사용하려면 다항식 근사가 필요하여 정확도가 떨어집니다.
제곱(`x²`)은 정확하게 계산할 수 있어 더 나은 정확도를 제공합니다.

---

## 03. CNN 추론

**파일:** `03_cnn_inference.py`

GPU 가속 암호화 CNN 추론:
- Conv2d → Square → AvgPool2d → Flatten → Linear 파이프라인
- im2col 변환으로 효율적인 합성곱
- CNN 전용 컨텍스트 설정
- 회전 기반 풀링

```python
from ckks_torch import CKKSInferenceContext
from ckks_torch.nn import (
    EncryptedConv2d, EncryptedSquare, EncryptedAvgPool2d,
    EncryptedFlatten, EncryptedLinear
)

# CNN 최적화 컨텍스트
ctx = CKKSInferenceContext(
    max_rotation_dim=576,
    use_bsgs=True,
    cnn_config={
        'image_height': 8,
        'image_width': 8,
        'channels': 4,
        'pool_size': 2,
        'pool_stride': 2,
    }
)

# 레이어 변환
enc_conv = EncryptedConv2d.from_torch(model.conv)
enc_square = EncryptedSquare()
enc_pool = EncryptedAvgPool2d.from_torch(model.pool)
enc_flatten = EncryptedFlatten()
enc_fc = EncryptedLinear.from_torch_cnn(model.fc, cnn_layout)

# CNN 입력 암호화 (im2col 자동 적용)
enc_input = ctx.encrypt_cnn_input(image, conv_params)

# 추론 파이프라인
enc_h = enc_conv(enc_input)
enc_h = enc_square(enc_h)
enc_h = enc_pool(enc_h)
enc_h = enc_flatten(enc_h)
enc_out = enc_fc(enc_h)
```

### CNN 레이아웃

암호화 CNN에서는 데이터 레이아웃이 중요합니다:
- **im2col**: 합성곱을 행렬 곱셈으로 변환
- **cnn_layout**: 풀링 후 공간 위치와 채널의 배치 정보

---

## 04. 통계 함수 (PP-STAT)

**파일:** `04_statistics.py`

PP-STAT 논문 기반의 프라이버시 보호 통계 함수:
- `encrypted_mean`: 암호화 평균 계산
- `encrypted_variance`: 암호화 분산 계산
- `encrypted_std`: 암호화 표준편차 (부트스트랩 필요)
- `crypto_inv_sqrt`: 암호화 역제곱근 (1/√x)
- `crypto_inv_sqrt_shallow`: 부트스트랩 없는 역제곱근

```python
from ckks_torch.stats import (
    encrypted_mean, encrypted_variance, encrypted_std,
    crypto_inv_sqrt_shallow
)

enc_x = ctx.encrypt(x)

# 평균과 분산
enc_mean = encrypted_mean(enc_x)
enc_var = encrypted_variance(enc_x)

# 역제곱근 (정규화에 사용)
# domain: 입력값의 예상 범위
enc_inv_sqrt = crypto_inv_sqrt_shallow(enc_var, domain=(1.0, 50.0))
```

### 정규화 패턴

데이터 정규화: `(x - mean) / std`

```python
# 1. 평균 계산
enc_mean = encrypted_mean(enc_x)

# 2. 분산 계산
enc_var = encrypted_variance(enc_x)

# 3. 역표준편차 계산 (1/std = 1/sqrt(var))
enc_var_eps = enc_var.add(epsilon)  # 수치 안정성
enc_inv_std = crypto_inv_sqrt_shallow(enc_var_eps, domain=(1.0, 50.0))

# 4. 정규화: (x - mean) * inv_std
# 실제 구현에서는 브로드캐스트 처리 필요
```

**참고 문헌:**
> Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving Statistical Analysis Framework. CIKM'25.

---

## 05. 부트스트랩

**파일:** `05_bootstrapping.py`

GPU 가속 CKKS 부트스트랩:
- 노이즈 및 레벨 복원
- 깊은 연산 패턴
- 성능 벤치마킹

```python
config = CKKSConfig(
    poly_mod_degree=32768,           # 부트스트랩 권장 크기
    coeff_mod_bits=[60] + [40] * 14 + [60],  # 충분한 깊이
    scale_bits=40,
    enable_bootstrap=True,           # 부트스트랩 활성화
)

ctx = CKKSContext(config, enable_gpu=True)
enc_x = ctx.encrypt(x)

# 많은 연산 수행 후...
enc_result = enc_x
for _ in range(5):
    enc_result = enc_result.mul(enc_result).rescale()

# 레벨이 낮아지면 부트스트랩으로 복원
print(f"부트스트랩 전 레벨: {enc_result.level()}")
enc_result = enc_result.bootstrap()
print(f"부트스트랩 후 레벨: {enc_result.level()}")
```

### 부트스트랩이란?

CKKS 암호문은 연산할 때마다:
1. **노이즈 증가**: 결과의 정확도 감소
2. **레벨 감소**: 더 이상 연산 불가

부트스트랩은 이 두 문제를 해결합니다:
- 노이즈를 초기화
- 레벨을 복원

### 주의사항

- 부트스트랩은 계산 집약적 (수백 ms ~ 수 초)
- `poly_mod_degree >= 32768` 권장
- 메모리 사용량이 큼 (수 GB)
- 가능하면 `crypto_inv_sqrt_shallow` 등 부트스트랩 없는 대안 사용

---

## 예제 실행

```bash
# 개별 예제 실행
python docs/examples/01_basic_encryption.py
python docs/examples/02_mlp_inference.py
python docs/examples/03_cnn_inference.py
python docs/examples/04_statistics.py
python docs/examples/05_bootstrapping.py
```

## 문제 해결

### GPU 사용 불가
```python
ctx = CKKSContext(config, enable_gpu=False)  # CPU 모드 강제
```

### 부트스트랩 오류
- `poly_mod_degree >= 32768` 확인
- `crypto_inv_sqrt_shallow` 사용 고려

### 메모리 부족
- `poly_mod_degree` 줄이기 (예: 32768 → 16384)
- 데이터를 작은 배치로 처리

---

## 추가 참고

- [API 레퍼런스](api.md)
- [핵심 개념](concepts.md)
- [영문 예제 가이드](../examples/README.md)
- [메인 README](../../README.md)
