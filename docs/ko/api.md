# API 레퍼런스

이 문서는 CKKS-Torch의 완전한 API 문서를 제공합니다.

## 목차

- [핵심 클래스](#핵심-클래스)
  - [CKKSInferenceContext](#ckksinferencecontext)
  - [InferenceConfig](#inferenceconfig)
  - [EncryptedTensor](#encryptedtensor)
- [모델 변환](#모델-변환)
  - [convert()](#convert)
  - [ModelConverter](#modelconverter)
  - [ConversionOptions](#conversionoptions)
  - [estimate_depth()](#estimate_depth)
- [신경망 모듈](#신경망-모듈)
  - [EncryptedModule](#encryptedmodule)
  - [EncryptedLinear](#encryptedlinear)
  - [EncryptedConv2d](#encryptedconv2d)
  - [활성화 함수](#활성화-함수)
  - [EncryptedSequential](#encryptedsequential)
  - [기타 레이어](#기타-레이어)
    - [EncryptedFlatten](#encryptedflatten)
    - [EncryptedAvgPool2d](#encryptedavgpool2d)
    - [EncryptedMaxPool2d](#encryptedmaxpool2d)
    - [EncryptedBatchNorm1d / EncryptedBatchNorm2d](#encryptedbatchnorm1d--encryptedbatchnorm2d)
    - [EncryptedLayerNorm](#encryptedlayernorm)
    - [EncryptedDropout](#encrypteddropout)
    - [EncryptedResidualBlock](#encryptedresidualblock)
    - [EncryptedApproxAttention](#encryptedapproxattention)
- [유틸리티](#유틸리티)
  - [SlotPacker](#slotpacker)

---

## 핵심 클래스

### CKKSInferenceContext

암호화 추론을 위한 고수준 컨텍스트입니다. 암호화 파라미터를 관리하고 텐서 암호화/복호화 메서드를 제공합니다.

```python
from ckks_torch import CKKSInferenceContext, InferenceConfig
```

#### 생성자

```python
CKKSInferenceContext(
    config: Optional[InferenceConfig] = None,
    *,
    device: Optional[str] = None,
    rotations: Optional[List[int]] = None,
    use_bsgs: bool = True,
    max_rotation_dim: Optional[int] = None,
    auto_bootstrap: bool = False,
    bootstrap_threshold: int = 2,
)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `config` | `InferenceConfig` | `None` | 암호화 설정. None이면 기본값 사용. |
| `device` | `str` | `"cuda"` 또는 `"cpu"` | 계산용 디바이스. |
| `rotations` | `List[int]` | 자동 | 생성할 회전 키. None이면 자동 계산. |
| `use_bsgs` | `bool` | `True` | 회전에 Baby-step Giant-step 최적화 사용. |
| `max_rotation_dim` | `int` | `None` | 회전 키의 최대 차원. |
| `auto_bootstrap` | `bool` | `False` | 깊이 소진 시 자동 부트스트래핑 활성화. |
| `bootstrap_threshold` | `int` | `2` | 자동 부트스트랩 트리거 깊이. |

#### 클래스 메서드

##### `for_model()`

특정 모델에 최적화된 컨텍스트를 생성합니다.

```python
ctx = CKKSInferenceContext.for_model(
    model: torch.nn.Module,
    use_bsgs: bool = True,
    **kwargs
) -> CKKSInferenceContext
```

##### `for_depth()`

특정 곱셈 깊이에 대한 컨텍스트를 생성합니다.

```python
ctx = CKKSInferenceContext.for_depth(
    depth: int,
    **kwargs
) -> CKKSInferenceContext
```

##### `load_context()`

저장된 컨텍스트 설정을 로드합니다.

```python
ctx = CKKSInferenceContext.load_context(path: str) -> CKKSInferenceContext
```

#### 인스턴스 메서드

##### `encrypt()`

PyTorch 텐서를 암호화합니다.

```python
enc_tensor = ctx.encrypt(tensor: torch.Tensor) -> EncryptedTensor
```

**파라미터:**
- `tensor`: 암호화할 입력 텐서. 평탄화됩니다.

**반환값:** 암호문을 감싸는 `EncryptedTensor`.

**예외:** 텐서 크기가 사용 가능한 슬롯을 초과하면 `ValueError`.

##### `decrypt()`

암호화된 텐서를 복호화합니다.

```python
tensor = ctx.decrypt(
    encrypted: EncryptedTensor,
    shape: Optional[Sequence[int]] = None
) -> torch.Tensor
```

**파라미터:**
- `encrypted`: 복호화할 암호화된 텐서.
- `shape`: 출력 텐서의 선택적 형상.

**반환값:** 복호화된 PyTorch 텐서.

##### `encrypt_batch()`

슬롯 패킹을 사용하여 여러 샘플을 하나의 암호문으로 암호화합니다.

```python
enc_batch = ctx.encrypt_batch(
    samples: List[torch.Tensor],
    slots_per_sample: Optional[int] = None
) -> EncryptedTensor
```

**파라미터:**
- `samples`: 암호화할 텐서 목록.
- `slots_per_sample`: 샘플당 슬롯 수. None이면 첫 샘플 크기 사용.

**반환값:** 모든 패킹된 샘플을 포함하는 `EncryptedTensor`.

##### `decrypt_batch()`

배치된 암호문을 개별 샘플로 복호화합니다.

```python
samples = ctx.decrypt_batch(
    encrypted: EncryptedTensor,
    num_samples: Optional[int] = None,
    sample_shape: Optional[Sequence[int]] = None
) -> List[torch.Tensor]
```

##### `save_context()`

컨텍스트 설정을 저장합니다.

```python
ctx.save_context(path: str) -> None
```

#### 속성

| 속성 | 타입 | 설명 |
|----------|------|-------------|
| `num_slots` | `int` | 사용 가능한 평문 슬롯 수 |
| `config` | `InferenceConfig` | 설정 객체 |
| `device` | `str` | 계산용 디바이스 |
| `auto_bootstrap` | `bool` | 자동 부트스트랩 활성화 여부 |
| `bootstrap_threshold` | `int` | 자동 부트스트랩용 깊이 임계값 |
| `backend` | `CKKSContext` | 기본 CKKS 컨텍스트 (지연 초기화) |

---

### InferenceConfig

CKKS 암호화 파라미터용 설정 데이터클래스입니다.

```python
from ckks_torch import InferenceConfig
```

#### 생성자

```python
InferenceConfig(
    poly_mod_degree: int = 16384,
    scale_bits: int = 40,
    security_level: str = "128_classic",
    mult_depth: int = 4,
    enable_bootstrap: bool = False,
)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `poly_mod_degree` | `int` | `16384` | 링 차원. 일반적 값: 8192, 16384, 32768. |
| `scale_bits` | `int` | `40` | 인코딩 정밀도 비트. |
| `security_level` | `str` | `"128_classic"` | 보안 수준: "128_classic", "192_classic", "256_classic". |
| `mult_depth` | `int` | `4` | 회로의 곱셈 깊이. |
| `enable_bootstrap` | `bool` | `False` | 깊은 네트워크용 부트스트래핑 활성화. |

#### 클래스 메서드

##### `for_depth()`

특정 깊이에 최적화된 설정을 생성합니다.

```python
config = InferenceConfig.for_depth(depth: int, **kwargs) -> InferenceConfig
```

##### `for_model()`

모델을 분석하고 최적의 설정을 생성합니다.

```python
config = InferenceConfig.for_model(model: torch.nn.Module, **kwargs) -> InferenceConfig
```

#### 속성

| 속성 | 타입 | 설명 |
|----------|------|-------------|
| `num_slots` | `int` | 평문 슬롯 수 (poly_mod_degree // 2) |
| `coeff_mod_bits` | `Tuple[int, ...]` | 계수 모듈러스 비트 크기 |

---

### EncryptedTensor

텐서와 유사한 연산을 가진 CKKS 암호화 데이터용 텐서 래퍼입니다.

```python
from ckks_torch import EncryptedTensor
```

#### 속성

| 속성 | 타입 | 설명 |
|----------|------|-------------|
| `shape` | `Tuple[int, ...]` | 텐서의 논리적 형상 |
| `size` | `int` | 총 요소 수 |
| `ndim` | `int` | 차원 수 |
| `context` | `CKKSInferenceContext` | 연관된 컨텍스트 |
| `depth` | `int` | 소비된 현재 곱셈 깊이 |
| `level` | `int` | 남은 곱셈 레벨 |
| `scale` | `float` | 현재 스케일 팩터 |
| `metadata` | `dict` | 암호문 메타데이터 |

#### 산술 연산

```python
# 덧셈 (암호문 + 암호문 또는 암호문 + 평문)
result = enc_tensor.add(other: Union[EncryptedTensor, torch.Tensor, float]) -> EncryptedTensor
result = enc_tensor + other  # 연산자 형태

# 뺄셈
result = enc_tensor.sub(other) -> EncryptedTensor
result = enc_tensor - other

# 곱셈
result = enc_tensor.mul(other) -> EncryptedTensor
result = enc_tensor * other

# 부정
result = enc_tensor.neg() -> EncryptedTensor
result = -enc_tensor

# 스칼라로 나눗셈
result = enc_tensor.div(divisor: float) -> EncryptedTensor

# 제곱
result = enc_tensor.square() -> EncryptedTensor
```

#### CKKS 특화 연산

```python
# 스케일 성장 관리를 위한 리스케일
result = enc_tensor.rescale() -> EncryptedTensor

# 슬롯 회전
result = enc_tensor.rotate(steps: int) -> EncryptedTensor

# 모든 슬롯 합계
result = enc_tensor.sum_slots() -> EncryptedTensor

# 복소 켤레
result = enc_tensor.conjugate() -> EncryptedTensor

# 암호문 새로고침을 위한 부트스트랩
result = enc_tensor.bootstrap() -> EncryptedTensor

# 임계값 도달 시 자동 부트스트랩
result = enc_tensor.maybe_bootstrap(context) -> EncryptedTensor
```

#### 행렬 연산

```python
# 행렬-벡터 곱
result = enc_tensor.matmul(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> EncryptedTensor

# 다항식 평가
result = enc_tensor.poly_eval(coeffs: Sequence[float]) -> EncryptedTensor
```

#### 형상 연산

```python
# reshape (논리적만)
result = enc_tensor.view(*shape) -> EncryptedTensor
result = enc_tensor.reshape(shape) -> EncryptedTensor

# 평탄화
result = enc_tensor.flatten() -> EncryptedTensor

# 복제
result = enc_tensor.clone() -> EncryptedTensor
```

#### 복호화

```python
# 연관된 컨텍스트를 사용하여 복호화
tensor = enc_tensor.decrypt(shape: Optional[Sequence[int]] = None) -> torch.Tensor
```

#### 직렬화

```python
# 암호화된 텐서 저장
enc_tensor.save(path: str) -> None

# 암호화된 텐서 로드
enc_tensor = EncryptedTensor.load(path: str, context: CKKSInferenceContext) -> EncryptedTensor
```

---

## 모델 변환

### convert()

모델 변환의 메인 진입점입니다. PyTorch 모델을 암호화 버전으로 변환합니다.

```python
from ckks_torch import convert

enc_model, ctx = convert(
    model: torch.nn.Module,
    ctx: Optional[CKKSInferenceContext] = None,
    *,
    fold_batchnorm: bool = True,
    activation_degree: int = 4,
    use_square_activation: bool = False,
) -> Tuple[EncryptedModule, CKKSInferenceContext]
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `model` | `torch.nn.Module` | 필수 | 변환할 PyTorch 모델 |
| `ctx` | `CKKSInferenceContext` | `None` | 사용할 컨텍스트. None이면 자동 생성. |
| `fold_batchnorm` | `bool` | `True` | BatchNorm을 이전 레이어에 폴딩 |
| `activation_degree` | `int` | `4` | 활성화 근사용 다항식 차수 |
| `use_square_activation` | `bool` | `False` | 모든 활성화를 x²로 대체 |

**반환값:** (암호화_모델, 컨텍스트) 튜플

**예제:**

```python
import torch.nn as nn
import ckks_torch

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

enc_model, ctx = ckks_torch.convert(model)
```

---

### ModelConverter

모델 변환에 대한 세밀한 제어를 위한 클래스입니다.

```python
from ckks_torch.converter import ModelConverter, ConversionOptions
```

#### 생성자

```python
converter = ModelConverter(options: Optional[ConversionOptions] = None)
```

#### 메서드

##### `convert()`

```python
enc_model = converter.convert(
    model: torch.nn.Module,
    ctx: Optional[CKKSInferenceContext] = None
) -> EncryptedModule
```

---

### ConversionOptions

모델 변환용 설정입니다.

```python
from ckks_torch.converter import ConversionOptions

options = ConversionOptions(
    fold_batchnorm: bool = True,
    activation_degree: int = 4,
    activation_map: Optional[Dict[Type, Type]] = None,
    use_square_activation: bool = False,
)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `fold_batchnorm` | `bool` | `True` | BatchNorm을 이전 레이어에 폴딩 |
| `activation_degree` | `int` | `4` | 활성화용 기본 다항식 차수 |
| `activation_map` | `Dict` | `None` | PyTorch에서 암호화 활성화로의 커스텀 매핑 |
| `use_square_activation` | `bool` | `False` | 모든 활성화를 x²로 대체 |

---

### estimate_depth()

모델에 필요한 곱셈 깊이를 추정합니다.

```python
from ckks_torch import estimate_depth

depth = estimate_depth(model: torch.nn.Module) -> int
```

---

## 신경망 모듈

### EncryptedModule

모든 암호화 신경망 모듈의 기본 클래스입니다.

```python
from ckks_torch.nn import EncryptedModule
```

#### 추상 메서드

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    """암호화된 입력에 대한 순전파."""
    pass
```

#### 메서드

```python
# 모듈을 호출 가능하게 만들기
result = module(x)  # module.forward(x)와 동일

# 모듈 순회
for mod in module.modules():
    ...

for name, mod in module.named_modules():
    ...

# 자식 순회
for child in module.children():
    ...

# 파라미터 순회
for param in module.parameters():
    ...

# 곱셈 깊이 추정
depth = module.mult_depth() -> int
```

---

### EncryptedLinear

암호화된 완전 연결 레이어입니다.

```python
from ckks_torch.nn import EncryptedLinear
```

#### 생성자

```python
EncryptedLinear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
)
```

#### 클래스 메서드

```python
# PyTorch Linear 레이어에서 생성
enc_linear = EncryptedLinear.from_torch(linear: torch.nn.Linear) -> EncryptedLinear
```

#### 속성

| 속성 | 타입 | 설명 |
|----------|------|-------------|
| `in_features` | `int` | 입력 차원 |
| `out_features` | `int` | 출력 차원 |
| `weight` | `torch.Tensor` | 가중치 행렬 |
| `bias` | `torch.Tensor` | 편향 벡터 (None일 수 있음) |

---

### EncryptedConv2d

im2col 방식을 사용하는 암호화된 2D 합성곱 레이어입니다.

```python
from ckks_torch.nn import EncryptedConv2d
```

#### 생성자

```python
EncryptedConv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `in_channels` | `int` | 필수 | 입력 채널 수 |
| `out_channels` | `int` | 필수 | 출력 채널 수 |
| `kernel_size` | `int` 또는 `Tuple[int, int]` | 필수 | 합성곱 커널 크기 |
| `weight` | `torch.Tensor` | 필수 | 합성곱 커널 가중치 |
| `bias` | `torch.Tensor` | `None` | 선택적 편향 벡터 |
| `stride` | `int` 또는 `Tuple[int, int]` | `1` | 합성곱 보폭 |
| `padding` | `int` 또는 `Tuple[int, int]` | `0` | 입력에 추가되는 패딩 |
| `groups` | `int` | `1` | 입력에서 출력 채널로의 블록 연결 수 |
| `dilation` | `int` 또는 `Tuple[int, int]` | `1` | 커널 요소 간 간격 |

#### 클래스 메서드

```python
enc_conv = EncryptedConv2d.from_torch(conv: torch.nn.Conv2d) -> EncryptedConv2d
```

---

### 활성화 함수

모든 활성화 모듈은 다항식 근사를 지원합니다.

#### EncryptedSquare

정확한 x² 활성화 (정확도 권장).

```python
from ckks_torch.nn import EncryptedSquare

activation = EncryptedSquare()
```

#### EncryptedReLU

ReLU의 다항식 근사입니다.

```python
from ckks_torch.nn import EncryptedReLU

activation = EncryptedReLU(
    degree: int = 4,
    domain: tuple = (-1, 1),
    method: str = "chebyshev"  # 또는 "minimax"
)
```

#### EncryptedGELU

GELU의 다항식 근사입니다.

```python
from ckks_torch.nn import EncryptedGELU

activation = EncryptedGELU(degree: int = 4)
```

#### EncryptedSiLU

SiLU (Swish)의 다항식 근사입니다.

```python
from ckks_torch.nn import EncryptedSiLU

activation = EncryptedSiLU(degree: int = 4)
```

#### EncryptedSigmoid

시그모이드의 다항식 근사입니다.

```python
from ckks_torch.nn import EncryptedSigmoid

activation = EncryptedSigmoid(degree: int = 4)
```

#### EncryptedTanh

tanh의 다항식 근사입니다.

```python
from ckks_torch.nn import EncryptedTanh

activation = EncryptedTanh(degree: int = 5)
```

#### EncryptedPolynomial

커스텀 다항식 활성화입니다.

```python
from ckks_torch.nn import EncryptedPolynomial

# coeffs = [a0, a1, a2, ...] for a0 + a1*x + a2*x² + ...
activation = EncryptedPolynomial(coeffs: Sequence[float])
```

---

### EncryptedSequential

암호화 모듈의 순차 실행을 위한 컨테이너입니다.

```python
from ckks_torch.nn import EncryptedSequential

model = EncryptedSequential(
    EncryptedLinear(...),
    EncryptedReLU(),
    EncryptedLinear(...),
)

# 또는 명명된 모듈로
model = EncryptedSequential(OrderedDict([
    ('fc1', EncryptedLinear(...)),
    ('act1', EncryptedReLU()),
    ('fc2', EncryptedLinear(...)),
]))
```

---

### 기타 레이어

#### EncryptedFlatten

```python
from ckks_torch.nn import EncryptedFlatten

flatten = EncryptedFlatten(start_dim: int = 1, end_dim: int = -1)
```

#### EncryptedAvgPool2d

```python
from ckks_torch.nn import EncryptedAvgPool2d

pool = EncryptedAvgPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0
)

# PyTorch 레이어에서 생성
pool = EncryptedAvgPool2d.from_torch(avg_pool: torch.nn.AvgPool2d)
```

**파라미터:**
- `kernel_size`: 풀링 윈도우 크기.
- `stride`: 풀링 보폭 (기본값: kernel_size와 동일).
- `padding`: 추가할 패딩 (기본값: 0).

**참고:** 4D 입력은 더 이상 지원되지 않으며, CNN 입력을 전처리하려면 `encrypt_cnn_input()`을 사용하세요. 2x2 풀링의 경우 회전 기반 최적화가 자동으로 적용되어 더 나은 성능을 제공합니다.

#### EncryptedMaxPool2d

다항식 근사를 사용한 근사 최대 풀링입니다.

```python
from ckks_torch.nn import EncryptedMaxPool2d

pool = EncryptedMaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    degree: int = 4,
)

# PyTorch 레이어에서 생성
pool = EncryptedMaxPool2d.from_torch(max_pool: torch.nn.MaxPool2d)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `kernel_size` | `int` 또는 `Tuple[int, int]` | 필수 | 풀링 윈도우 크기 |
| `stride` | `int` 또는 `Tuple[int, int]` | `None` | 풀링 보폭. 기본값은 `kernel_size`. |
| `padding` | `int` 또는 `Tuple[int, int]` | `0` | 추가할 패딩 |
| `degree` | `int` | `4` | |x| 근사를 위한 다항식 차수. 높을수록 정확하지만 회로 깊이 증가. |

**참고:** 최대 풀링은 `max(a, b) ≈ (a + b + |a - b|) / 2`를 사용하여 근사됩니다. |x|는 다항식 피팅으로 근사합니다. 4D 입력은 더 이상 지원되지 않으며, CNN 입력을 전처리하려면 `encrypt_cnn_input()`을 사용하세요. 정확도는 입력 정규화에 따라 달라지며, [-1, 1] 범위의 값에서 최적입니다.

#### EncryptedBatchNorm1d / EncryptedBatchNorm2d

참고: BatchNorm은 일반적으로 변환 중에 이전 레이어로 폴딩됩니다.

```python
from ckks_torch.nn import EncryptedBatchNorm1d, EncryptedBatchNorm2d

bn = EncryptedBatchNorm1d.from_torch(bn: torch.nn.BatchNorm1d)
```

#### EncryptedLayerNorm

순수 HE 다항식 근사를 사용한 암호화 레이어 정규화입니다.

```python
from ckks_torch.nn import EncryptedLayerNorm

ln = EncryptedLayerNorm(
    normalized_shape: Union[int, List[int], Tuple[int, ...]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
)

# PyTorch 레이어에서 생성
ln = EncryptedLayerNorm.from_torch(layer_norm: torch.nn.LayerNorm)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `normalized_shape` | `int` 또는 `List[int]` | 필수 | 정규화를 위한 입력 형상 |
| `weight` | `torch.Tensor` | `None` | 학습 가능한 스케일 파라미터 (감마) |
| `bias` | `torch.Tensor` | `None` | 학습 가능한 시프트 파라미터 (베타) |
| `eps` | `float` | `1e-5` | 수치 안정성 상수 |

**참고:** LayerNorm은 평균/분산 계산을 위해 sum_and_broadcast를 사용하고 1/sqrt(var+eps)를 위해 체비쇼프 15차 다항식을 사용하는 순수 HE 다항식 근사를 사용합니다. 곱셈 깊이: ~18 (역제곱근 다항식 평가 포함).

#### EncryptedDropout

암호화 추론을 위한 드롭아웃 레이어입니다. 추론 시에는 no-op (입력을 변경 없이 반환)으로 동작합니다.

```python
from ckks_torch.nn import EncryptedDropout

dropout = EncryptedDropout(p: float = 0.5)

# PyTorch 레이어에서 생성
dropout = EncryptedDropout.from_torch(dropout_layer: torch.nn.Dropout)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `p` | `float` | `0.5` | 드롭아웃 확률. 호환성을 위해 저장되지만 추론 시 효과 없음. |

**참고:** 추론 중에는 드롭아웃이 비활성화됩니다 (no-op). 이 모듈은 `convert()`에서 `nn.Dropout` 레이어를 올바르게 처리하기 위한 모델 변환 호환성을 위해 존재합니다.

#### EncryptedResidualBlock

잔차 연결 블록입니다.

```python
from ckks_torch.nn import EncryptedResidualBlock

block = EncryptedResidualBlock(main_branch: EncryptedModule)
```

#### EncryptedApproxAttention

순수 HE 암호문-암호문 연산을 사용하는 암호화된 트랜스포머 추론을 위한 근사 멀티헤드 어텐션입니다. exp(x)의 테일러 전개를 통한 다항식 근사로 소프트맥스를 구현합니다.

```python
from ckks_torch.nn import EncryptedApproxAttention

attention = EncryptedApproxAttention(
    embed_dim: int,
    num_heads: int,
    softmax_degree: int = 4,
)

# PyTorch 레이어에서 생성
attention = EncryptedApproxAttention.from_torch(
    attention: torch.nn.MultiheadAttention,
    softmax_degree: int = 4,
)
```

**파라미터:**

| 이름 | 타입 | 기본값 | 설명 |
|------|------|---------|-------------|
| `embed_dim` | `int` | 필수 | 총 임베딩 차원 |
| `num_heads` | `int` | 필수 | 어텐션 헤드 수. `embed_dim`은 `num_heads`로 나누어떨어져야 합니다. |
| `softmax_degree` | `int` | `4` | exp(x) 테일러 근사를 위한 다항식 차수. 높을수록 정확하지만 회로 깊이 증가. |

**메서드:**

- `forward(x)`: 셀프 어텐션 — `x`를 쿼리, 키, 밸류로 사용합니다. 순수 HE 모드에서는 seq_len=1만 지원합니다.
- `forward_attention(query, key, value)`: 별도의 Q, K, V 입력을 사용한 전체 어텐션. 순수 HE 모드에서는 seq_len=1만 지원합니다.
- `mult_depth()`: 추정 곱셈 깊이 반환 (프로젝션 + Q@K^T + 소프트맥스 다항식 + attn@V + 출력 프로젝션 포함).

**참고:** 이것은 Q@K^T를 위해 암호문-암호문 곱셈을 사용하고 어텐션 집계를 위해 sum_and_broadcast를 사용하는 순수 HE 다항식 근사입니다. 정확도는 입력 범위와 다항식 차수에 따라 달라집니다. 어텐션 스코어가 작은 범위(예: [-2, 2])로 정규화되었을 때 최적의 결과를 얻습니다. `from_torch()`는 `nn.MultiheadAttention`에서 Q, K, V 및 출력 프로젝션 가중치를 추출합니다. 순수 HE 모드에서는 seq_len=1로 제한됩니다.

---

## 유틸리티

### SlotPacker

여러 샘플을 CKKS 슬롯에 패킹하기 위한 유틸리티입니다.

```python
from ckks_torch.batching import SlotPacker

packer = SlotPacker(
    slots_per_sample: int,
    total_slots: int
)
```

#### 메서드

```python
# 샘플을 단일 텐서로 패킹
packed = packer.pack(samples: List[torch.Tensor]) -> torch.Tensor

# 텐서를 샘플로 언패킹
samples = packer.unpack(
    packed: torch.Tensor,
    num_samples: int
) -> List[torch.Tensor]
```

#### 속성

| 속성 | 타입 | 설명 |
|----------|------|-------------|
| `slots_per_sample` | `int` | 샘플당 할당된 슬롯 |
| `total_slots` | `int` | 총 사용 가능한 슬롯 |
| `max_samples` | `int` | 패킹 가능한 최대 샘플 수 |

---

## 모듈 함수

### 최상위 함수

```python
import ckks_torch

# 백엔드 사용 가능 여부 확인
available = ckks_torch.is_available() -> bool

# 백엔드 정보 가져오기
info = ckks_torch.get_backend_info() -> dict
# 반환값: {"backend": "openfhe-gpu", "available": True, "cuda": True}
```
