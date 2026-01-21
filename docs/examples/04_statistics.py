#!/usr/bin/env python3
"""
PP-STAT 암호화 통계 함수 예제

PP-STAT 논문 기반의 암호화 통계 함수를 사용합니다.
- encrypted_mean: 암호화 평균
- encrypted_variance: 암호화 분산
- encrypted_std: 암호화 표준편차
- crypto_inv_sqrt: 암호화 역제곱근 (1/sqrt(x))

Reference:
    Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
    Statistical Analysis Framework. CIKM'25.
"""

import torch
from ckks.torch_api import CKKSContext, CKKSConfig
from ckks_torch.stats import (
    encrypted_mean,
    encrypted_variance,
    encrypted_std,
    crypto_inv_sqrt,
    crypto_inv_sqrt_shallow,
)


def example_basic_statistics():
    """기본 통계 함수 예제: 평균, 분산"""
    print("=" * 60)
    print("1. 기본 통계 함수 (평균, 분산)")
    print("=" * 60)
    
    # 1. CKKS 설정 (통계 연산용)
    # 분산 계산에는 depth 2 필요 (제곱 + 스케일링)
    config = CKKSConfig(
        poly_mod_degree=8192,
        coeff_mod_bits=[60, 40, 40, 40, 60],  # 충분한 깊이
        scale_bits=40,
        rotations=list(range(-512, 513)),  # 합계 회전 키
        security_level=None,
    )
    
    ctx = CKKSContext(config, enable_gpu=True)
    print(f"GPU 활성화: {ctx.gpu_enabled}")
    
    # 2. 테스트 데이터 (1024 이하 크기)
    # PP-STAT v1은 1024 슬롯 제한이 있음
    n = 100
    torch.manual_seed(42)
    x = torch.randn(n, dtype=torch.float64) * 2 + 5  # mean≈5, std≈2
    print(f"\n데이터 크기: {n}")
    
    # 3. 평문 통계
    plain_mean = x.mean().item()
    plain_var = x.var(unbiased=False).item()  # population variance
    print(f"\n평문 평균: {plain_mean:.6f}")
    print(f"평문 분산: {plain_var:.6f}")
    
    # 4. 암호화
    enc_x = ctx.encrypt(x)
    print(f"\n암호화 완료")
    
    # 5. 암호화 평균
    enc_mean = encrypted_mean(enc_x)
    he_mean = ctx.decrypt(enc_mean)[0].item()
    mean_error = abs(plain_mean - he_mean)
    print(f"\n=== 암호화 평균 ===")
    print(f"암호화 평균: {he_mean:.6f}")
    print(f"오차: {mean_error:.2e}")
    
    # 6. 암호화 분산
    enc_var = encrypted_variance(enc_x)
    he_var = ctx.decrypt(enc_var)[0].item()
    var_error = abs(plain_var - he_var)
    print(f"\n=== 암호화 분산 ===")
    print(f"암호화 분산: {he_var:.6f}")
    print(f"오차: {var_error:.2e}")
    
    # 7. 결과 검증
    if mean_error < 1e-4 and var_error < 1e-3:
        print("\n✅ 기본 통계 함수 테스트 통과!")
    else:
        print("\n⚠️ 오차 확인 필요")


def example_inv_sqrt_shallow():
    """crypto_inv_sqrt_shallow 예제: 부트스트랩 없이 역제곱근"""
    print("\n" + "=" * 60)
    print("2. 역제곱근 (Shallow 버전 - 부트스트랩 불필요)")
    print("=" * 60)
    
    # 1. CKKS 설정 (부트스트랩 없음)
    config = CKKSConfig(
        poly_mod_degree=16384,  # 더 큰 링 차원 (다항식 깊이용)
        coeff_mod_bits=[60] + [40] * 8 + [60],  # 충분한 깊이
        scale_bits=40,
        security_level=None,
    )
    
    ctx = CKKSContext(config, enable_gpu=True)
    print(f"GPU 활성화: {ctx.gpu_enabled}")
    
    # 2. 테스트 데이터 (domain: [1.0, 10.0])
    # crypto_inv_sqrt_shallow는 좁은 도메인에서 잘 동작
    x = torch.tensor([1.0, 2.0, 4.0, 9.0, 10.0], dtype=torch.float64)
    expected = 1.0 / torch.sqrt(x)  # [1.0, 0.707, 0.5, 0.333, 0.316]
    print(f"\n입력: {x.tolist()}")
    print(f"기대값 (1/sqrt(x)): {expected.tolist()}")
    
    # 3. 암호화
    enc_x = ctx.encrypt(x)
    
    # 4. 암호화 역제곱근 (shallow 버전)
    # domain=(1.0, 10.0)은 좁아서 정확도가 좋음
    enc_result = crypto_inv_sqrt_shallow(enc_x, domain=(1.0, 10.0))
    he_result = ctx.decrypt(enc_result)[:5]
    
    print(f"\n암호화 결과: {he_result.tolist()}")
    
    # 5. 오차 분석
    abs_errors = (he_result - expected).abs()
    rel_errors = abs_errors / expected
    mre = rel_errors.mean().item()
    max_re = rel_errors.max().item()
    
    print(f"\n=== 오차 분석 ===")
    print(f"평균 상대 오차 (MRE): {mre:.2e}")
    print(f"최대 상대 오차: {max_re:.2e}")
    
    if mre < 0.05:  # 5% 이하
        print("\n✅ 역제곱근 (shallow) 테스트 통과!")
    else:
        print("\n⚠️ 정확도 확인 필요")


def example_std_with_bootstrap():
    """encrypted_std 예제: 부트스트랩 포함 표준편차"""
    print("\n" + "=" * 60)
    print("3. 표준편차 (부트스트랩 포함)")
    print("=" * 60)
    print("※ 이 예제는 enable_bootstrap=True 설정이 필요합니다.")
    print("※ GPU 부트스트랩이 불안정한 경우 건너뛸 수 있습니다.")
    
    try:
        # 1. CKKS 설정 (부트스트랩 활성화)
        config = CKKSConfig(
            poly_mod_degree=32768,
            coeff_mod_bits=[60] + [40] * 10 + [60],
            scale_bits=40,
            rotations=list(range(-512, 513)),
            enable_bootstrap=True,  # 부트스트랩 활성화
            security_level=None,
        )
        
        ctx = CKKSContext(config, enable_gpu=True)
        print(f"\nGPU 활성화: {ctx.gpu_enabled}")
        print(f"부트스트랩 활성화: {config.enable_bootstrap}")
        
        # 2. 테스트 데이터
        n = 50
        torch.manual_seed(42)
        x = torch.randn(n, dtype=torch.float64) * 3 + 10  # mean≈10, std≈3
        
        plain_std = x.std(unbiased=False).item()
        print(f"\n데이터 크기: {n}")
        print(f"평문 표준편차: {plain_std:.6f}")
        
        # 3. 암호화
        enc_x = ctx.encrypt(x)
        
        # 4. 암호화 표준편차
        # epsilon=0.1은 수치 안정성을 위해 필요
        enc_std = encrypted_std(enc_x, epsilon=0.1)
        he_std = ctx.decrypt(enc_std)[0].item()
        
        # epsilon을 고려한 기대값: sqrt(var + epsilon)
        plain_var = x.var(unbiased=False).item()
        expected_std = (plain_var + 0.1) ** 0.5
        
        print(f"\n=== 암호화 표준편차 ===")
        print(f"암호화 결과: {he_std:.6f}")
        print(f"기대값 (sqrt(var+ε)): {expected_std:.6f}")
        print(f"오차: {abs(he_std - expected_std):.2e}")
        
        if abs(he_std - expected_std) < 0.1:
            print("\n✅ 표준편차 테스트 통과!")
        else:
            print("\n⚠️ 오차 확인 필요")
            
    except RuntimeError as e:
        if "bootstrap" in str(e).lower():
            print(f"\n⚠️ 부트스트랩 오류 (OpenFHE GPU 제한):")
            print(f"   {e}")
            print("   → crypto_inv_sqrt_shallow 사용을 권장합니다.")
        else:
            raise


def example_normalization():
    """정규화 예제: (x - mean) / std 패턴"""
    print("\n" + "=" * 60)
    print("4. 데이터 정규화 패턴")
    print("=" * 60)
    
    # 1. CKKS 설정
    config = CKKSConfig(
        poly_mod_degree=16384,
        coeff_mod_bits=[60] + [40] * 8 + [60],
        scale_bits=40,
        rotations=list(range(-512, 513)),
        security_level=None,
    )
    
    ctx = CKKSContext(config, enable_gpu=True)
    
    # 2. 테스트 데이터
    n = 100
    torch.manual_seed(42)
    x = torch.randn(n, dtype=torch.float64) * 5 + 20  # mean≈20, std≈5
    
    # 3. 평문 정규화
    plain_mean = x.mean()
    plain_std = x.std(unbiased=False)
    plain_normalized = (x - plain_mean) / plain_std
    
    print(f"데이터 통계: mean={plain_mean:.2f}, std={plain_std:.2f}")
    print(f"정규화 후: mean≈0, std≈1")
    
    # 4. 암호화 정규화 (shallow 버전 사용)
    enc_x = ctx.encrypt(x)
    
    # 4a. 평균 계산
    enc_mean = encrypted_mean(enc_x)
    he_mean_value = ctx.decrypt(enc_mean)[0].item()
    
    # 4b. 분산 계산 후 역제곱근 (shallow)
    enc_var = encrypted_variance(enc_x)
    he_var_value = ctx.decrypt(enc_var)[0].item()
    
    # 분산에 epsilon 추가 (수치 안정성)
    epsilon = 0.1
    enc_var_eps = enc_var.add(epsilon)
    
    # 역제곱근 (domain이 분산 범위에 맞게 설정)
    # 분산이 약 25 정도이므로 domain=(1.0, 50.0) 사용
    enc_inv_std = crypto_inv_sqrt_shallow(enc_var_eps, domain=(1.0, 50.0))
    he_inv_std_value = ctx.decrypt(enc_inv_std)[0].item()
    
    print(f"\n=== 암호화 통계 ===")
    print(f"암호화 평균: {he_mean_value:.4f} (평문: {plain_mean:.4f})")
    print(f"암호화 분산: {he_var_value:.4f} (평문: {plain_std**2:.4f})")
    print(f"암호화 1/std: {he_inv_std_value:.4f} (평문: {1/plain_std:.4f})")
    
    # 5. 정규화 수행: (x - mean) * (1/std)
    # 실제로는 브로드캐스트가 필요하지만, 개념 설명용
    print(f"\n※ 전체 정규화는 mean/std를 브로드캐스트하여 수행합니다.")
    print(f"   normalized = (enc_x - mean) * inv_std")
    print("\n✅ 정규화 패턴 데모 완료!")


def main():
    """모든 예제 실행"""
    print("\n" + "=" * 60)
    print("PP-STAT 암호화 통계 함수 예제")
    print("=" * 60)
    print("""
이 예제는 PP-STAT 논문의 암호화 통계 함수를 보여줍니다:

1. encrypted_mean: 암호화 데이터의 평균 계산
2. encrypted_variance: 암호화 데이터의 분산 계산
3. encrypted_std: 암호화 데이터의 표준편차 (부트스트랩 필요)
4. crypto_inv_sqrt: 암호화 역제곱근 (1/sqrt(x))
5. crypto_inv_sqrt_shallow: 부트스트랩 없는 역제곱근

Reference:
    Choi, H. (2025). PP-STAT: An Efficient Privacy-Preserving
    Statistical Analysis Framework. CIKM'25.
""")
    
    # 예제 1: 기본 통계
    example_basic_statistics()
    
    # 예제 2: 역제곱근 (shallow)
    example_inv_sqrt_shallow()
    
    # 예제 3: 표준편차 (부트스트랩)
    # 부트스트랩이 불안정할 수 있으므로 선택적 실행
    # example_std_with_bootstrap()
    
    # 예제 4: 정규화 패턴
    example_normalization()
    
    print("\n" + "=" * 60)
    print("모든 예제 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
