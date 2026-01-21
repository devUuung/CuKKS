#!/usr/bin/env python3
"""
GPU 부트스트랩 예제

CKKS 동형암호에서 GPU 가속 부트스트랩을 사용합니다.
- 부트스트랩: 암호문의 노이즈를 제거하고 연산 가능한 깊이를 복원
- GPU 가속으로 부트스트랩 성능 향상
- 깊은 연산 (deep computation)을 위한 필수 기능

Warning:
    OpenFHE GPU 백엔드의 부트스트랩은 일부 환경에서 불안정할 수 있습니다.
    문제 발생 시 crypto_inv_sqrt_shallow 등 부트스트랩 없는 대안을 고려하세요.
"""

import torch
from ckks.torch_api import CKKSContext, CKKSConfig


def example_basic_bootstrap():
    """기본 부트스트랩 예제: 노이즈 복원"""
    print("=" * 60)
    print("1. 기본 부트스트랩")
    print("=" * 60)
    
    # 1. CKKS 설정 (부트스트랩 활성화)
    # 부트스트랩은 큰 링 차원과 깊은 모듈러스 체인이 필요
    config = CKKSConfig(
        poly_mod_degree=32768,  # 부트스트랩 권장 크기
        coeff_mod_bits=[60] + [40] * 14 + [60],  # 충분한 깊이
        scale_bits=40,
        enable_bootstrap=True,  # 부트스트랩 활성화
        security_level=None,
    )
    
    print(f"링 차원: {config.poly_mod_degree}")
    print(f"슬롯 수: {config.poly_mod_degree // 2}")
    print(f"모듈러스 레벨: {len(config.coeff_mod_bits)}")
    print(f"부트스트랩: 활성화")
    
    # 2. 컨텍스트 생성 (GPU 활성화)
    ctx = CKKSContext(config, enable_gpu=True)
    print(f"\nGPU 활성화: {ctx.gpu_enabled}")
    
    # 3. 테스트 데이터
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    print(f"\n입력 데이터: {x.tolist()}")
    
    # 4. 암호화 및 레벨 확인
    enc_x = ctx.encrypt(x)
    initial_level = enc_x.level()
    print(f"초기 레벨: {initial_level}")
    
    # 5. 연산으로 레벨 소모
    print("\n연산 수행 중...")
    result = enc_x
    for i in range(5):
        result = result.mul(result)  # 제곱 연산 (레벨 소모)
        result = result.rescale()    # 리스케일 (레벨 감소)
        print(f"  반복 {i+1}: 레벨 = {result.level()}")
    
    level_before_bootstrap = result.level()
    print(f"\n부트스트랩 전 레벨: {level_before_bootstrap}")
    
    # 6. 부트스트랩 수행
    print("부트스트랩 수행 중... (GPU 가속)")
    result = result.bootstrap()
    level_after_bootstrap = result.level()
    print(f"부트스트랩 후 레벨: {level_after_bootstrap}")
    
    # 7. 레벨 복원 확인
    level_restored = level_after_bootstrap - level_before_bootstrap
    print(f"\n복원된 레벨 수: {level_restored}")
    
    # 8. 정확도 확인
    # 부트스트랩 후에도 값이 유지되는지 확인
    decrypted = ctx.decrypt(result)[:5]
    # x^32 (5번 제곱)
    expected = x ** 32
    
    print(f"\n=== 정확도 검증 ===")
    print(f"기대값 (x^32): {expected.tolist()}")
    print(f"복호화 결과: {decrypted.tolist()}")
    
    # 상대 오차
    rel_error = ((decrypted - expected).abs() / expected.abs()).mean().item()
    print(f"평균 상대 오차: {rel_error:.2e}")
    
    if rel_error < 0.01:  # 1% 이하
        print("\n✅ 부트스트랩 테스트 통과!")
    else:
        print("\n⚠️ 정확도 확인 필요")


def example_deep_computation():
    """깊은 연산 예제: 부트스트랩으로 연속 연산"""
    print("\n" + "=" * 60)
    print("2. 깊은 연산 (Deep Computation)")
    print("=" * 60)
    
    # 1. CKKS 설정
    config = CKKSConfig(
        poly_mod_degree=32768,
        coeff_mod_bits=[60] + [40] * 14 + [60],
        scale_bits=40,
        enable_bootstrap=True,
        security_level=None,
    )
    
    ctx = CKKSContext(config, enable_gpu=True)
    
    # 2. 테스트 데이터
    x = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64)
    print(f"입력 데이터: {x.tolist()}")
    
    # 3. 암호화
    enc_x = ctx.encrypt(x)
    
    # 4. 부트스트랩 없이 가능한 최대 곱셈 깊이 확인
    max_depth_without_bootstrap = len(config.coeff_mod_bits) - 2  # 시작/끝 모듈러스 제외
    print(f"\n부트스트랩 없이 최대 곱셈 횟수: {max_depth_without_bootstrap}")
    
    # 5. 부트스트랩을 사용한 깊은 연산
    # 예: 다항식 f(x) = x^10을 반복 평가
    total_multiplications = 0
    result = enc_x
    bootstrap_count = 0
    
    print("\n연산 시작...")
    for round in range(3):  # 3 라운드
        print(f"\n=== 라운드 {round + 1} ===")
        
        # 각 라운드에서 x^3 = x * x * x 계산
        for mult in range(3):
            result = result.mul(result)  # x^2, x^4, x^8
            result = result.rescale()
            total_multiplications += 1
            print(f"  곱셈 #{total_multiplications}: 레벨 = {result.level()}")
            
            # 레벨이 낮으면 부트스트랩
            if result.level() < 3:
                print(f"  → 부트스트랩 (레벨 부족)")
                result = result.bootstrap()
                bootstrap_count += 1
                print(f"  → 복원 후 레벨: {result.level()}")
    
    print(f"\n=== 연산 통계 ===")
    print(f"총 곱셈 횟수: {total_multiplications}")
    print(f"부트스트랩 횟수: {bootstrap_count}")
    
    # 6. 결과 확인
    decrypted = ctx.decrypt(result)[:5]
    # x^512 = x^(2^9) (9번 곱셈, 하지만 rescale로 정밀도 손실 있음)
    expected = x ** (2 ** 9)
    
    print(f"\n복호화 결과 (처음 5개): {decrypted.tolist()}")
    print("※ 깊은 연산에서는 정밀도 손실이 누적됩니다.")
    
    print("\n✅ 깊은 연산 데모 완료!")


def example_bootstrap_timing():
    """부트스트랩 시간 측정 예제"""
    print("\n" + "=" * 60)
    print("3. 부트스트랩 성능 측정")
    print("=" * 60)
    
    import time
    
    # 1. CKKS 설정
    config = CKKSConfig(
        poly_mod_degree=32768,
        coeff_mod_bits=[60] + [40] * 14 + [60],
        scale_bits=40,
        enable_bootstrap=True,
        security_level=None,
    )
    
    ctx = CKKSContext(config, enable_gpu=True)
    
    # 2. 테스트 데이터 (풀 슬롯)
    num_slots = config.poly_mod_degree // 2
    x = torch.randn(num_slots, dtype=torch.float64)
    print(f"슬롯 수: {num_slots:,}")
    
    # 3. 암호화
    enc_x = ctx.encrypt(x)
    
    # 4. 레벨 소모 (부트스트랩이 의미있도록)
    result = enc_x
    for _ in range(5):
        result = result.mul(result)
        result = result.rescale()
    
    print(f"부트스트랩 전 레벨: {result.level()}")
    
    # 5. 부트스트랩 시간 측정
    # Warmup
    _ = result.bootstrap()
    result = enc_x
    for _ in range(5):
        result = result.mul(result)
        result = result.rescale()
    
    # 실제 측정
    times = []
    for i in range(3):
        # 레벨 소모
        temp = enc_x
        for _ in range(5):
            temp = temp.mul(temp)
            temp = temp.rescale()
        
        start = time.perf_counter()
        temp = temp.bootstrap()
        end = time.perf_counter()
        
        elapsed = (end - start) * 1000  # ms
        times.append(elapsed)
        print(f"  실행 {i+1}: {elapsed:.1f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"\n=== 성능 결과 ===")
    print(f"평균 부트스트랩 시간: {avg_time:.1f} ms")
    print(f"GPU: {'활성화' if ctx.gpu_enabled else '비활성화'}")
    
    # 6. 슬롯당 처리량
    throughput = num_slots / (avg_time / 1000)  # slots per second
    print(f"처리량: {throughput:,.0f} 슬롯/초")
    
    print("\n✅ 성능 측정 완료!")


def main():
    """모든 예제 실행"""
    print("\n" + "=" * 60)
    print("GPU 부트스트랩 예제")
    print("=" * 60)
    print("""
부트스트랩은 CKKS 동형암호에서 가장 중요한 연산입니다:

1. 노이즈 복원: 암호문의 누적된 노이즈를 제거
2. 레벨 복원: 연산 가능한 깊이(레벨)를 다시 높임
3. 무한 연산: 부트스트랩으로 이론적으로 무한한 연산 가능

GPU 가속:
- 부트스트랩은 계산 집약적 (수백 ms ~ 수 초)
- GPU는 병렬 연산으로 10-100배 가속 가능
- OpenFHE GPU 백엔드 사용

주의사항:
- 부트스트랩 자체도 약간의 오차 발생
- poly_mod_degree >= 32768 권장
- 메모리 사용량이 큼 (수 GB)
""")
    
    # 예제 실행 여부 확인
    print("\n⚠️ 부트스트랩은 메모리/시간 집약적입니다.")
    print("   환경에 따라 일부 예제가 실패할 수 있습니다.\n")
    
    try:
        # 예제 1: 기본 부트스트랩
        example_basic_bootstrap()
        
        # 예제 2: 깊은 연산
        # example_deep_computation()
        
        # 예제 3: 성능 측정
        # example_bootstrap_timing()
        
    except RuntimeError as e:
        print(f"\n⚠️ 부트스트랩 오류 발생:")
        print(f"   {e}")
        print("\n   OpenFHE GPU 백엔드의 알려진 제한사항일 수 있습니다.")
        print("   → crypto_inv_sqrt_shallow 등 대안 사용을 권장합니다.")
    
    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
