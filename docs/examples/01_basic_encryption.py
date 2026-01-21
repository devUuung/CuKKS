#!/usr/bin/env python3
"""
기본 암호화/복호화 예제

CKKS 동형암호의 기본 사용법을 보여줍니다.
- 컨텍스트 생성
- 벡터 암호화
- 암호문 연산 (덧셈, 곱셈)
- 복호화 및 결과 확인
"""

import torch
from ckks.torch_api import CKKSContext, CKKSConfig


def main():
    # 1. CKKS 설정 생성
    config = CKKSConfig(
        poly_mod_degree=8192,      # 링 차원 (슬롯 수 = 4096)
        coeff_mod_bits=[60, 40, 40, 60],  # 계수 모듈러스 비트
        scale_bits=40,             # 인코딩 정밀도
        rotations=list(range(-16, 17)),  # 회전 키 범위
        security_level=None,       # 보안 수준 (None = 체크 비활성화)
    )
    
    # 2. 컨텍스트 생성 (GPU 자동 활성화)
    ctx = CKKSContext(config, enable_gpu=True)
    print(f"GPU 활성화: {ctx.gpu_enabled}")
    print(f"슬롯 수: {config.poly_mod_degree // 2}")
    
    # 3. 데이터 준비 및 암호화
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    
    enc_x = ctx.encrypt(x)
    enc_y = ctx.encrypt(y)
    print(f"\n암호화 완료")
    
    # 4. 암호문 덧셈
    enc_add = enc_x + enc_y
    dec_add = ctx.decrypt(enc_add)[:5]
    print(f"\n=== 덧셈 ===")
    print(f"x + y (평문): {x + y}")
    print(f"x + y (암호문): {dec_add}")
    print(f"오차: {(dec_add - (x + y)).abs().max():.2e}")
    
    # 5. 암호문 곱셈
    enc_mul = enc_x * enc_y
    dec_mul = ctx.decrypt(enc_mul)[:5]
    print(f"\n=== 곱셈 ===")
    print(f"x * y (평문): {x * y}")
    print(f"x * y (암호문): {dec_mul}")
    print(f"오차: {(dec_mul - (x * y)).abs().max():.2e}")
    
    # 6. 평문 상수 연산
    scalar = 2.5
    enc_scale = enc_x * scalar
    dec_scale = ctx.decrypt(enc_scale)[:5]
    print(f"\n=== 상수 곱셈 ===")
    print(f"x * {scalar} (평문): {x * scalar}")
    print(f"x * {scalar} (암호문): {dec_scale}")
    print(f"오차: {(dec_scale - (x * scalar)).abs().max():.2e}")
    
    # 7. 제곱 연산
    enc_sq = enc_x.square()
    dec_sq = ctx.decrypt(enc_sq)[:5]
    print(f"\n=== 제곱 ===")
    print(f"x² (평문): {x ** 2}")
    print(f"x² (암호문): {dec_sq}")
    print(f"오차: {(dec_sq - (x ** 2)).abs().max():.2e}")


if __name__ == "__main__":
    main()
