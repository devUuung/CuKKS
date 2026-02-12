#!/usr/bin/env python3
"""
MLP 암호화 추론 예제

완전연결 신경망(MLP)에서 암호화된 추론을 수행합니다.
- PyTorch 모델을 암호화 레이어로 변환
- BSGS 최적화로 빠른 행렬 곱셈
- 평문 결과와 비교 검증
"""

import torch
import torch.nn as nn
from cukks import CKKSInferenceContext
from cukks.nn import EncryptedLinear, EncryptedSquare


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x ** 2  # CKKS에서 ReLU 대신 제곱 사용
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(42)
    
    # 1. 모델 생성 및 평가 모드 설정
    model = SimpleMLP(input_dim=16, hidden_dim=32, output_dim=10)
    model.eval()
    print("=== MLP 구조 ===")
    print(f"입력: 16 → 은닉층: 32 → 출력: 10")
    
    # 2. 테스트 입력 생성
    x = torch.randn(16, dtype=torch.float64)
    
    # 3. 평문 추론
    with torch.no_grad():
        plain_output = model(x.float()).to(torch.float64)
    print(f"\n평문 출력 (처음 5개): {plain_output[:5]}")
    
    # 4. CKKS 컨텍스트 생성
    ctx = CKKSInferenceContext(
        max_rotation_dim=64,  # 최대 회전 차원
        use_bsgs=True,        # Baby-step Giant-step 최적화
    )
    print(f"\n=== 암호화 설정 ===")
    print(f"BSGS 사용: {ctx.use_bsgs}")
    
    # 5. PyTorch 레이어를 암호화 레이어로 변환
    enc_fc1 = EncryptedLinear.from_torch(model.fc1)
    enc_square = EncryptedSquare()
    enc_fc2 = EncryptedLinear.from_torch(model.fc2)
    
    # 6. 입력 암호화
    enc_x = ctx.encrypt(x)
    print(f"\n입력 암호화 완료")
    
    # 7. 암호화 추론
    print("암호화 추론 중...")
    enc_h = enc_fc1(enc_x)      # 첫 번째 FC
    enc_h = enc_square(enc_h)   # 제곱 활성화
    enc_out = enc_fc2(enc_h)    # 두 번째 FC
    
    # 8. 결과 복호화
    he_output = ctx.decrypt(enc_out)[:10]
    print(f"암호화 출력 (처음 5개): {he_output[:5]}")
    
    # 9. 정확도 검증
    cos_sim = torch.nn.functional.cosine_similarity(
        plain_output.unsqueeze(0),
        he_output.unsqueeze(0)
    ).item()
    mae = (plain_output - he_output).abs().mean().item()
    
    print(f"\n=== 정확도 ===")
    print(f"코사인 유사도: {cos_sim:.4f}")
    print(f"평균 절대 오차: {mae:.6f}")
    
    if cos_sim > 0.99:
        print("✅ MLP 암호화 추론 성공!")
    else:
        print("⚠️ 정확도 확인 필요")


if __name__ == "__main__":
    main()
