#!/usr/bin/env python3
"""
CNN 암호화 추론 예제

합성곱 신경망(CNN)에서 GPU 가속 암호화 추론을 수행합니다.
- Conv2d → Square → AvgPool2d → Flatten → Linear 파이프라인
- cnn_config를 사용한 CNN 전용 컨텍스트 설정
- im2col 기반 암호화 합성곱 연산
- 평문 결과와 비교 검증
"""

import torch
import torch.nn as nn
from cukks import CKKSInferenceContext
from cukks.nn import (
    EncryptedConv2d,
    EncryptedSquare,
    EncryptedAvgPool2d,
    EncryptedFlatten,
    EncryptedLinear,
)


class SimpleCNN(nn.Module):
    """간단한 CNN 모델
    
    구조: Conv2d(4→4) → Square → AvgPool2d(2x2) → Flatten → Linear(64→10)
    8x8 입력 이미지 기준:
    - Conv2d 후: 8x8x4 = 256
    - AvgPool2d 후: 4x4x4 = 64
    - FC 입력: 64
    """
    
    def __init__(self, in_channels=4, out_channels=4, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # 8x8 → (pool 2x2) → 4x4, 4채널 = 64
        self.fc = nn.Linear(out_channels * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x ** 2  # CKKS에서 ReLU 대신 제곱 사용
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main():
    torch.manual_seed(42)
    
    # === 1. 모델 및 입력 설정 ===
    image_height = 8
    image_width = 8
    in_channels = 4
    out_channels = 4
    num_classes = 10
    
    model = SimpleCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=num_classes
    )
    model.eval()
    
    print("=== CNN 구조 ===")
    print(f"입력: {in_channels}채널 {image_height}x{image_width}")
    print(f"Conv2d: {in_channels}→{out_channels} (3x3, padding=1)")
    print(f"AvgPool2d: 2x2")
    print(f"FC: {out_channels * 4 * 4}→{num_classes}")
    
    # === 2. 테스트 이미지 생성 ===
    # 배치 없이 (C, H, W) 형식
    image = torch.randn(in_channels, image_height, image_width, dtype=torch.float64)
    
    # === 3. 평문 추론 ===
    with torch.no_grad():
        # PyTorch는 (N, C, H, W) 형식 필요
        plain_output = model(image.unsqueeze(0).float()).squeeze(0).to(torch.float64)
    print(f"\n평문 출력 (처음 5개): {plain_output[:5].tolist()}")
    
    # === 4. CKKS 컨텍스트 생성 (CNN 전용 설정) ===
    # max_rotation_dim: Conv2d 출력 크기 (H * W * out_channels)
    # = 8 * 8 * 4 = 256, 풀링 후 4 * 4 * 4 = 64, FC 입력 차원
    pool_size = 2
    pool_stride = 2
    
    ctx = CKKSInferenceContext(
        max_rotation_dim=576,  # 패딩 포함한 최대 회전 차원
        use_bsgs=True,         # Baby-step Giant-step 최적화
        cnn_config={
            'image_height': image_height,
            'image_width': image_width,
            'channels': out_channels,  # Conv 출력 채널
            'pool_size': pool_size,
            'pool_stride': pool_stride,
        }
    )
    
    print(f"\n=== 암호화 설정 ===")
    print(f"BSGS 사용: {ctx.use_bsgs}")
    print(f"CNN 설정: {image_height}x{image_width}, {out_channels}채널")
    print(f"풀링: {pool_size}x{pool_size}, stride={pool_stride}")
    
    # === 5. 암호화 레이어 변환 ===
    # Conv2d → EncryptedConv2d
    enc_conv = EncryptedConv2d.from_torch(model.conv)
    
    # 제곱 활성화
    enc_square = EncryptedSquare()
    
    # AvgPool2d → EncryptedAvgPool2d
    enc_pool = EncryptedAvgPool2d.from_torch(model.pool)
    
    # Flatten (CNN 레이아웃용)
    enc_flatten = EncryptedFlatten()
    
    # Linear → EncryptedLinear (CNN 레이아웃으로 변환)
    # 풀링 후 패치 수: (4x4) = 16, 각 패치의 특징 수: 4채널
    pooled_height = image_height // pool_stride
    pooled_width = image_width // pool_stride
    num_patches = pooled_height * pooled_width  # 16
    patch_features = out_channels               # 4
    
    enc_fc = EncryptedLinear.from_torch_cnn(
        model.fc,
        cnn_layout={
            'num_patches': num_patches,
            'patch_features': patch_features,
        }
    )
    
    print(f"\n레이어 변환 완료")
    print(f"FC 레이아웃: num_patches={num_patches}, patch_features={patch_features}")
    
    # === 6. CNN 입력 암호화 ===
    # im2col 변환을 위한 합성곱 파라미터
    conv_params = [{
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'out_channels': out_channels,
    }]
    
    enc_input = ctx.encrypt_cnn_input(image, conv_params)
    print(f"\nCNN 입력 암호화 완료 (im2col 적용)")
    
    # === 7. 암호화 추론 실행 ===
    print("암호화 추론 중...")
    
    # Conv2d: 행렬 곱셈으로 합성곱 수행
    enc_h = enc_conv(enc_input)
    
    # Square: 제곱 활성화
    enc_h = enc_square(enc_h)
    
    # AvgPool2d: 회전 기반 평균 풀링
    enc_h = enc_pool(enc_h)
    
    # Flatten: CNN → FC 레이아웃 변환
    enc_h = enc_flatten(enc_h)
    
    # Linear: 최종 분류
    enc_out = enc_fc(enc_h)
    
    # === 8. 결과 복호화 ===
    he_output = ctx.decrypt(enc_out)[:num_classes]
    print(f"암호화 출력 (처음 5개): {he_output[:5].tolist()}")
    
    # === 9. 정확도 검증 ===
    # 코사인 유사도
    cos_sim = torch.nn.functional.cosine_similarity(
        plain_output.unsqueeze(0),
        he_output.unsqueeze(0)
    ).item()
    
    # 평균 절대 오차
    mae = (plain_output - he_output).abs().mean().item()
    
    # 최대 절대 오차
    max_error = (plain_output - he_output).abs().max().item()
    
    print(f"\n=== 정확도 검증 ===")
    print(f"코사인 유사도: {cos_sim:.4f}")
    print(f"평균 절대 오차 (MAE): {mae:.6f}")
    print(f"최대 절대 오차: {max_error:.6f}")
    
    # 결과 판정
    if cos_sim > 0.85:
        print("✅ CNN 암호화 추론 성공!")
    else:
        print("⚠️ 정확도 확인 필요")
    
    # === 10. 예측 클래스 비교 ===
    plain_pred = plain_output.argmax().item()
    he_pred = he_output.argmax().item()
    print(f"\n예측 클래스 - 평문: {plain_pred}, 암호화: {he_pred}")
    if plain_pred == he_pred:
        print("✅ 예측 클래스 일치!")
    else:
        print("⚠️ 예측 클래스 불일치")


if __name__ == "__main__":
    main()
