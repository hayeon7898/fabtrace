"""웨이퍼 맵 패턴 분류용 소형 CNN.

입력 (B, 3, 64, 64) 원-핫 맵 -> 출력 (B, 9) 클래스 로짓.
합성곱 블록 4개(3x3 합성곱, 배치 정규화, ReLU, 2x2 풀링)로 64 -> 4까지 줄이고,
전역 평균 풀링과 드롭아웃 뒤 선형층으로 분류한다. 파라미터 약 10만 개.
"""

import torch
from torch import nn

from fabtrace.data.dataset import N_CLASSES

CHANNELS = (16, 32, 64, 128)


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        layers, prev = [], in_channels
        for ch in CHANNELS:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)