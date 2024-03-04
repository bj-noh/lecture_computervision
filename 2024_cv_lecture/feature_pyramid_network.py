import torch
import torch.nn as nn
import torchvision

class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        # 백본 네트워크 초기화 (예: ResNet)
        self.backbone = backbone
        # FPN 레이어 초기화
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(2048, 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        # 백본 네트워크를 통과
        features = self.backbone(x)
        
        # 백본 네트워크로부터 추출된 특징 맵
        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        f4 = features[3]
        
        # FPN 상향 샘플링 및 통합
        p4 = self.conv4(f4)
        p3 = self.upsample(p4) + self.conv3(f3)
        p2 = self.upsample(p3) + self.conv2(f2)
        p1 = self.upsample(p2) + self.conv1(f1)
        
        # 수정된 특징 맵 반환
        return [p1, p2, p3, p4]

# 예시: ResNet50을 백본 네트워크로 사용하는 FPN
backbone = torchvision.models.resnet50(pretrained=True).features
fpn = FPN(backbone)
