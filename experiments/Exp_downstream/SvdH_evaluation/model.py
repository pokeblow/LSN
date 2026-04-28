import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class SvdHClassification(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, pretrained=False):
        super(SvdHClassification, self).__init__()

        # 加载 ResNet 模型
        self.resnet = resnet50(pretrained=pretrained)

        # 修改第一层以适应自定义的输入通道数
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.num_features = self.resnet.fc.in_features

        # 替换最后的全连接层，使其输出5类
        self.resnet.fc = nn.Linear(self.num_features, out_channels)

    def forward(self, x):
        output = self.resnet(x)  # 提取特征  # 回归头输出
        return output