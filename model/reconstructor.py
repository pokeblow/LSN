'''
version: 1.0
data: 2023.3.8
model: FCN
'''

import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.if_CP = False

        # 编码器部分
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        # 解码器部分
        self.deconv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.liner = nn.Sequential(
            nn.Linear(256 * 256 * 1, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def reconstruct(self, org, k, masks):
        # image
        upper_image = org[:, 1, :, :].unsqueeze(1)
        lower_image = org[:, 0, :, :].unsqueeze(1)

        upper_trans = 1 - upper_image
        lower_trans = 1 - lower_image

        org_mask_and = torch.sum(masks, 1).unsqueeze(1)
        org_mask_and[org_mask_and != 2] = 0
        and_trans = (org_mask_and / 2) * k

        and_trans = 1 - and_trans

        if self.if_CP:
            output = (1 - (upper_trans * lower_trans / and_trans))
        else:
            output = (1 - (upper_trans * lower_trans))

        return output

    def set_correct_parameter(self, if_CP=False):
        self.if_CP = if_CP

    def forward(self, image, org, masks):
        # 编码器
        x = torch.clone(image)

        x = self.conv1(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 解码器
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        x = self.deconv1(x)

        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        x = self.deconv2(x)

        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        x = self.deconv3(x)

        x = x.view(x.size(0), -1)
        x = self.liner(x) / 2

        _, _, h, w = image.shape
        batch_size, _ = x.shape

        k = x.view(batch_size, 1, 1, 1)
        k = k.expand(batch_size, 1, h, w)

        recon_image = self.reconstruct(org, k, masks)

        return recon_image, x
