import torch
import torch.nn as nn
from torch.autograd import gradcheck
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def layer_print(x, channel):
    print('Tensor Shape:', x.shape)
    image = x.cpu().detach().numpy()[0][channel]
    plt.imshow(image)
    plt.show()

def show_layers(layers, title=''):
    if isinstance(layers, torch.Tensor):
        layers = layers.cpu().detach().numpy()[0]

    c, _, _ = layers.shape

    plt.figure(figsize=(c, 1.5))
    for i in range(c):
        plt.subplot(1, c, (i+1))
        plt.imshow(layers[i], 'gray', vmax=1, vmin=0)
        plt.axis('off')
        plt.title('{} L{}'.format(title, c))
    plt.show()


class ResidualBlock_input(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_input, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


def registration(output, x1, x2):
    batch_size = output.shape[0]

    # Scale
    value_S = F.tanh(output[:, 0]) + 1
    value_S = torch.unsqueeze(value_S, 1)
    ones_S = torch.ones(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    zeros_S = torch.zeros(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    theta_S = torch.cat((value_S, zeros_S, zeros_S,
                         zeros_S, value_S, zeros_S,
                         zeros_S, zeros_S, ones_S), 1)
    theta_S = theta_S.view(-1, 3, 3)

    # Rotation
    value_R = F.tanh(output[:, 1]) * np.pi / 2
    value_R = torch.unsqueeze(value_R, 1)
    ones_R = torch.ones(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    zeros_R = torch.zeros(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    theta_R = torch.cat((torch.cos(value_R), -torch.sin(value_R), zeros_R,
                         torch.sin(value_R), torch.cos(value_R), zeros_R,
                         zeros_R, zeros_R, ones_R), 1)
    theta_R = theta_R.view(-1, 3, 3)

    # Translation
    value_Tx = output[:, 2]
    value_Ty = output[:, 3]
    value_Tx = torch.unsqueeze(value_Tx, 1)
    value_Ty = torch.unsqueeze(value_Ty, 1)
    ones_T = torch.ones(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    zeros_T = torch.zeros(batch_size, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    theta_T = torch.cat((ones_T, zeros_T, value_Tx,
                         zeros_T, ones_T, value_Ty,
                         zeros_T, zeros_T, ones_T), 1)


    theta_T = theta_T.view(-1, 3, 3)

    theta = torch.matmul(torch.matmul(theta_S, theta_R), theta_T)

    theta = theta[:, :2, :]

    grid = F.affine_grid(theta, x2.shape, align_corners=False)
    x1_reg = F.grid_sample(x1, grid, align_corners=False)

    range_crop = 5
    x1_reg = x1_reg[:, :, range_crop: 256 - range_crop, range_crop: 256 - range_crop]
    x2 = x2[:, :, range_crop: 256 - range_crop, range_crop: 256 - range_crop]

    length = (256 - range_crop) / 2

    output = torch.cat((value_S, value_R, value_Tx, value_Ty), 1)

    return x1_reg, x2, output


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Registrater(torch.nn.Module):
    def __init__(self, in_channels=4):
        super(Registrater, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(2, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, 8)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构造layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x1_org, x2_org, x1_mask, x2_mask):
        '''
        :param x1_org: moving layer image in 2 channels
        :param x2_org: fixed layer image in 2 channels
        :return: warpped moving image, cropped fixed image, parameter
        '''
        x1 = x1_org.clone()
        x2 = x2_org.clone()

        x = torch.cat((x1, x2), dim=1)

        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        output = output.view(-1, 2, 4)

        output_upper = output[:, 0]
        output_lower = output[:, 1]

        # upper
        x1_org_upper_mask = x1_mask
        x2_org_upper_mask = x2_mask
        x1_org_upper = x1_org * x1_org_upper_mask
        x2_org_upper = x2_org * x2_org_upper_mask

        x1_upper, x2_upper, parameter_upper = registration(output_upper, x1_org, x2_org)
        x1_upper_mask, x2_upper_mask, parameter_upper = registration(output_upper, x1_org_upper_mask, x2_org_upper_mask)

        # union = (x1_upper_mask + x2_upper_mask) / 2 
        # union[union != 1] = 0

        # x1_upper = x1_upper * union
        # x2_upper = x2_upper * union

        x1_upper = x1_upper * x2_upper_mask
        x2_upper = x2_upper * x2_upper_mask

        # lower
        x1_org_lower_mask = 1 - x1_mask
        x2_org_lower_mask = 1 - x2_mask
        x1_org_lower = x1_org * x1_org_lower_mask
        x2_org_lower = x2_org * x2_org_lower_mask

        x1_lower, x2_lower, parameter_lower = registration(output_lower, x1_org, x2_org)
        x1_lower_mask, x2_lower_mask, parameter_lower = registration(output_lower, x1_org_lower_mask, x2_org_lower_mask)

        # union = (x1_lower_mask + x2_lower_mask) / 2
        # union[union != 1] = 0

        # x1_lower = x1_lower * union
        # x2_lower = x2_lower * union

        x1_lower = x1_lower * x2_lower_mask
        x2_lower = x2_lower * x2_lower_mask

        x1_reg = torch.cat((x1_upper, x1_lower), dim=1)
        x2_crop = torch.cat((x2_upper, x2_lower), dim=1)

        x1_reg_mask = torch.cat((x1_upper_mask, x1_lower_mask), dim=1)
        x2_crop_mask = torch.cat((x2_upper_mask, x2_lower_mask), dim=1)

        parameter_upper = parameter_upper.unsqueeze(1)
        parameter_lower = parameter_lower.unsqueeze(1)

        parmeter = torch.cat((parameter_upper, parameter_lower), 1)

        return x1_reg, x2_crop, x1_reg_mask, x2_crop_mask, parmeter
