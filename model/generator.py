import torch
import torch.nn as nn
from model.backbones.resnet50 import ResNet
from model.backbones.denseunet import DenseUnet
from model.backbones.unet import Unet
from model.backbones.nestedunet import NestedUnet
from model.backbones.esrt import ESRT
from model.backbones.transunet import TransUnet
import timm
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_layers, backbone=None):
        super(Generator, self).__init__()
        self.backbone = backbone
        self.input_channel = n_layers + 1
        self.output_channel = n_layers

        if backbone == 'test':
            self.backbone_network = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=1, padding=1)


        if backbone == 'unet':
            self.backbone_network = Unet(in_channels=self.input_channel, out_channels=self.output_channel)
        if backbone == 'nestedunet':
            self.backbone_network = NestedUnet(num_channels=self.input_channel, num_class=self.output_channel)
        if backbone == 'deeplab':
            self.backbone_network = models.segmentation.deeplabv3_resnet50(pretrained=False)
            self.backbone_network.backbone.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone_network.classifier[4] = nn.Conv2d(256, self.output_channel, kernel_size=(1, 1), stride=(1, 1))
        if backbone == 'denseunet':
            self.backbone_network = DenseUnet(in_ch=self.input_channel, out_ch=self.output_channel)
        if backbone == 'resnet':
            self.backbone_network = ResNet(in_channels=self.input_channel, out_channels=self.output_channel)
        if backbone == 'esrt':
             self.backbone_network = ESRT(in_channels=self.input_channel, out_channels=self.output_channel, mlpDim=128, scaleFactor=4)
        if backbone == 'transunet':
            self.backbone_network = TransUnet(in_channels=self.input_channel, class_num=self.output_channel, 
                                              img_dim=256, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16)


    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x = self.backbone_network(x)
        if self.backbone == 'deeplab':
            x = x['out']
        output = F.sigmoid(x) * mask
        # output = F.sigmoid(x)

        return output
