import torch.nn.functional as F
import torch
import torch.nn as nn
from model.backbones.resnet50 import ResNet
from model.backbones.denseunet import DenseUnet
from model.backbones.unet import Unet
from model.backbones.nestedunet import NestedUnet
from model.backbones.transunet import TransUnet
import timm
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class Parameter(nn.Module):
    def __init__(self, in_channels, out_channels, backbone=None):
        super(Parameter, self).__init__()
        self.backbone = backbone
        self.input_channel = in_channels
        self.num_classes = out_channels

        if backbone == 'resnet':
            self.backbone_network = models.resnet18(pretrained=False)
            self.backbone_network.conv1 = nn.Conv2d(
                    in_channels=self.input_channel,  # Change input channels from 3 to 1
                    out_channels=self.backbone_network.conv1.out_channels,
                    kernel_size=self.backbone_network.conv1.kernel_size,
                    stride=self.backbone_network.conv1.stride,
                    padding=self.backbone_network.conv1.padding,
                    bias=self.backbone_network.conv1.bias
                )

            # Modify the final fully connected layer
            self.backbone_network.fc = nn.Linear(
                in_features=self.backbone_network.fc.in_features,
                out_features=self.num_classes  # Set to the number of classes you have
            )

    def forward(self, x):
        x = self.backbone_network(x)
        x = F.sigmoid(x) / 3

        return x


# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# from model.backbones.resnet50 import ResNet
# from model.backbones.denseunet import DenseUnet
# from model.backbones.unet import Unet
# from model.backbones.nestedunet import NestedUnet
# from model.backbones.transunet import TransUnet
# import timm
# import torchvision.models as models
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# class Parameter(nn.Module):
#     def __init__(self, in_channels, out_channels, backbone=None):
#         super(Parameter, self).__init__()
#         self.backbone = backbone
#         self.input_channel = in_channels
#         self.output_channel = out_channels

#         if backbone == 'test':
#             self.backbone_network = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=1, padding=1)

#         if backbone == 'unet':
#             self.backbone_network = Unet(in_channels=self.input_channel, out_channels=self.output_channel)
#         if backbone == 'nestedunet':
#             self.backbone_network = NestedUnet(num_channels=self.input_channel, num_class=self.output_channel)
#         if backbone == 'deeplab':
#             self.backbone_network = models.segmentation.deeplabv3_resnet50(pretrained=False)
#             self.backbone_network.backbone.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             self.backbone_network.classifier[4] = nn.Conv2d(256, self.output_channel, kernel_size=(1, 1), stride=(1, 1))
#         if backbone == 'densenet':
#             self.backbone_network = DenseUnet(img_ch=self.input_channel, output_ch=self.output_channel)
#         if backbone == 'resnet':
#             self.backbone_network = ResNet(in_channels=self.input_channel, out_channels=self.output_channel)
#         if backbone == 'transunet':
#             self.backbone_network = TransUnet(in_channels=self.input_channel, class_num=self.output_channel, 
#                                               img_dim=128, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16)


#     def forward(self, x):
#         x = self.backbone_network(x)
#         if self.backbone == 'deeplab':
#             x = x['out']
#         # output_seg = x[:, :3, :, :]
#         # output_aut = x[:, 3:, :, :]

#         # # output_seg = self.softmax(output_seg)
#         # output_seg = output_seg
#         x = F.sigmoid(x)
#         # output = torch.cat((output_seg, output_aut), dim=1)

#         return x
