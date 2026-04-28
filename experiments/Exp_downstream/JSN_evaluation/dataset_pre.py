import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import library as lb
import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pandas as pd
import copy
import random
import json
import torch.nn.functional as F

ROOT_PATH = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir))


class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.root_path = ROOT_PATH + '/Result_Data/proposed/all'
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.transform = transform

    def seg_Gully(self, image, kernel_size=3):
        segmentation = lb.separation(lb.gullyDetect(image * 255))[0]
        segmentation[segmentation != 0] = 1
 
        h, w = segmentation.shape
        for i in range(h):
            if segmentation[i][1] == 0:
                break
        for j in range(i):
            segmentation[j][0] = 1

        segmentation = lb.soomth(segmentation, kernel_size=kernel_size)

        return segmentation

    def reconstruction_all(self, upper_layer, lower_layer, soft_tissue):
        upper_trans = 1 - upper_layer
        lower_trans = 1 - lower_layer
        soft_tissue_trans = 1 - soft_tissue

        output = (1 - (upper_trans * lower_trans * soft_tissue_trans))

        return output

    def random_move(self, parameter, input):
        input = input.unsqueeze(0)
        device = input.device
        # Scale
        value_S = torch.tensor((parameter[0], ))
        value_S = value_S.unsqueeze(0)
        ones_S = torch.ones(1, 1, device=device)
        zeros_S = torch.zeros(1, 1, device=device)
        theta_S = torch.cat((value_S, zeros_S, zeros_S,
                             zeros_S, value_S, zeros_S,
                             zeros_S, zeros_S, ones_S))
        theta_S = theta_S.view(-1, 3, 3)

        # Rotation
        value_R = torch.tensor((parameter[1], ))
        value_R = value_R.unsqueeze(0)
        ones_R = torch.ones(1, 1, device=device)
        zeros_R = torch.zeros(1, 1, device=device)
        theta_R = torch.cat((torch.cos(value_R), -torch.sin(value_R), zeros_R,
                             torch.sin(value_R), torch.cos(value_R), zeros_R,
                             zeros_R, zeros_R, ones_R))
        theta_R = theta_R.view(-1, 3, 3)

        # Translation
        value_Tx = torch.tensor((parameter[2], ))
        value_Ty = torch.tensor((parameter[3], ))
        value_Tx = value_Tx.unsqueeze(0)
        value_Ty = value_Ty.unsqueeze(0)
        ones_T = torch.ones(1, 1, device=device)
        zeros_T = torch.zeros(1, 1, device=device)
        theta_T = torch.cat((ones_T, zeros_T, value_Tx,
                             zeros_T, ones_T, value_Ty,
                             zeros_T, zeros_T, ones_T), 1)

        theta_T = theta_T.view(-1, 3, 3)

        theta = torch.matmul(torch.matmul(theta_S, theta_R), theta_T)

        theta = theta[:, :2, :]

        grid = F.affine_grid(theta, input.shape, align_corners=False)
        output = F.grid_sample(input, grid, align_corners=False)

        parameter = torch.cat((value_S, value_R, value_Tx, value_Ty), 1)

        output = output[0, :, :, :]

        return output, parameter

    def input_image(self, index=0, target=''):
        soft_tissue_path = self.root_path + '/soft_tissue/' + self.data[index]['fixed']
        upper_bone_path = soft_tissue_path.replace('soft_tissue', 'upper_layer')
        lower_bone_path = soft_tissue_path.replace('soft_tissue', 'lower_layer')

        parameter_upper = self.data[index]['moving_parameter_upper']
        parameter_lower = self.data[index]['moving_parameter_lower']

        soft_tissue = np.array(Image.open(soft_tissue_path).convert("L"))
        lower_bone = np.array(Image.open(lower_bone_path).convert("L"))
        upper_bone = np.array(Image.open(upper_bone_path).convert("L"))

        soft_tissue = self.transform(Image.fromarray(soft_tissue))
        lower_bone = self.transform(Image.fromarray(lower_bone))
        upper_bone = self.transform(Image.fromarray(upper_bone))

        if target == 'moved':
            lower_bone_moved, parameter_lower = self.random_move(parameter_lower, lower_bone)
            upper_bone_moved, parameter_upper = self.random_move(parameter_upper, upper_bone)

            recon_image = self.reconstruction_all(upper_bone_moved, lower_bone_moved, soft_tissue)

            segmentation = self.seg_Gully(np.array(recon_image[0]))
            segmentation[segmentation != 0] = 1.0
            segmentation = torch.tensor(segmentation).unsqueeze(0)

            parameter = torch.cat((parameter_upper, parameter_lower))

            return recon_image, segmentation, parameter
        else:
            recon_image = self.reconstruction_all(upper_bone, lower_bone, soft_tissue)
            segmentation = self.seg_Gully(np.array(recon_image[0]))
            segmentation[segmentation != 0] = 1.0
            segmentation = torch.tensor(segmentation).unsqueeze(0)

            return recon_image, segmentation

    def __getitem__(self, index):
        # Data path
        fixed_image, fixed_mask, parameter = self.input_image(index=index, target='moved')
        moving_image, moving_mask = self.input_image(index=index, target='fixed')

        return moving_image, fixed_image, moving_mask, fixed_mask, parameter

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = '/Users/wanghaolin/PycharmProjects/LSN/experiments/Exp_downstream/JSN_evaluation/pre_dataset.json'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=2,
                                       shuffle=True,
                                       drop_last=True
                                       )
    print(len(data))
    for moving_image, fixed_image, moving_masks, fixed_masks, parameter in data:
        print(parameter)
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 5, 1)
        plt.imshow(moving_image.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 2)
        plt.imshow(fixed_image.numpy()[0][0], 'gray')
        # plt.imshow(moving_image_cropping.numpy()[0][1], alpha=0.5)
        plt.subplot(1, 5, 3)
        plt.imshow(moving_masks.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 4)
        plt.imshow(fixed_masks.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 5)
        plt.tight_layout()
        plt.show()
    print(len(data))
