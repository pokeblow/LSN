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
    def __init__(self, data_path, transform=None, create_overlap_flag=False):
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.root_path = '/workspace/Zero_JSW_Generation/Data/Dataset_cases'
        self.transform = transform
        self.create_overlap_flag = create_overlap_flag

    def seg_Gully(self, image, kernel_size=3):
        segmentation = lb.separation(lb.gullyDetect(image))[0]
        segmentation[segmentation != 0] = 1
        h, w = segmentation.shape
        for i in range(h):
            if segmentation[i][1] == 0:
                break
        for j in range(i):
            segmentation[j][0] = 1

        segmentation = lb.soomth(segmentation, kernel_size=kernel_size)

        return segmentation

    def input_image(self, target='', index=0):
        image_path = self.root_path + '/' + self.data[index][target]

        image = np.array(Image.open(image_path).convert("L"))

        segmentation = self.seg_Gully(image)

        segmentation[segmentation != 0] = 1.0

        return image, segmentation

    def __getitem__(self, index):
        # Data path
        # moving
        moving_image, moving_mask = self.input_image(target='moving', index=index)
        fixed_image, fixed_mask = self.input_image(target='fixed', index=index)

        # Transform
        def to_tensor(image, mask):
            image = self.transform(Image.fromarray(image))
            # mask = torch.tensor(mask).unsqueeze(0)
            mask = self.transform(Image.fromarray(mask))

            return image, mask

        moving_image, moving_masks = to_tensor(moving_image, moving_mask)

        fixed_image, fixed_masks = to_tensor(fixed_image, fixed_mask)

        return moving_image, fixed_image, moving_masks, fixed_masks

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = ROOT_PATH + '/Data/DR_K1_train'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform, create_overlap_flag=False)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=2,
                                       shuffle=True,
                                       drop_last=True
                                       )
    for moving_image, fixed_image, moving_masks, fixed_masks in data:
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 5, 1)
        plt.imshow(moving_image.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 2)
        plt.imshow(fixed_image.numpy()[0][0])
        # plt.imshow(moving_image_cropping.numpy()[0][1], alpha=0.5)
        plt.subplot(1, 5, 3)
        plt.imshow(moving_masks.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 4)
        plt.imshow(fixed_masks.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 5)
        plt.tight_layout()
        plt.show()
    print(len(data))
