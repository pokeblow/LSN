import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
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

ROOT_PATH = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), os.pardir))


class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.transform = transform

    def translated_image(self, image, parameter):
        image = Image.fromarray(image)
        x_shift = parameter[0]
        y_shift = parameter[1]

        # 随机生成旋转角度
        rotation_angle = parameter[2]

        # 平移图像
        translated_image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -x_shift, 0, 1, y_shift)
        )

        # 旋转图像
        rotated_image = translated_image.rotate(rotation_angle, expand=False)

        rotated_image = np.array(rotated_image)
        return rotated_image

    def __getitem__(self, index):
        lower_path = ROOT_PATH + '/experiments/Result_Data/for_labeling/lower_layer/' + self.data[index]['image_layer']
        upper_path = ROOT_PATH + '/experiments/Result_Data/for_labeling/upper_layer/' + self.data[index]['image_layer']
        soft_tissue_path = ROOT_PATH + '/experiments/Result_Data/for_labeling/soft_tissue/' + self.data[index]['image_layer']
        parameter = self.data[index]['parameter']
        jsw = self.data[index]['jsw']

        # base moved creat JSW=0
        lower_layer = np.array(Image.open(lower_path).convert("L")) / 255
        upper_layer = np.array(Image.open(upper_path).convert("L")) / 255
        soft_tissue = np.array(Image.open(soft_tissue_path).convert("L")) / 255

        # base moved creat JSW=0
        upper_moved = self.translated_image(upper_layer, parameter=parameter[:3])
        lower_moved = self.translated_image(lower_layer, parameter=parameter[3:])

        recon_image = 1 - ((1 - upper_moved) * (1 - lower_moved) * (1 - soft_tissue))
        recon_image = self.transform(Image.fromarray(recon_image))

        jsw = torch.tensor(jsw).unsqueeze(0)

        return recon_image, jsw

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = '/Users/wanghaolin/PycharmProjects/LSN/experiments/Exp_downstream/JSW_evaluation/pre_JSW_8_data.json'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       drop_last=True
                                       )
    print(len(data))
    for image, jsw in data:
        print(jsw.shape)
        plt.imshow(image.numpy()[0][0], 'gray')
        plt.title('{}'.format(jsw))
        plt.show()
