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

    def __getitem__(self, index):
        image_path = ROOT_PATH + '/Data/' + self.data[index]['image_GT']
        image = np.array(Image.open(image_path).convert("L"))
        parameter = self.data[index]['parameter']
        label = [0, 0, 0, 0, 0]

        jsw = abs(parameter[1]) / 256 * 140 * 0.15

        normal_jsw = 1.75
        intervals = [(3 * normal_jsw, normal_jsw),
                     (normal_jsw, normal_jsw * (1 - 0.25)),
                     (normal_jsw * (1 - 0.25), normal_jsw * 0.5),
                     (normal_jsw * 0.5, normal_jsw * (1 - 0.75)),
                     (normal_jsw * (1 - 0.75), 0)
                     ]

        for index, (high, low) in enumerate(intervals):
            if high > jsw >= low:
                label[index] = 1

        image = self.transform(Image.fromarray(image))
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = '/Users/wanghaolin/PycharmProjects/LSN/experiments/Exp_downstream/JSW_evaluation/main_JSW_train_data.json'
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
    labelsum = torch.tensor([0, 0, 0, 0, 0]).unsqueeze(0)
    for image, label in data:
        labelsum = labelsum + label
    print(labelsum)
    # plt.imshow(image.numpy()[0][0], 'gray')
    # plt.show()
