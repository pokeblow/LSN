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
        image = np.array(Image.open(image_path).convert("L")) / 255
        parameter = self.data[index]['parameter']

        image = self.transform(Image.fromarray(image))
        jsw = torch.tensor(abs(parameter[1])).unsqueeze(0)

        return image, jsw

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
    for image, jsw in data:
        print(jsw.shape)
        # plt.imshow(image.numpy()[0][0], 'gray')
        # plt.show()

