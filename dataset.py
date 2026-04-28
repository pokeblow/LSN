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
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import copy

ROOT_PATH = os.getcwd()


class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None, mode='All'):
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.root_path = os.path.dirname(self.data_path)
        self.transform = transform

    def mask_edge_smooth(self, mask):
        mask_smooth = cv2.GaussianBlur(mask, (11, 11), sigmaX=10, sigmaY=10)
        mask_smooth[mask_smooth > 130] = 255
        mask_smooth[mask_smooth <= 130] = 0

        # # 合成平滑mask
        # smoothed_mask = cv2.addWeighted(mask, 0.8, blurred_edges, 0.2, 0)

        return mask_smooth

    def create_overlap(self, image, upper_mask, lower_mask, background_mask, upper_layer, lower_layer, background,
                       soft_tissue_layer):
        def get_center(mask):
            M = cv2.moments(mask)

            if M["m00"] != 0:
                Cx = int(M["m10"] / M["m00"])
                Cy = int(M["m01"] / M["m00"])
            else:
                Cx, Cy = 0, 0
            return (Cx, Cy)

        def translation(layer, mask, tans_factor, scale_factor):
            org_center = get_center(mask)
            width, height = layer.shape

            M_scale = np.float32([
                [scale_factor, 0, 0],  # 缩放
                [0, scale_factor, 0]
            ])

            layer_scaled = cv2.warpAffine(layer, M_scale, (width, height))
            mask_scaled = cv2.warpAffine(mask, M_scale, (width, height))

            new_center = get_center(mask_scaled)

            back_tans = (new_center[0] - org_center[0], new_center[1] - org_center[1])

            M_translate_back = np.float32([
                [1, 0, -back_tans[0]],  # x方向移动回去
                [0, 1, -back_tans[1] + tans_factor]  # y方向移动回去并下移
            ])

            output_layer = cv2.warpAffine(layer_scaled, M_translate_back, (width, height))
            output_mask = cv2.warpAffine(mask_scaled, M_translate_back, (width, height))

            return output_layer, output_mask

        def edge_GaussianBlur(mask, image, kernel_size=4, ksize=3):
            image = (image * 255).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)

            edges = cv2.Canny(mask, 100, 200)

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            blurred_edges = cv2.GaussianBlur(image, (ksize, ksize), 0)
            blurred_mask = cv2.bitwise_and(blurred_edges, blurred_edges, mask=dilated_edges)

            inverse_mask = cv2.bitwise_not(dilated_edges)
            non_blurred_image = cv2.bitwise_and(image, image, mask=inverse_mask)

            final_image = cv2.add(blurred_mask, non_blurred_image)

            return final_image / 255

        def resize(image, range_size, height, width):
            image = image[range_size:width - range_size, range_size:height - range_size]
            image = cv2.resize(image, (width, height))
            return image

        # translation parameter
        height, width = image.shape
        random_A = random.randint(2, 8)
        random_B = -random.randint(2, 8)
        scaling = 1 + (max(random_A, random_B) * 3 / height)

        # translation
        upper_layer, upper_mask = translation(upper_layer, upper_mask, random_A, scaling)
        lower_layer, lower_mask = translation(lower_layer, lower_mask, random_B, scaling)

        mask_overlap = cv2.bitwise_and(upper_mask, lower_mask)
        mask_bone = cv2.bitwise_or(upper_mask, lower_mask)

        soft_tissue_mask = 1 - cv2.bitwise_or(mask_bone, background_mask)

        soft_tissue_layer = image * soft_tissue_mask

        # recon to image
        overlap = background * mask_overlap
        non_zero_values = overlap[overlap != 0]
        if np.sum(mask_overlap) != 0:
            average_value = np.mean(non_zero_values)
            overlap = mask_overlap * average_value

        overlap_trans = 1 - overlap

        upper_layer = upper_layer * upper_mask
        lower_layer = lower_layer * lower_mask
        image = 1 - ((1 - upper_layer) * (1 - lower_layer) / overlap_trans) + soft_tissue_layer
        image = edge_GaussianBlur(mask_overlap, image)
        image = edge_GaussianBlur(mask_bone, image, kernel_size=5, ksize=5)

        # image resize
        range_size = max(random_A, random_B) * 2 + 3

        image = resize(image, range_size, height, width)
        upper_layer = resize(upper_layer, range_size, height, width)
        lower_layer = resize(lower_layer, range_size, height, width)
        soft_tissue_layer = resize(soft_tissue_layer, range_size, height, width)

        upper_mask = resize(upper_mask, range_size, height, width)
        lower_mask = resize(lower_mask, range_size, height, width)
        soft_tissue_mask = resize(soft_tissue_mask, range_size, height, width)

        mask_all = cv2.bitwise_or(self.gaussianBlur(cv2.bitwise_or(upper_mask, lower_mask)), soft_tissue_mask)

        image = image * mask_all

        # # plt.imshow(upper_mask,alpha=0.5)
        # # plt.imshow(lower_mask,alpha=0.5)
        # plt.imshow(soft_tissue_mask,alpha=0.5)
        # plt.show()

        return image, upper_mask, lower_mask, soft_tissue_mask, mask_all, upper_layer, lower_layer, soft_tissue_layer

    def gaussianBlur(self, mask):
        mask = cv2.GaussianBlur(mask * 255, (3, 3), 0)
        mask[mask != 0] = 1.0
        return mask

    def __getitem__(self, index):
        # Data path
        image_name = self.data[index]['image']
        image_path = self.root_path + '/' + self.data[index]['image']
        upper_mask_path = self.root_path + '/' + self.data[index]['upper']
        lower_mask_path = self.root_path + '/' + self.data[index]['lower']
        background_mask_path = self.root_path + '/' + self.data[index]['image'][:-4] + '_mask_background.bmp'
        background_path = self.root_path + '/' + self.data[index]['background']

        image = np.array(Image.open(image_path).convert("L"))
        background = np.array(Image.open(background_path).convert("L"))
        background_mask = np.array(Image.open(background_mask_path))
        upper_mask = np.array(Image.open(upper_mask_path))
        lower_mask = np.array(Image.open(lower_mask_path))

        # upper_mask = self.mask_edge_smooth(upper_mask)
        # lower_mask = self.mask_edge_smooth(lower_mask)

        # Normalization to 0-1
        image = image / 255
        background = background / 255
        background_mask[background_mask != 0] = 1.0
        upper_mask[upper_mask != 0] = 1.0
        lower_mask[lower_mask != 0] = 1.0

        def gaussianBlur(mask):
            mask = cv2.GaussianBlur(mask * 255, (3, 3), 0)
            mask[mask != 0] = 1.0
            return mask

        # layer mask
        upper_mask = gaussianBlur(upper_mask)
        lower_mask = gaussianBlur(lower_mask)
        background_mask = gaussianBlur(background_mask)
        bone_mask = cv2.bitwise_or(upper_mask, lower_mask)
        overlap_size = np.sum(cv2.bitwise_and(upper_mask, lower_mask))

        soft_tissue_mask = 1 - cv2.bitwise_or(bone_mask, background_mask)

        return image, upper_mask, lower_mask, background_mask, soft_tissue_mask, background, overlap_size, image_name

    def __len__(self):
        return len(self.data)


class Data_Loader_All(Data_Loader):
    def __init__(self, data_path, transform=None):
        super().__init__(data_path, transform)

    def __getitem__(self, index):
        # 调用父类的__getitem__方法，获取基础的返回值
        image, upper_mask, lower_mask, background_mask, soft_tissue_mask, background, overlap_size, image_name = super().__getitem__(
            index)
        

        # layer image
        upper_layer = image * upper_mask
        lower_layer = image * lower_mask
        soft_tissue_layer = image * soft_tissue_mask

        # creat overlap
        if overlap_size <= 10:
            image, upper_mask, lower_mask, soft_tissue_mask, mask_all, upper_layer, lower_layer, soft_tissue_layer = self.create_overlap(
                image, upper_mask, lower_mask, background_mask,
                upper_layer, lower_layer,
                background, soft_tissue_layer)
        else:
            image = image * (1-background_mask)
            mask_all = cv2.bitwise_or(self.gaussianBlur(cv2.bitwise_or(upper_mask, lower_mask)), soft_tissue_mask)

        # Generator:
        image = self.transform(Image.fromarray(image))
        lower_layer = self.transform(Image.fromarray(lower_layer))
        upper_layer = self.transform(Image.fromarray(upper_layer))


        upper_mask = torch.tensor(
            cv2.resize(upper_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        lower_mask = torch.tensor(
            cv2.resize(lower_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        soft_tissue_mask = torch.tensor(cv2.resize(soft_tissue_mask, (image.shape[-1], image.shape[-1]),
                                                   interpolation=cv2.INTER_NEAREST)).unsqueeze(0)

        mask_all = torch.tensor(
            cv2.resize(mask_all, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        
        layer_masks = torch.cat((lower_mask, upper_mask, mask_all))

        # Discriminator:
        overlap_mask = torch.sum(torch.cat((upper_mask, lower_mask)), dim=0).unsqueeze(0)
        overlap_mask[overlap_mask != 2] = 0
        overlap_mask = overlap_mask // 2

        layer_masks_crop = torch.cat((lower_mask - overlap_mask, upper_mask - overlap_mask, soft_tissue_mask))

        return image, layer_masks, layer_masks_crop


class Data_Loader_Overlap(Data_Loader):
    def __init__(self, data_path, transform=None):
        super().__init__(data_path, transform)

        self.recent_image_name = ''

    def __getitem__(self, index):
        # 调用父类的__getitem__方法，获取基础的返回值
        image, upper_mask, lower_mask, background_mask, soft_tissue_mask, background, overlap_size, image_name = super().__getitem__(
            index)
        
        self.recent_image_name = image_name
        # layer image
        upper_layer = image * upper_mask
        lower_layer = image * lower_mask
        soft_tissue_layer = image * soft_tissue_mask

        image = image * (1-background_mask)
        mask_all = cv2.bitwise_or(self.gaussianBlur(cv2.bitwise_or(upper_mask, lower_mask)), soft_tissue_mask)

        # Generator:
        image = self.transform(Image.fromarray(image))
        lower_layer = self.transform(Image.fromarray(lower_layer))
        upper_layer = self.transform(Image.fromarray(upper_layer))
        

        upper_mask = torch.tensor(
            cv2.resize(upper_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        lower_mask = torch.tensor(
            cv2.resize(lower_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        soft_tissue_mask = torch.tensor(cv2.resize(soft_tissue_mask, (image.shape[-1], image.shape[-1]),
                                                   interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        mask_all = torch.tensor(
            cv2.resize(mask_all, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        
        layer_masks = torch.cat((lower_mask, upper_mask, mask_all))

        # Discriminator:
        overlap_mask = torch.sum(torch.cat((upper_mask, lower_mask)), dim=0).unsqueeze(0)
        overlap_mask[overlap_mask != 2] = 0
        overlap_mask = overlap_mask // 2

        layer_masks_crop = torch.cat((lower_mask - overlap_mask, upper_mask - overlap_mask, soft_tissue_mask))

        return image, layer_masks, layer_masks_crop, image_name


class Data_Loader_Nonoverlap(Data_Loader):
    def __init__(self, data_path, transform=None):
        super().__init__(data_path, transform)

    def __getitem__(self, index):
        # 调用父类的__getitem__方法，获取基础的返回值
        image, upper_mask, lower_mask, background_mask, soft_tissue_mask, background, overlap_size, image_name = super().__getitem__(
            index)

        # image with nonoverlap
        image_nonoverlap = copy.deepcopy(image)
        upper_mask_nonoverlap = copy.deepcopy(upper_mask)
        lower_mask_nonoverlap = copy.deepcopy(lower_mask)
        soft_tissue_mask_nonoverlap = copy.deepcopy(soft_tissue_mask)

        mask_all_nonoverlap = cv2.bitwise_or(self.gaussianBlur(cv2.bitwise_or(upper_mask_nonoverlap, lower_mask_nonoverlap)), soft_tissue_mask_nonoverlap)
        image_nonoverlap = image_nonoverlap * mask_all_nonoverlap

        # layer image
        upper_layer = image * upper_mask
        lower_layer = image * lower_mask
        soft_tissue_layer = image * soft_tissue_mask

        # creat overlap
        image, upper_mask, lower_mask, soft_tissue_mask, mask_all, upper_layer, lower_layer, soft_tissue_layer = self.create_overlap(
            image, upper_mask, lower_mask, background_mask,
            upper_layer, lower_layer,
            background, soft_tissue_layer)

        # To Tensor
        image_nonoverlap = self.transform(Image.fromarray(image_nonoverlap))
        upper_mask_nonoverlap = torch.tensor(
            cv2.resize(upper_mask_nonoverlap, (image_nonoverlap.shape[-1], image_nonoverlap.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        lower_mask_nonoverlap = torch.tensor(
            cv2.resize(lower_mask_nonoverlap, (image_nonoverlap.shape[-1], image_nonoverlap.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        soft_tissue_mask_nonoverlap = torch.tensor(cv2.resize(soft_tissue_mask_nonoverlap, (image_nonoverlap.shape[-1], image_nonoverlap.shape[-1]),
                                                   interpolation=cv2.INTER_NEAREST)).unsqueeze(0)

        layer_masks_nonoverlap = torch.cat((lower_mask_nonoverlap, upper_mask_nonoverlap, soft_tissue_mask_nonoverlap))


        # Generator:
        image = self.transform(Image.fromarray(image))
        lower_layer = self.transform(Image.fromarray(lower_layer))
        upper_layer = self.transform(Image.fromarray(upper_layer))
        bone_layers = torch.cat((image, image))

        upper_mask = torch.tensor(
            cv2.resize(upper_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        lower_mask = torch.tensor(
            cv2.resize(lower_mask, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        soft_tissue_mask = torch.tensor(cv2.resize(soft_tissue_mask, (image.shape[-1], image.shape[-1]),
                                                   interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        mask_all = torch.tensor(
            cv2.resize(mask_all, (image.shape[-1], image.shape[-1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0)

        bone_layers = bone_layers * torch.cat((lower_mask, upper_mask))
        layer_masks = torch.cat((lower_mask, upper_mask, mask_all))

        # Discriminator:
        overlap_mask = torch.sum(torch.cat((upper_mask, lower_mask)), dim=0).unsqueeze(0)
        overlap_mask[overlap_mask != 2] = 0
        overlap_mask = overlap_mask // 2

        layer_masks_crop = torch.cat((lower_mask - overlap_mask, upper_mask - overlap_mask, soft_tissue_mask))

        bone_layers_GT = torch.cat((lower_layer, upper_layer))

        return image, layer_masks, layer_masks_crop, bone_layers, bone_layers_GT, image_nonoverlap, layer_masks_nonoverlap


if __name__ == "__main__":
    data_path = ROOT_PATH + '/Data/LS_K1_nonoverlap_train'
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader_Nonoverlap(data_path, transform)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       drop_last=True,

                                       )
    for image, layer_masks, layer_masks_crop, bone_layers, bone_layers_GT, image_nonoverlap, layer_masks_nonoverlap in data:
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy()[0][0], 'gray')
        plt.subplot(1, 3, 2)
        plt.imshow(layer_masks_nonoverlap.numpy()[0][0])
        # plt.imshow(layer_masks.numpy()[0][1], alpha=0.5)
        # plt.imshow(layer_masks.numpy()[0][2], alpha=0.5)
        plt.subplot(1, 3, 3)
        plt.imshow(layer_masks_nonoverlap.numpy()[0][2])
        # plt.imshow(layer_masks_crop.numpy()[0][1], alpha=0.5)
        # plt.imshow(layer_masks_crop.numpy()[0][2], alpha=0.5)
        plt.tight_layout()
        plt.show()
