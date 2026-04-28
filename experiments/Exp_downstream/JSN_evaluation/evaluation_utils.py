import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import random
# import utils.retinex as retinex
import os
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(root_dir)

import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import library as lb

import pandas as pd
import json
import library as lb

ROOT_PATH = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), os.pardir))


def soomth(mask, kernel_size):
    x_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[j][i] == 0:
                x_list.append(j)
                break

    half = kernel_size // 2
    x_list_conv = []
    for k in range(half):
        x_list_conv.append(x_list[k])
    for k in range(half, len(x_list) - half):
        tmp = []
        tmp.append(x_list[k])
        for m in range(1, half + 1):
            tmp.append(x_list[k - m])
            tmp.append(x_list[k + m])
        x_list_conv.append(int(np.mean(tmp)))
    for k in range(len(x_list) - half, len(x_list)):
        x_list_conv.append(x_list[k])

    mask_new = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_new[j][i] = 1
            if j == x_list_conv[i]:
                break
    return mask_new


def removePoint(list):
    return [x for x in list if not x.startswith(('.'))]

def seg_Gully(image, kernel_size=3):
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


def predict(net_R, moving_image, fixed_image, moving_mask, fixed_mask):
    org_size = moving_image.shape[-1]
    range_crop = 5

    loss_org = F.mse_loss(moving_image, fixed_image)
    loss_org = loss_org.cpu().detach().numpy()

    moving_mask = moving_mask[:, 0, :, :].unsqueeze(1)
    fixed_mask = fixed_mask[:, 0, :, :].unsqueeze(1)

    moving_reg, fixed_crop, moving_masks_reg, fixed_masks_reg, parameter = net_R(moving_image, fixed_image, moving_mask,
                                                                                 fixed_mask)
    parameter = parameter.cpu().detach().numpy()[0]

    loss_lower = F.mse_loss(moving_reg[:, 0], fixed_crop[:, 0])
    loss_upper = F.mse_loss(moving_reg[:, 1], fixed_crop[:, 1])

    upper_result = parameter[1][3] * (org_size / 256) * 0.15
    lower_result = parameter[0][3] * (org_size / 256) * 0.15

    loss = 0.5 * loss_upper + 0.5 * loss_lower
    loss = loss.cpu().detach().numpy()

    JSN = upper_result - lower_result

    loss_image = F.mse_loss(moving_reg, fixed_crop, reduction='none')
    loss_image = torch.mean(loss_image, dim=1).cpu().detach().numpy()[0]

    moving_reg = moving_reg.cpu().detach().numpy()[0]

    return JSN, loss, loss_org, moving_reg, loss_image


def inputimage_path(image_path):
    image = np.array(Image.open(image_path).convert('L'))
    upper_path = image_path[:-4] + '_mask_upper.bmp'
    lower_path = image_path[:-4] + '_mask_lower.bmp'
    background_mask_path = image_path[:-4] + '_mask_background.bmp'

    mask = lb.separation(lb.gullyDetect(image))[0]
    mask[mask != 0] = 1

    upper_mask = np.array(Image.open(upper_path))
    lower_mask = np.array(Image.open(lower_path))
    background_mask = np.array(Image.open(background_mask_path))

    # Normalization to 0-1
    image = image / 255
    upper_mask[upper_mask != 0] = 1.0
    lower_mask[lower_mask != 0] = 1.0
    upper_mask = cv2.GaussianBlur(upper_mask * 255, (3, 3), 0)
    upper_mask[upper_mask != 0] = 1
    lower_mask = cv2.GaussianBlur(lower_mask * 255, (3, 3), 0)
    lower_mask[lower_mask != 0] = 1
    background_mask = cv2.GaussianBlur(background_mask * 255, (3, 3), 0)
    lower_mask[background_mask != 0] = 1

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])

    image = image * (1-background_mask)

    overlap_count = np.sum(cv2.bitwise_and(upper_mask, lower_mask))

    upper_mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    lower_mask = torch.tensor(1 - mask).unsqueeze(0).unsqueeze(0).float()

    image = transform(Image.fromarray(image)).unsqueeze(0)

    image_mask = torch.cat((lower_mask, upper_mask), dim=1)

    return image, image_mask, overlap_count


def save_image(joint_name, i, j, moving, fixed, moving_reg, loss_image):
    root_path = ROOT_PATH + '/experiments/Exp_downstream/JSN_evaluation/result_image/'
    print(root_path)
    i = i[:-4]
    j = j[:-4]

    range_crop = 5
    moving = moving[:, :, range_crop: 256 - range_crop, range_crop: 256 - range_crop]
    fixed = fixed[:, :, range_crop: 256 - range_crop, range_crop: 256 - range_crop]

    cv2.imwrite(root_path + 'moving/{}_{}_{}.jpg'.format(joint_name, i, j),
                moving.cpu().detach().numpy()[0][0] * 255)
    cv2.imwrite(root_path + 'fixed/{}_{}_{}.jpg'.format(joint_name, i, j), fixed.cpu().detach().numpy()[0][0] * 255)

    cv2.imwrite(root_path + 'loss/{}_{}_{}.jpg'.format(joint_name, i, j), loss_image * 255)

    cv2.imwrite(root_path + 'warpped_upper/{}_{}_{}.jpg'.format(joint_name, i, j), moving_reg[1] * 255)
    cv2.imwrite(root_path + 'warpped_lower/{}_{}_{}.jpg'.format(joint_name, i, j), moving_reg[0] * 255)


def moving_1pixel_image(image, image_mask):
    image_mask = copy.deepcopy(image_mask.numpy()[0])
    image = copy.deepcopy(image.numpy()[0][0])

    mask_upper = image_mask[1]
    mask_lower = image_mask[0]

    h, w = image.shape

    x = random.randint(-3, 3)
    y = random.randint(-3, 3)
    M_T = np.float32([[1, 0, x], [0, 1, y]])

    mask_upper = cv2.warpAffine(mask_upper, M_T, (h, w))
    mask_lower = cv2.warpAffine(mask_lower, M_T, (h, w))

    image = cv2.warpAffine(image, M_T, (h, w))

    mask_upper = torch.tensor([mask_upper])
    mask_lower = torch.tensor([mask_lower])

    image = torch.tensor([image]).unsqueeze(0)

    image_mask = torch.cat((mask_upper, mask_lower)).unsqueeze(0)

    return image, image_mask


def input_image(target='', index=0, data=[]):
    root_path = ROOT_PATH + '/Data/Dataset_cases/'
    image_path = root_path + '/' + data[index][target]
    upper_mask_path = root_path + '/' + data[index]['{}_upper'.format(target)]
    lower_mask_path = root_path + '/' + data[index]['{}_lower'.format(target)]
    bcakground_root_path = root_path + '/background'
    background_path = bcakground_root_path + '/' + data[index][target].replace('/', '_')

    image = np.array(Image.open(image_path).convert("L"))
    background = np.array(Image.open(background_path).convert("L"))
    upper_mask = np.array(Image.open(upper_mask_path))
    lower_mask = np.array(Image.open(lower_mask_path))

    # Normalization to 0-1
    image = image / 255
    background = background / 255
    upper_mask[upper_mask != 0] = 1.0
    lower_mask[lower_mask != 0] = 1.0

    upper_mask = cv2.GaussianBlur(upper_mask * 255, (3, 3), 0)
    upper_mask[upper_mask != 0] = 1

    lower_mask = cv2.GaussianBlur(lower_mask * 255, (3, 3), 0)
    lower_mask[lower_mask != 0] = 1

    return image, upper_mask, lower_mask


def filter(data_list):
    data_filtered_list = []
    for index in range(len(data_list)):
        moving_image, moving_upper_mask, moving_lower_mask = input_image(target='moving', index=index, data=data_list)
        fixed_image, fixed_upper_mask, fixed_lower_mask = input_image(target='fixed', index=index, data=data_list)

        moving_mask_overlap = np.sum(cv2.bitwise_and(moving_upper_mask, moving_lower_mask))
        fixed_mask_overlap = np.sum(cv2.bitwise_and(fixed_upper_mask, fixed_lower_mask))

        if moving_mask_overlap == 0 and fixed_mask_overlap == 0:
            data_filtered_list.append(data_list[index])

    return data_filtered_list


def evaluation_main(net_R, data_path, data_root, save_path):
    SD_max = 0.3

    # 读取Excel文件
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)

    joint_list = []
    for i in data:
        joint_list.append(i['joint_name'])
    joint_list = list(set(joint_list))
    joint_list.sort()
    print(len(joint_list))

    with open(save_path + '_JSN_loss.txt', 'a+', encoding='utf-8') as file:
        file.truncate(0)
    with open(save_path + '_JSN_summary.txt', 'a+', encoding='utf-8') as file:
        file.truncate(0)

    for joint_name in joint_list:
        finger_name = joint_name[:-2]
        finger_index = joint_name[-1]
        print('Finger:', finger_name, 'Joint index', finger_index)

        if True:
            joint_path = data_root + joint_name
            print(joint_path)
            # joint
            joint_SD_list = []
            joint_loss_list = []
            joint_SD_list_2 = []
            joint_mis_list = []
            count = 0
            mis = 0

            # remove .bmp
            phase_list = os.listdir(joint_path)
            phase_list = [file for file in phase_list if '.bmp' not in file]

            len_phase = len(phase_list)
            print('Joint Name:', joint_name, 'Phase len:', len_phase)
            JSN_MATRIX = np.ones([len_phase, len_phase])

            overlap_count_list = []
            # overlap count
            phase_list_new = []
            for p in range(len(phase_list)):
                image_path = joint_path + '/' + phase_list[p]

                image, image_layer, overlap_size = inputimage_path(image_path)
                overlap_count_list.append(overlap_size)
                phase_list_new.append(phase_list[p])
                print('phase: {}, overlap: {}'.format(p, overlap_size))

            phase_list = phase_list_new

            # phase
            for i in range(len(phase_list)):
                for j in range(len(phase_list)):
                    if i != j:

                        moving_path = joint_path + '/' + phase_list[i]
                        fixed_path = joint_path + '/' + phase_list[j]

                        moving, moving_mask, overlap_size_moving = inputimage_path(moving_path)
                        fixed, fixed_mask, overlap_size_fixed = inputimage_path(fixed_path)

                        tmp_JSN_list = []
                        tmp_loss_list = []
                        tmp_img_list = []

                        # JSN output
                        JSN, loss, loss_org, moving_reg, loss_image = predict(net_R, moving, fixed, moving_mask,
                                                                              fixed_mask)

                        save_image(joint_name, phase_list[i], phase_list[j], moving, fixed, moving_reg, loss_image)

                        tmp_JSN_list.append(JSN)
                        tmp_loss_list.append(loss)
                        tmp_img_list.append(moving_reg)

                        for loop in range(9):
                            moving_tmp, moving_mask_tmp = moving_1pixel_image(moving, moving_mask)
                            fixed_tmp, fixed_mask_tmp = moving_1pixel_image(fixed, fixed_mask)

                            JSN_loop, loss_loop, loss_org_loop, image_loop, _ = predict(net_R, moving_tmp, fixed_tmp,
                                                                                        moving_mask_tmp,
                                                                                        fixed_mask_tmp)

                            print(JSN_loop)

                            tmp_JSN_list.append(JSN_loop)
                            tmp_loss_list.append(loss_loop)
                            tmp_img_list.append(image_loop)

                            tmp_miss_index_list = []

                        for loopk in range(9):
                            if np.std(tmp_JSN_list, ddof=1) < SD_max:
                                break
                            distanceToMean = abs(tmp_JSN_list - np.mean(tmp_JSN_list))
                            maxindex = np.argmax(distanceToMean)
                            tmp_miss_index_list.append(maxindex)
                            tmp_JSN_list = np.delete(tmp_JSN_list, maxindex)
                            tmp_loss_list = np.delete(tmp_loss_list, maxindex)

                        f_loss = open(save_path + '_JSN_loss.txt', 'a')
                        f_loss.writelines(
                            '{},{},{},{},{},{},{},{},{},{}'.format(finger_name, finger_index, phase_list[i],
                                                                   phase_list[j],
                                                                   JSN, loss, loss_org,
                                                                   np.std(tmp_JSN_list, ddof=1),
                                                                   overlap_size_moving,
                                                                   overlap_size_fixed))
                        f_loss.write('\n')
                        f_loss.close()

                        if len(tmp_JSN_list) > 3:
                            distance = np.mean(tmp_JSN_list)
                            tmp_SD = np.std(tmp_JSN_list, ddof=1)
                            joint_SD_list.append(tmp_SD)
                            joint_loss_list.append(np.mean(tmp_loss_list))
                            mis += 10 - len(tmp_JSN_list)
                            joint_mis_list.append((10 - len(tmp_JSN_list)) / 10)
                            count += 10

                            # JSN_MATRIX
                            JSN_MATRIX[i][j] = distance
                            print('{}, {}, ({}, {}) JSN: {} Loss: {} SD: {}, overlap: ({}, {})'.format(finger_name,
                                                                                                       finger_index, i,
                                                                                                       j, distance,
                                                                                                       loss,
                                                                                                       tmp_SD,
                                                                                                       overlap_size_moving,
                                                                                                       overlap_size_fixed
                                                                                                       ))
                        else:
                            JSN_MATRIX[i][j] = 2000
                            mis += 10
                            joint_mis_list.append(1)
                            count += 10

            # SD
            for i in range(len(phase_list)):
                for j in range(len(phase_list)):
                    if (i != j) and (JSN_MATRIX[i][j] < 100):
                        tmp_JSN_list_2 = []
                        JSN_ij = JSN_MATRIX[i][j]
                        tmp_JSN_list_2.append(JSN_ij)
                        for k in range(len(phase_list)):
                            if ((k != i) and (k != j)) and ((JSN_MATRIX[k][j] < 100) and (JSN_MATRIX[i][k] < 100)):
                                JSN_ik = JSN_MATRIX[i][k]
                                JSN_kj = JSN_MATRIX[k][j]
                                JSN_ij_2 = JSN_ik + JSN_kj
                                tmp_JSN_list_2.append(JSN_ij_2)

                        # if len(tmp_JSN_list_2) == 1:
                        #     tmp_SD = 0

                        tmp_SD = np.std(tmp_JSN_list_2, ddof=1)
                        joint_SD_list_2.append(tmp_SD)

            Joint_LOSS_Mean = np.mean(joint_loss_list)
            Joint_SD_Mean = np.nanmean(joint_SD_list)
            Joint_SD2_Mean = np.nanmean(joint_SD_list_2)
            Joint_mis_Mean = np.nanmean(joint_mis_list)
            print('{}, Mean Loss: {}, Mean SD: {} Mean SD2: {} Miss Rate: {}, {}/{}, overlap: {}'.format(joint_name,
                                                                                                         Joint_LOSS_Mean,
                                                                                                         Joint_SD_Mean,
                                                                                                         Joint_SD2_Mean,
                                                                                                         Joint_mis_Mean,
                                                                                                         mis,
                                                                                                         count,
                                                                                                         overlap_count_list))
            f_test = open(save_path + '_JSN_summary.txt', 'a')
            f_test.writelines(
                '{},{},{},{},{},{},{},{},{}'.format(finger_name, finger_index, Joint_LOSS_Mean, Joint_SD_Mean,
                                                    Joint_SD2_Mean, mis, count,
                                                    Joint_mis_Mean, overlap_count_list))
            f_test.write('\n')
            f_test.close()
