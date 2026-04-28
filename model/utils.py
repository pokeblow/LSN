import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision
import numpy as np
import random

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

def random_move_and_reconstruct(layer, soft_tissue_mask):
    x_max = 0
    y_max = 20
    a_max = 0
    s_range = 0
    
    shift_x = random.randint(-x_max, x_max)
    shift_y = random.randint(-y_max, y_max)
    rotate_a = random.randint(-a_max, a_max)
    scale_s = random.randint(10 - s_range, 10 + s_range) / 10
    output_layer = torchvision.transforms.functional.affine(layer, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)

    output_layer = output_layer * soft_tissue_mask

    return output_layer

def random_movement(layer, mask, soft_tissue_mask):
    # x_max = 20
    # y_max = x_max + 10
    # a_max = 10
    # s_range = 0
    
    # shift_x = random.randint(-x_max, x_max)
    # shift_y = random.randint(-y_max, y_max)
    # rotate_a = random.randint(-a_max, a_max)
    # scale_s = random.randint(10 - s_range, 10 + s_range) / 10
    # output_mask = torchvision.transforms.functional.affine(mask, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)
    # output_layer = torchvision.transforms.functional.affine(layer, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)

    # output_mask = output_mask * soft_tissue_mask
    # output_layer = output_layer * soft_tissue_mask

    # # while torch.min(soft_tissue_mask - output_mask) < 0:
    # #     shift_x = random.randint(-x_max_shift, x_max_shift)
    # #     shift_y = random.randint(-y_max_shift, y_max_shift)
    # #     output_mask = torchvision.transforms.functional.affine(mask, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)

    # # output_mask = torchvision.transforms.functional.affine(mask, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)
    # # output_layer = torchvision.transforms.functional.affine(layer, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)

    output_mask = mask * soft_tissue_mask
    output_layer = layer * soft_tissue_mask
    
    return output_layer, output_mask

def random_movement_with_GT(layer, layer_GT, mask, soft_tissue_mask):
    # x_max = 20
    # y_max = x_max + 10
    # a_max = 10
    # s_range = 0
    
    # shift_x = random.randint(-x_max, x_max)
    # shift_y = random.randint(-y_max, y_max)
    # rotate_a = random.randint(-a_max, a_max)
    # scale_s = random.randint(10 - s_range, 10 + s_range) / 10
    # output_mask = torchvision.transforms.functional.affine(mask, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)
    # output_layer = torchvision.transforms.functional.affine(layer, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)
    # output_layer_GT = torchvision.transforms.functional.affine(layer_GT, angle=rotate_a, translate=(shift_x, shift_y), scale=scale_s, shear=[0, 0], fill=0)

    # output_mask = output_mask * soft_tissue_mask
    # output_layer = output_layer * soft_tissue_mask
    # output_layer_GT = output_layer_GT * soft_tissue_mask

    # # while torch.min(soft_tissue_mask - output_mask) < 0:
    # #     shift_x = random.randint(-x_max_shift, x_max_shift)
    # #     shift_y = random.randint(-y_max_shift, y_max_shift)
    # #     output_mask = torchvision.transforms.functional.affine(mask, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)

    # # output_mask = torchvision.transforms.functional.affine(mask, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)
    # # output_layer = torchvision.transforms.functional.affine(layer, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)
    # # output_layer_GT = torchvision.transforms.functional.affine(layer_GT, angle=0, translate=(shift_x, shift_y), scale=1, shear=[0, 0], fill=0)

    output_mask = mask * soft_tissue_mask
    output_layer = layer * soft_tissue_mask
    output_layer_GT = layer_GT * soft_tissue_mask

    return output_layer, output_layer_GT, output_mask


def reconstruction_2layer(layers, i, j):
    layer = layers[:, i, :, :].unsqueeze(1)
    soft_tissue_layer = layers[:, j, :, :].unsqueeze(1)

    trans = 1 - layer
    soft_tissue_trans = 1 - soft_tissue_layer

    output = (1 - (trans * soft_tissue_trans))

    return output

def reconstruction_all(layers):
    upper_layer = layers[:, 1, :, :].unsqueeze(1)
    lower_layer = layers[:, 0, :, :].unsqueeze(1)
    soft_tissue_layer = layers[:, 2, :, :].unsqueeze(1)

    upper_trans = 1 - upper_layer
    lower_trans = 1 - lower_layer
    soft_tissue_trans = 1 - soft_tissue_layer

    output = (1 - (upper_trans * lower_trans * soft_tissue_trans))

    return output


# with parameter
def reconstruction_st_layer_w_p(layers, parameter):
    soft_tissue_layer = layers[:, 2, :, :].unsqueeze(1)
    soft_tissue_trans = 1 - soft_tissue_layer
    parameter_trans = 1 - parameter

    output = (1 - (soft_tissue_trans * parameter_trans))

    return output

def reconstruction_2layer_w_p(layers, parameter, i, j):
    layer = layers[:, i, :, :].unsqueeze(1)
    soft_tissue_layer = layers[:, j, :, :].unsqueeze(1)

    trans = 1 - layer
    soft_tissue_trans = 1 - soft_tissue_layer
    parameter_trans = 1 - parameter

    output = (1 - (trans * soft_tissue_trans * parameter_trans))

    return output

def reconstruction_all_w_p(layers, parameter):
    upper_layer = layers[:, 1, :, :].unsqueeze(1)
    lower_layer = layers[:, 0, :, :].unsqueeze(1)
    soft_tissue_layer = layers[:, 2, :, :].unsqueeze(1)

    upper_trans = 1 - upper_layer
    lower_trans = 1 - lower_layer
    soft_tissue_trans = 1 - soft_tissue_layer
    parameter_trans = 1 - parameter

    output = (1 - (upper_trans * lower_trans * soft_tissue_trans * parameter_trans))

    return output



def bone_to_layers(layers, masks):
    dis_masks = torch.clone(masks[:, :2, :, :])
    lower_layer = reconstruction_2layer(layers, 0, 2)
    upper_layer = reconstruction_2layer(layers, 1, 2)
    bone_layers = torch.cat((lower_layer, upper_layer), dim=1) * dis_masks
    return bone_layers

def make_moved_label(layer_masks):
    all_bone_mask = torch.sum(torch.clone(layer_masks[:, :2, :, :]), dim=1).unsqueeze(1)
    all_bone_mask[all_bone_mask != 0] = 1

    overlap_mask = torch.sum(torch.clone(layer_masks[:, :2, :, :]), dim=1).unsqueeze(1) / 2
    overlap_mask[overlap_mask != 1] = 0
    overlap_masks = torch.cat((overlap_mask, overlap_mask), dim=1)

    label_cropped = torch.cat((layer_masks[:, :2, :, :] - overlap_masks, layer_masks[:, 2:, :, :] - all_bone_mask), dim=1)

    return label_cropped


def make_fake_label(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)
    zeros_layers = torch.zeros_like(layer_masks, device=device)

    label_fake_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask, zeros_layers), dim=1)
    label_fake_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask, zeros_layers), dim=1)
    label_fake_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask, zeros_layers), dim=1)

    return label_fake_lower, label_fake_upper, label_fake_soft_tissue


def make_fake_label_2(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)

    label_fake_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask, zeros_layer), dim=1)
    label_fake_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask, zeros_layer), dim=1)
    label_fake_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask, zeros_layer), dim=1)

    return label_fake_lower, label_fake_upper, label_fake_soft_tissue

# for 1_3
def make_fake_label_3(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)

    label_fake_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    label_fake_lower = torch.cat((label_fake_lower, label_fake_lower), dim=1)

    label_fake_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    label_fake_upper = torch.cat((label_fake_upper, label_fake_upper), dim=1)

    label_fake_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)
    label_fake_soft_tissue = torch.cat((label_fake_soft_tissue, label_fake_soft_tissue), dim=1)

    return label_fake_lower, label_fake_upper, label_fake_soft_tissue

# for 1_4
def make_seg_label(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)

    label_fake_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    label_fake_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    label_fake_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)

    return label_fake_lower, label_fake_upper, label_fake_soft_tissue



def make_real_label(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)

    label_real_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    label_real_lower = torch.cat((label_real_lower, label_real_lower), dim=1)

    label_real_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    label_real_upper = torch.cat((label_real_upper, label_real_upper), dim=1)

    label_real_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)
    label_real_soft_tissue = torch.cat((label_real_soft_tissue, label_real_soft_tissue), dim=1)

    return label_real_lower, label_real_upper, label_real_soft_tissue


def make_real_label_2(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)
    ones_layer = torch.ones_like(soft_tissue_mask, device=device)

    label_real_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    label_real_lower = torch.cat((label_real_lower, ones_layer), dim=1)

    label_real_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    label_real_upper = torch.cat((label_real_upper, ones_layer), dim=1)

    label_real_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)
    label_real_soft_tissue = torch.cat((label_real_soft_tissue, ones_layer), dim=1)

    return label_real_lower, label_real_upper, label_real_soft_tissue

# for 1_3
def make_real_label_3(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)
    zeros_layers = torch.zeros_like(layer_masks, device=device)

    label_real_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    label_real_lower = torch.cat((label_real_lower, zeros_layers), dim=1)

    label_real_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    label_real_upper = torch.cat((label_real_upper, zeros_layers), dim=1)

    label_real_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)
    label_real_soft_tissue = torch.cat((label_real_soft_tissue, zeros_layers), dim=1)

    return label_real_lower, label_real_upper, label_real_soft_tissue


def make_lossmask(layer_masks, device):
    lower_mask = layer_masks[:, 0, :, :].unsqueeze(1)
    upper_mask = layer_masks[:, 1, :, :].unsqueeze(1)
    soft_tissue_mask = layer_masks[:, 2, :, :].unsqueeze(1)
    zeros_layer = torch.zeros_like(soft_tissue_mask, device=device)

    lossmask_lower = torch.cat((lower_mask, zeros_layer, soft_tissue_mask - lower_mask), dim=1)
    lossmask_lower = torch.cat((lossmask_lower, lossmask_lower), dim=1)

    lossmask_upper = torch.cat((zeros_layer, upper_mask, soft_tissue_mask - upper_mask), dim=1)
    lossmask_upper = torch.cat((lossmask_upper, lossmask_upper), dim=1)

    lossmask_soft_tissue = torch.cat((zeros_layer, zeros_layer, soft_tissue_mask), dim=1)
    lossmask_soft_tissue = torch.cat((lossmask_soft_tissue, lossmask_soft_tissue), dim=1)

    return lossmask_lower, lossmask_upper, lossmask_soft_tissue


def move_layers_masks(layers, masks):
    soft_tissue_mask = masks[:, 2:, :, :]
    lower_layer_tmp, lower_mask_tmp = random_movement(layers[:, 0, :, :].unsqueeze(1), masks[:, 0, :, :].unsqueeze(1), soft_tissue_mask)
    upper_layer_tmp, upper_mask_tmp = random_movement(layers[:, 1, :, :].unsqueeze(1), masks[:, 1, :, :].unsqueeze(1), soft_tissue_mask)

    layers_moved = torch.cat((lower_layer_tmp, upper_layer_tmp, layers[:, 2:, :, :]), dim=1)
    masks_moved = torch.cat((lower_mask_tmp, upper_mask_tmp, masks[:, 2:, :, :]), dim=1)

    return layers_moved, masks_moved



def result_image_show(image_box):
    image, recon_image, pre_layers, D_show_Real_overalp, D_show_Real_nonoveralp, D_show_Fake_lower, D_show_Fake_upper, D_show_Fake_ST, D_show_Fake_recon_image, D_show_Fake_recon_image_tmp = image_box
    # 创建一个 figure 对象
    fig = plt.figure(figsize=(15, 12))

    # 创建一个 2 行 1 列的 gridspec 布局，上方占 1 行，下方占 1 行
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

    # 上方 5 张图像
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0])

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Original Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 1])
    ax.imshow(recon_image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Recon Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 2])
    ax.imshow(pre_layers[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Lower Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 3])
    ax.imshow(pre_layers[1], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Upper Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 4])
    ax.imshow(pre_layers[2], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Soft Tissue')
    ax.axis('off')

    # 下方 3x6 图像组
    gs_bottom = gridspec.GridSpecFromSubplotSpec(7, 6, subplot_spec=gs[1], hspace=0.25)
    # 定义每行的标题
    row_titles = ['Real Overlap', 'Real Nonoverlap', 'Upper', 'Lower', 'Soft Tissue', 'Recon Image', 'Recon Image moved']

    for j in range(6):
        ax = fig.add_subplot(gs_bottom[0, j])
        ax.imshow(D_show_Real_overalp[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title(row_titles[0], loc='left', fontsize=12)
            
        ax = fig.add_subplot(gs_bottom[1, j])
        ax.imshow(D_show_Real_nonoveralp[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[1], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[2, j])
        ax.imshow(D_show_Fake_upper[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[2], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[3, j])
        ax.imshow(D_show_Fake_lower[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[3], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[4, j])
        ax.imshow(D_show_Fake_ST[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[4], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[5, j])
        ax.imshow(D_show_Fake_recon_image[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[5], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[6, j])
        ax.imshow(D_show_Fake_recon_image_tmp[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title(row_titles[6], loc='left', fontsize=12)


    plt.tight_layout()
    plt.show()



def result_image_show_2(image_box):
    image, recon_image, pre_layers, pre_bone_layers, pre_bone_masks, pre_real_masks, pre_real_masks_2, bone_layers_tmp, bone_masks_tmp, recon_image_tmp = image_box
    # 创建一个 figure 对象
    fig = plt.figure(figsize=(20, 15))

    # 创建一个 2 行 1 列的 gridspec 布局，上方占 1 行，下方占 1 行
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # 上方 5 张图像
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0])

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Original Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 1])
    ax.imshow(recon_image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Recon Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 2])
    ax.imshow(pre_bone_layers[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Lower Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 3])
    ax.imshow(pre_bone_layers[1], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Upper Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 4])
    ax.imshow(pre_layers[2], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Soft Tissue')
    ax.axis('off')

    # 下方 3x6 图像组
    gs_bottom = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=gs[1])
    # 定义每行的标题
    row_titles = ['Generated', 'Real Sample',  'Real Layers Sample Moved']

    for j in range(4):
        ax = fig.add_subplot(gs_bottom[0, j])
        ax.imshow(pre_bone_masks[j], cmap='gray')
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title(row_titles[0], loc='left', fontsize=12)
            
        ax = fig.add_subplot(gs_bottom[1, j])
        ax.imshow(pre_real_masks[j], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[1], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[2, j])
        ax.imshow(pre_real_masks_2[j], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[2], loc='left', fontsize=12)



    ax = fig.add_subplot(gs_bottom[3, 0])
    ax.imshow(bone_layers_tmp[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Lower Moved')
    ax.axis('off')

    ax = fig.add_subplot(gs_bottom[3, 1])
    ax.imshow(bone_layers_tmp[1], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Upper Moved')
    ax.axis('off')

    ax = fig.add_subplot(gs_bottom[3, 2])
    ax.imshow(recon_image_tmp[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Recon Joint')
    ax.axis('off')

    for j in range(4):
        ax = fig.add_subplot(gs_bottom[4, j])
        ax.imshow(bone_masks_tmp[j], cmap='gray')
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title('moved masks', loc='left', fontsize=12)


    plt.tight_layout()
    plt.show()



def result_image_show_4(image_box):
    image, recon_joint, pre_layers, recon_moved_image, show_overlap, show_nonoverlap, show_joint, show_joint_moved = image_box

    # 创建一个 figure 对象
    fig = plt.figure(figsize=(15, 12))

    # 创建一个 2 行 1 列的 gridspec 布局，上方占 1 行，下方占 1 行
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

    # 上方 5 张图像
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0])

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Original Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 1])
    ax.imshow(pre_layers[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Lower Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 2])
    ax.imshow(pre_layers[1], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Upper Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 3])
    ax.imshow(pre_layers[2], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Soft Tissue')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 4])
    ax.imshow(recon_joint[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Recon Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 5])
    ax.imshow(recon_moved_image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Moved Recon Image')
    ax.axis('off')

    # 下方 3x6 图像组
    gs_bottom = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[1], hspace=0.25)
    # 定义每行的标题
    row_titles = ['Real Overlap', 'Real Nonoverlap', 'Recon Image', 'Recon Image moved']

    for j in range(5):
        ax = fig.add_subplot(gs_bottom[0, j])
        ax.imshow(show_overlap[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title(row_titles[0], loc='left', fontsize=12)
            
        ax = fig.add_subplot(gs_bottom[1, j])
        ax.imshow(show_nonoverlap[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[1], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[2, j])
        ax.imshow(show_joint[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[2], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[3, j])
        ax.imshow(show_joint_moved[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[3], loc='left', fontsize=12)



    plt.tight_layout()
    plt.show()


def result_image_show_4_2(image_box):
    image, recon_joint, pre_layers, show_overlap, show_nonoverlap, show_joint, show_joint_moved = image_box

    # 创建一个 figure 对象
    fig = plt.figure(figsize=(15, 12))

    # 创建一个 2 行 1 列的 gridspec 布局，上方占 1 行，下方占 1 行
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

    # 上方 5 张图像
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0])

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Original Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 1])
    ax.imshow(recon_joint[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Recon Image')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 2])
    ax.imshow(pre_layers[0], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Lower Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 3])
    ax.imshow(pre_layers[1], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Upper Bone')
    ax.axis('off')

    ax = fig.add_subplot(gs_top[0, 4])
    ax.imshow(pre_layers[2], cmap='gray', vmax=1, vmin=0)
    ax.set_title('Soft Tissue')
    ax.axis('off')

    # 下方 3x6 图像组
    gs_bottom = gridspec.GridSpecFromSubplotSpec(4, 7, subplot_spec=gs[1], hspace=0.25)
    # 定义每行的标题
    row_titles = ['Real Overlap', 'Real Nonoverlap', 'Recon Image', 'Recon Image moved']

    for j in range(7):
        ax = fig.add_subplot(gs_bottom[0, j])
        ax.imshow(show_overlap[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:  # 在第一列时添加纵向标题
            ax.set_title(row_titles[0], loc='left', fontsize=12)
            
        ax = fig.add_subplot(gs_bottom[1, j])
        ax.imshow(show_nonoverlap[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[1], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[2, j])
        ax.imshow(show_joint[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[2], loc='left', fontsize=12)

        ax = fig.add_subplot(gs_bottom[3, j])
        ax.imshow(show_joint_moved[j], cmap='gray', vmax=1, vmin=0)
        ax.axis('off')
        if j == 0:
            ax.set_title(row_titles[3], loc='left', fontsize=12)



    plt.tight_layout()
    plt.show()




