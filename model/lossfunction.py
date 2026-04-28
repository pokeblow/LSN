import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils import *

import numpy as np
import os
import matplotlib.pyplot as plt

import cv2

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.criterion_BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion_CE = nn.CrossEntropyLoss(reduction='none')
        self.criterion_Dice = DiceLoss()
    
    def forward(self, pre, gt, mask):
        # loss_ce = self.criterion_CE(pre, gt)
        loss_bce = self.criterion_BCE(pre, gt)
        # masked_loss_ce = loss_ce * mask[:, 2:, :, :]

        masked_loss_bce = loss_bce * mask

        # loss_dice = self.criterion_Dice(F.sigmoid(pre), gt, mask)

        # loss = ((masked_loss_bce.sum() / mask.sum()) + loss_dice) / 2
        loss = masked_loss_bce.sum() / mask.sum()

        return loss
    
class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.criterion_CE = nn.CrossEntropyLoss(reduction='none')
        self.criterion_Dice = DiceLoss()
    
    def forward(self, pre, gt, mask):
        loss_ce = self.criterion_CE(pre, gt)
        masked_loss_ce = loss_ce * mask[:, 2:, :, :] / 10

        # loss_dice = self.criterion_Dice(F.sigmoid(pre), gt, mask)

        # loss = ((masked_loss_bce.sum() / mask.sum()) + loss_dice) / 2
        loss = masked_loss_ce.sum() / mask[:, 2:, :, :].sum()

        return loss
    
    # def forward(self, pre, gt, mask):
    #     loss_bce = self.criterion_BCE(pre, gt)
    #     masked_loss_bce = loss_bce * mask

    #     loss = masked_loss_bce.sum() / mask.sum()
    #     return loss
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pre, gt, mask):
        loss = self.criterion(pre, gt)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, mask):
        smooth = 1
        num_classes = pred.size(1)
        size = pred.size(0)

        dice_loss = 0

        for i in range(num_classes):
            pred_ = pred[:, i].view(size, -1)
            target_ = target[:, i].view(size, -1)
            mask_ =  mask[:, i].view(size, -1)

            pred_ = pred_ * mask_
            target_ = target_ * mask_

            intersection = pred_ * target_
            dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
            dice_loss += 1 - dice_score.sum() / size

        return dice_loss / num_classes


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets, mask):
        bs = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.contiguous().view(bs, -1)
        m2 = targets.view(bs, -1)
        mask = mask.view(bs, -1)
        
        m1_masked = m1 * mask
        m2_masked = m2 * mask

        intersection = (m1_masked * m2_masked)

        score = 2. * (intersection.sum(1) + smooth) / (m1_masked.sum(1) + m2_masked.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score
    


class SegDisloss(nn.Module):
    def __init__(self):
        super(SegDisloss, self).__init__()
        self.S_criterion = MaskedBCELoss()
        self.D_criterion = MaskedBCELoss()

    def forward(self, pre_masks, pre_label, label, masks, context_masks):
        loss_seg = self.S_criterion(pre_masks, masks, context_masks)
        loss_dis = self.D_criterion(pre_label, label, context_masks)

        return 0.5 * loss_seg + 0.5 * loss_dis




class LayerSegLoss(nn.Module):
    def __init__(self):
        super(LayerSegLoss, self).__init__()
        self.G_criterion = MaskedMSELoss()
        self.S_criterion = MaskedBCELoss()
        self.D_criterion = MaskedBCELoss()

    def segdis(self, masks, pre_layers, segmenter, discriminator):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        real_labels = torch.ones_like(masks[:, 2:, :, :])

        context_mask = torch.clone(masks[:, 2:, :, :])
        context_masks = context_mask.repeat(1, 3, 1, 1)

        label_cropped = make_moved_label(masks)

        def get_loss(layer, label, mask):
            pre_mask = segmenter(layer)
            # pre_label = discriminator(layer)

            # loss = 0.5 * self.S_criterion(pre_mask, mask, context_masks) + 0.5 * self.D_criterion(pre_label, label, context_mask)
            loss = self.S_criterion(pre_mask, mask, context_masks)

            return loss

        D_loss = get_loss(reconstruction_all(pre_layers), real_labels, label_cropped)

        return D_loss

    

class LayerSegLoss_Pre(LayerSegLoss):
    def __init__(self):
        super().__init__()

    def forward(self, org_image, org_masks, pre_layers, bone_layers_GT, segmenter, discriminator, times=3):
        batch_size, _, _, _ = org_image.shape
        '''
        :param org_image: original image from dataset
        :param org_masks: original masks from dataset
        :param pre_layers: layer images from Generator
        :param times: moved times
        :param discriminator: discriminator
        :return: loss
        '''

        context_mask = org_masks[:, 2, :, :].unsqueeze(1)
        context_masks = context_mask.repeat(1, 6, 1, 1)
        overlap_mask = torch.sum(torch.clone(org_masks[:, :2, :, :]), dim=1).unsqueeze(1) / 2
        overlap_mask[overlap_mask != 1] = 0
        overlap_masks = overlap_mask.repeat(1, 6, 1, 1)

        # Reconstruction Loss
        org_image = org_image * context_mask
        recon_image = reconstruction_all(pre_layers) * context_mask

        Recon_loss = torch.sqrt(self.G_criterion(recon_image, org_image, context_mask)) + 0.2 * torch.sqrt(self.G_criterion(recon_image, org_image, overlap_mask)) 
        # Recon_loss = torch.sqrt(self.GS_criterion(recon_image, org_image, org_mask_all))

        # Bone Layer Loss
        bone_masks = org_masks[:, :2, :, :]
        overlap_mask_x2 = torch.cat((overlap_mask, overlap_mask), dim=1)

        pre_bone_lower = reconstruction_2layer(pre_layers, 0, 2)
        pre_bone_upper = reconstruction_2layer(pre_layers, 1, 2)
        pre_bone_layer = torch.cat((pre_bone_lower, pre_bone_upper), dim=1) * bone_masks
        
        Bone_loss = torch.sqrt(self.G_criterion(pre_bone_layer, bone_layers_GT, bone_masks)) + 0.2 * torch.sqrt(self.G_criterion(pre_bone_layer, bone_layers_GT, overlap_mask_x2))
        # Bone_loss = torch.sqrt(self.GS_criterion(pre_bone_layer, bone_layers_GT, bone_masks))

        # Discriminator Value Loss (Non-moved)
        S_loss_non_moved = self.segdis(org_masks, pre_layers, segmenter, discriminator)

        # Discriminator Value Loss (Moved)
        lower_mask = org_masks[:, 0, :, :].unsqueeze(1)
        upper_mask = org_masks[:, 1, :, :].unsqueeze(1)
        soft_tissue_mask = org_masks[:, 2, :, :].unsqueeze(1)

        lower_layer = pre_layers[:, 0, :, :].unsqueeze(1)
        upper_layer = pre_layers[:, 1, :, :].unsqueeze(1)

        lower_layer_GT = bone_layers_GT[:, 0, :, :].unsqueeze(1)
        upper_layer_GT = bone_layers_GT[:, 1, :, :].unsqueeze(1)

        soft_tissue_layer = pre_layers[:, 2, :, :].unsqueeze(1)

        S_loss_moved = torch.zeros_like(S_loss_non_moved)
        Bone_loss_moved = torch.zeros_like(Bone_loss)

        for i in range(times):
            lower_layer_tmp, lower_layer_GT_tmp, lower_mask_tmp = random_movement_with_GT(lower_layer, lower_layer_GT, lower_mask, soft_tissue_mask)
            upper_layer_tmp, upper_layer_GT_tmp, upper_mask_tmp = random_movement_with_GT(upper_layer, upper_layer_GT, upper_mask, soft_tissue_mask)
                
            layers_tmp = torch.cat((lower_layer_tmp, upper_layer_tmp, soft_tissue_layer), dim=1)
            masks_tmp = torch.cat((lower_mask_tmp, upper_mask_tmp, soft_tissue_mask), dim=1)

            S_loss_moved += self.segdis(masks_tmp, layers_tmp, segmenter, discriminator)

            bone_masks_tmp = masks_tmp[:, :2, :, :]
            pre_bone_lower_tmp = reconstruction_2layer(layers_tmp, 0, 2)
            pre_bone_upper_tmp = reconstruction_2layer(layers_tmp, 1, 2)
            pre_bone_layer_tmp = torch.cat((pre_bone_lower_tmp, pre_bone_upper_tmp), dim=1)
            bone_layers_GT_tmp = torch.cat((lower_layer_GT_tmp, upper_layer_GT_tmp), dim=1)

            Bone_loss_moved += torch.sqrt(self.G_criterion(pre_bone_layer_tmp, bone_layers_GT_tmp, bone_masks_tmp))

        S_loss_moved = S_loss_moved / times
        Bone_loss_moved = Bone_loss_moved / times

        Bone_loss_moved = Bone_loss # without transformed

        Bone_loss = 0.5 * Bone_loss + 0.5 * Bone_loss_moved

        # Artifactual Region Discriminator training
        layer_masks_crop = make_moved_label(torch.clone(org_masks))
        artifactual_region = 1-layer_masks_crop[:, 2:, :, :]

        # artifactual_region = torch.zeros_like(layer_masks_crop[:, 2:, :, :])

        per_AR_mask = discriminator(soft_tissue_layer)
        D_loss = 1 - self.D_criterion(per_AR_mask, artifactual_region, context_mask)

        print('\t', 
            'Recon_loss:', format(Recon_loss.item(), '.6f'), 
            'Bone_loss (Non-moved):', format(Bone_loss.item(), '.6f'), 
            'Bone_loss (Moved):', format(Bone_loss_moved.item(), '.6f'), 
            'S_loss (Moved):', format(S_loss_moved.item(), '.6f'),
            'D_loss (AR):', format(D_loss.item(), '.6f'))

        loss = 0.5 * Recon_loss + 0.2 * S_loss_moved + 0.2 * Bone_loss + 0.1 * D_loss
        # loss = Recon_loss + 0.4 * S_loss_moved + 0.4 * Bone_loss

        return loss



class LayerSegLoss_Main(LayerSegLoss):
    def __init__(self):
        super().__init__()

    def forward(self, org_image, org_masks, pre_layers, segmenter, discriminator, times=3):
        batch_size, _, _, _ = org_image.shape
        '''
        :param org_image: original image from dataset
        :param org_masks: original masks from dataset
        :param pre_layers: layer images from Generator
        :param times: moved times
        :param discriminator: discriminator
        :return: loss
        '''

        context_mask = org_masks[:, 2, :, :].unsqueeze(1)
        context_masks = context_mask.repeat(1, 6, 1, 1)
        overlap_mask = torch.sum(torch.clone(org_masks[:, :2, :, :]), dim=1).unsqueeze(1) / 2
        overlap_mask[overlap_mask != 1] = 0
        overlap_masks = overlap_mask.repeat(1, 6, 1, 1)

        # Reconstruction Loss
        org_image = org_image * context_mask
        recon_image = reconstruction_all(pre_layers) * context_mask

        Recon_loss = torch.sqrt(self.G_criterion(recon_image, org_image, context_mask)) + 0.2 * torch.sqrt(self.G_criterion(recon_image, org_image, overlap_mask)) 
        # Recon_loss = torch.sqrt(self.GS_criterion(recon_image, org_image, org_mask_all))

        # Discriminator Value Loss (Non-moved)
        S_loss_non_moved = self.segdis(org_masks, pre_layers, segmenter, discriminator)

        # Discriminator Value Loss (Moved)
        lower_mask = org_masks[:, 0, :, :].unsqueeze(1)
        upper_mask = org_masks[:, 1, :, :].unsqueeze(1)
        soft_tissue_mask = org_masks[:, 2, :, :].unsqueeze(1)

        lower_layer = pre_layers[:, 0, :, :].unsqueeze(1)
        upper_layer = pre_layers[:, 1, :, :].unsqueeze(1)

        soft_tissue_layer = pre_layers[:, 2, :, :].unsqueeze(1)

        S_loss_moved = torch.zeros_like(S_loss_non_moved)

        for i in range(times):
            lower_layer_tmp, lower_mask_tmp = random_movement(lower_layer, lower_mask, soft_tissue_mask)
            upper_layer_tmp, upper_mask_tmp = random_movement(upper_layer, upper_mask, soft_tissue_mask)
                
            layers_tmp = torch.cat((lower_layer_tmp, upper_layer_tmp, soft_tissue_layer), dim=1)
            masks_tmp = torch.cat((lower_mask_tmp, upper_mask_tmp, soft_tissue_mask), dim=1)

            S_loss_moved += self.segdis(masks_tmp, layers_tmp, segmenter, discriminator)

        S_loss_moved = S_loss_moved / times

        # Artifactual Region Discriminator training
        layer_masks_crop = make_moved_label(torch.clone(org_masks))
        artifactual_region = 1-layer_masks_crop[:, 2:, :, :]

        # artifactual_region = torch.zeros_like(layer_masks_crop[:, 2:, :, :])

        per_AR_mask = discriminator(soft_tissue_layer)
        D_loss = 1 - self.D_criterion(per_AR_mask, artifactual_region, context_mask)

        print('\t', 'Recon_loss:', format(Recon_loss.item(), '.6f'),
            'S_loss (Moved):', format(S_loss_moved.item(), '.6f'),
            'D_loss (AR):', format(D_loss.item(), '.6f'))

        loss = 0.6 * Recon_loss + 0.3 * S_loss_moved + 0.1 * D_loss
        # loss = 0.6 * Recon_loss
        # loss = 0.5 * Recon_loss + 0.5 * S_loss_moved

        return loss


if __name__ == "__main__":
    batch_size = 2
    height, width = 256, 256

    org_image = torch.rand(batch_size, 1, height, width)

    org_masks = torch.zeros(batch_size, 3, height, width)
    org_masks[:, 0, 150:256, 68:256-68] = 1  # 下层掩码
    org_masks[:, 1, 0:120, 68:256-68] = 1  # 上层掩码
    org_masks[:, 2, 0:256, 10:246] = 1  # 软组织掩码

    pre_layers = torch.rand(batch_size, 3, height, width)

    class SimpleDiscriminator(nn.Module):
        def __init__(self):
            super(SimpleDiscriminator, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = F.sigmoid(self.conv1(x))
            return x


    # 实例化判别器
    discriminator = SimpleDiscriminator()

    # 实例化 LayerSegLoss_2 并计算损失
    loss_fn = LayerSegLoss()
    loss = loss_fn(org_image, org_masks, pre_layers, discriminator)

    # 打印损失值
    print("Loss:", loss.item())
