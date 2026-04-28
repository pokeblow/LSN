from trainer import BaseTrainModule, Trainer
from model.generator import Generator
from model.discriminator import Discriminator
from model.segmenter import Segmenter
from model.lossfunction import LayerSegLoss_Main, MaskedBCELoss, MaskedCELoss, MaskedMSELoss
from model.utils import *
from dataset import Data_Loader_All

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import os

import time

ROOT_PATH = os.getcwd()


class MyTrainModule(BaseTrainModule):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        layer = 3
        self.generator_backbone = 'transunet'
        self.segmenter_backbone = 'unet'
        self.discriminator_backbone = 'unet'
        self.net_G = Generator(n_layers=layer, backbone=self.generator_backbone)
        self.net_S = Segmenter(in_channels=1, out_channels=3, backbone=self.segmenter_backbone)
        self.net_D = Discriminator(in_channels=1, out_channels=1, backbone=self.discriminator_backbone)

        self.net_G = nn.DataParallel(self.net_G)
        self.net_S = nn.DataParallel(self.net_S)
        self.net_D = nn.DataParallel(self.net_D)

        self.net_G.to(device=self.device)
        self.net_D.to(device=self.device)
        self.net_S.to(device=self.device)

        self.save_name = 'Abl_256_(1_4)_3_4'
        # Abl_256_(1_4)_3_3: G + r
        # Abl_256_(1_4)_3_0: G + S + r + t
        # Abl_256_(1_4)_3_1: G + S + D + r + t
        # Proposed:          G + S + D + p + r + t
        # Abl_256_(1_4)_3_4: G + S + D + p + r

        # Abl_256_(1_4)_3_2: wo dis (LSN + Pre) G +
        # pre_name = '256_(1_4)_2_4'
        # pre_name = 'Proposed_2'
        pre_name = '256_(1_4)_2_5'
        checkpoint = '_checkpoint_180'

        pretrain_path = ROOT_PATH + '/parameter/1_4/generator_Pre_{}_{}_{}_{}{}.pth'.format(pre_name,
                                                                                            self.generator_backbone,
                                                                                            self.segmenter_backbone,
                                                                                            self.discriminator_backbone,
                                                                                            checkpoint)
        state_dict = torch.load(pretrain_path)
        self.net_G.module.load_state_dict(state_dict)

        pretrain_path = ROOT_PATH + '/parameter/1_4/segmenter_Pre_{}_{}_{}_{}{}.pth'.format(pre_name,
                                                                                            self.generator_backbone,
                                                                                            self.segmenter_backbone,
                                                                                            self.discriminator_backbone,
                                                                                            checkpoint)
        state_dict = torch.load(pretrain_path)
        self.net_S.module.load_state_dict(state_dict)

        # pretrain_path = ROOT_PATH + '/parameter/1_4/discriminator_Pre_{}_{}_{}_{}{}.pth'.format(pre_name, self.generator_backbone, self.segmenter_backbone, self.discriminator_backbone, checkpoint)
        # state_dict = torch.load(pretrain_path)
        # self.net_D.module.load_state_dict(state_dict)

    def configure_lossfunctions(self):
        self.criterion_G = LayerSegLoss_Main()
        self.criterion_S = MaskedBCELoss()
        self.criterion_D = MaskedBCELoss()

    def configure_optimizers(self, lr_G, lr_S, lr_D):
        self.optimizer_G = torch.optim.AdamW(self.net_G.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=1e-2,
                                             amsgrad=True)
        self.optimizer_S = torch.optim.AdamW(self.net_S.parameters(), lr=lr_S, betas=(0.5, 0.999), weight_decay=1e-2,
                                             amsgrad=True)
        self.optimizer_D = torch.optim.AdamW(self.net_D.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=1e-2,
                                             amsgrad=True)

        return self.optimizer_G, self.optimizer_S, self.optimizer_D

    def configure_scheduler(self, optimizers):
        optimizer_G, optimizer_S, optimizer_D = optimizers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G, step_size=100, gamma=0.5)
        self.scheduler_S = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_S, step_size=100, gamma=0.5)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_D, step_size=100, gamma=0.5)

        return self.scheduler_G, self.scheduler_S, self.scheduler_D

    def configure_logs(self):
        # learning rate
        self.set_log(name='lr_G', obj=self.optimizer_G, category='lr')
        self.set_log(name='lr_S', obj=self.optimizer_S, category='lr')
        self.set_log(name='lr_D', obj=self.optimizer_D, category='lr')
        # loss function
        self.set_log(name='Loss_G', obj=self.criterion_G, category='loss', mode='train')
        self.set_log(name='Loss_S', obj=self.criterion_S, category='loss', mode='train')
        self.set_log(name='Loss_D', obj=self.criterion_D, category='loss', mode='train')
        self.set_log(name='Loss_G_val', obj=self.criterion_G, category='loss', mode='valid')
        self.set_log(name='Loss_S_val', obj=self.criterion_S, category='loss', mode='valid')
        self.set_log(name='Loss_D_val', obj=self.criterion_D, category='loss', mode='valid')

    def get_fake_D_loss(self, layers, masks, labels, moved=False, eval=False):
        soft_tissue_mask = masks[:, 2:, :, :]
        context_mask = torch.clone(masks[:, 2:, :, :])

        if moved:
            lower_layer_tmp, lower_mask_tmp = random_movement(layers[:, 0, :, :].unsqueeze(1),
                                                              masks[:, 0, :, :].unsqueeze(1), soft_tissue_mask)
            upper_layer_tmp, upper_mask_tmp = random_movement(layers[:, 1, :, :].unsqueeze(1),
                                                              masks[:, 1, :, :].unsqueeze(1), soft_tissue_mask)

            fake_layers_tmp = torch.cat((lower_layer_tmp, upper_layer_tmp, layers[:, 2:, :, :]), dim=1)
        else:
            fake_layers_tmp = layers

        recon_image = reconstruction_all(fake_layers_tmp)

        # pre_labels = self.net_D(recon_image)

        # loss_fake_image = self.criterion_D(pre_labels, labels, context_mask)
        loss_fake_image = 0

        if eval:
            pre_masks = self.net_S(recon_image)
            show_image_moved = torch.cat((recon_image, F.sigmoid(pre_masks), F.sigmoid(labels)), dim=1)

            return loss_fake_image, show_image_moved
        else:
            return loss_fake_image

    def training_step(self, batch_idx, batch):
        self.net_G.train()
        self.net_S.train()
        self.net_D.train()

        image, layer_masks, layer_masks_crop = batch

        context_mask = torch.clone(layer_masks[:, 2:, :, :])
        context_masks = context_mask.repeat(1, 3, 1, 1)

        # Segmenter training
        self.optimizer_S.zero_grad()

        pre_mask_overlap = self.net_S(image)

        loss_S = self.criterion_S(pre_mask_overlap, layer_masks_crop, context_masks)

        loss_S.backward()
        self.optimizer_S.step()

        # Generator training
        self.optimizer_G.zero_grad()

        pre_layers = self.net_G(image, layer_masks)

        loss_G = self.criterion_G(org_image=image, org_masks=layer_masks, pre_layers=pre_layers, segmenter=self.net_S,
                                  discriminator=self.net_D, times=2)

        loss_G.backward()
        self.optimizer_G.step()

        # Discriminator training
        self.optimizer_D.zero_grad()

        soft_tissue_layer = pre_layers.detach()[:, 2:, :, :]
        artifactual_region = 1 - torch.clone(layer_masks_crop)[:, 2:, :, :]

        per_AR_mask = self.net_D(soft_tissue_layer)
        loss_D = self.criterion_D(per_AR_mask, artifactual_region, context_mask)

        loss_D.backward()
        self.optimizer_D.step()

        return loss_G, loss_S, loss_D

    def validation_step(self, batch_idx, batch):
        self.net_G.eval()
        self.net_D.eval()
        self.net_S.eval()

        image, layer_masks, layer_masks_crop = batch

        context_mask = torch.clone(layer_masks[:, 2:, :, :])
        context_masks = context_mask.repeat(1, 3, 1, 1)

        # Segmenter training
        pre_mask_overlap = self.net_S(image)

        loss_S_val = self.criterion_S(pre_mask_overlap, layer_masks_crop, context_masks)
        # Discriminator training
        real_labels = torch.ones_like(image)
        fake_labels = torch.zeros_like(image)

        # # fake sample
        starttime = time.time()
        pre_layers = self.net_G(image, layer_masks).detach()
        endtime = time.time()
        print(endtime - starttime)

        _, show_joint = self.get_fake_D_loss(pre_layers, layer_masks, fake_labels, moved=False, eval=True)
        _, show_joint_moved = self.get_fake_D_loss(pre_layers, layer_masks, fake_labels, moved=True, eval=True)

        # Generator testing
        loss_G_val = self.criterion_G(org_image=image, org_masks=layer_masks, pre_layers=pre_layers,
                                      segmenter=self.net_S, discriminator=self.net_D, times=4)

        # Discriminator testing
        soft_tissue_layer = pre_layers.detach()[:, 2:, :, :]
        artifactual_region = 1 - torch.clone(layer_masks_crop)[:, 2:, :, :]

        per_AR_mask = self.net_D(soft_tissue_layer)
        loss_D_val = self.criterion_D(per_AR_mask, artifactual_region, context_mask)

        # Showing
        recon_joint = reconstruction_all(pre_layers)

        show_overlap = torch.cat((image, F.sigmoid(pre_mask_overlap), F.sigmoid(per_AR_mask)), dim=1)

        lower_layer_tmp = random_move_and_reconstruct(pre_layers[:, 0, :, :].unsqueeze(1), context_mask)
        upper_layer_tmp = random_move_and_reconstruct(pre_layers[:, 1, :, :].unsqueeze(1), context_mask)
        moved_layers = torch.cat((lower_layer_tmp, upper_layer_tmp, pre_layers[:, 2:, :, :]), dim=1)

        recon_moved_image = reconstruction_all(moved_layers)

        image_list = [image, recon_joint, pre_layers, recon_moved_image,
                      show_overlap, show_overlap,
                      show_joint, show_joint_moved
                      ]

        return loss_G_val, loss_S_val, loss_D_val, image_list

    def configure_saveprocess(self):
        self.set_save_parameter(model=self.net_G, loss_name='Loss_G',
                                save_path='{}/parameter/1_4/generator_Main_{}_{}_{}_{}.pth'.format(ROOT_PATH,
                                                                                                   self.save_name,
                                                                                                   self.generator_backbone,
                                                                                                   self.segmenter_backbone,
                                                                                                   self.discriminator_backbone))
        self.set_save_parameter(model=self.net_S, loss_name='Loss_S',
                                save_path='{}/parameter/1_4/segmenter_Main_{}_{}_{}_{}.pth'.format(ROOT_PATH,
                                                                                                   self.save_name,
                                                                                                   self.generator_backbone,
                                                                                                   self.segmenter_backbone,
                                                                                                   self.discriminator_backbone))
        self.set_save_parameter(model=self.net_D, loss_name='Loss_D',
                                save_path='{}/parameter/1_4/discriminator_Main_{}_{}_{}_{}.pth'.format(ROOT_PATH,
                                                                                                       self.save_name,
                                                                                                       self.generator_backbone,
                                                                                                       self.segmenter_backbone,
                                                                                                       self.discriminator_backbone))

    def show_single_log_image(self, image_box):
        result_image_show_4(image_box)


if __name__ == "__main__":
    # Data Loading
    image_size = 256
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                    ])

    train_dataset = Data_Loader_All(ROOT_PATH + '/Data/LS_K1_all_train', transform)
    valid_dataset = Data_Loader_All(ROOT_PATH + '/Data/LS_K1_all_test', transform)

    my_train_module = MyTrainModule()
    trainer = Trainer(train_module=my_train_module, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.configure(batch_size=9, epochs=100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      result_number=2, lr_G=1e-4, lr_S=1e-5, lr_D=4e-4)
    trainer.fit_valid()
