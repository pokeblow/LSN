from trainer import BaseTrainModule, Trainer
from model.generator import Generator
from model.discriminator import Discriminator
from model.lossfunction import LayerSegLoss
from model.utils import reconstruction_all, reconstruction_2layer, make_fake_label, result_image_show
from dataset import Data_Loader

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import os

ROOT_PATH = os.getcwd()


class MyTrainModule(BaseTrainModule):
    def __init__(self):
        super().__init__()

        layer = 3
        generator_backbone = 'test'
        discriminator_backbone = 'test'
        self.net_G = Generator(n_layers=layer, backbone=generator_backbone)
        self.net_D = Discriminator(in_channels=1, out_channels=6, backbone=discriminator_backbone)

        self.net_G = nn.DataParallel(self.net_G)
        self.net_D = nn.DataParallel(self.net_D)

        self.net_G.to(device=self.device)
        self.net_D.to(device=self.device)

    def configure_lossfunctions(self):
        self.criterion_LS = LayerSegLoss()
        self.criterion_D = nn.BCELoss()

    def configure_optimizers(self, lr_G, lr_D):
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    def configure_logs(self):
        # learning rate
        self.set_log(name='lr_G', obj=self.optimizer_G, category='lr')
        self.set_log(name='lr_D', obj=self.optimizer_D, category='lr')
        # loss function
        self.set_log(name='Loss_G', obj=self.criterion_LS, category='loss', mode='train')
        self.set_log(name='Loss_D', obj=self.criterion_D, category='loss', mode='train')
        self.set_log(name='Loss_G_val', obj=self.criterion_LS, category='loss', mode='valid')
        self.set_log(name='Loss_D_val', obj=self.criterion_D, category='loss', mode='valid')

    def training_step(self, batch_idx, batch):
        self.net_G.train()
        self.net_D.train()

        image, layer_masks, layer_masks_crop = batch

        # Discriminator training
        self.optimizer_D.zero_grad()

        # real image
        label_real = torch.cat((layer_masks_crop, layer_masks_crop), dim=1)
        # fake image
        label_fake_lower, label_fake_upper, label_fake_soft_tissue = make_fake_label(layer_masks)

        fake_layers = self.net_G(image, layer_masks)

        dis_mask = torch.clone(layer_masks[:, 2, :, :].unsqueeze(1))
        dis_masks = torch.cat((dis_mask, dis_mask, dis_mask, dis_mask, dis_mask, dis_mask), dim=1)

        loss_D_real = self.criterion_D(self.net_D(image), label_real)

        loss_D_fake_lower = self.criterion_D(self.net_D(reconstruction_2layer(fake_layers, 0, 2)) * dis_masks,
                                             label_fake_lower)
        loss_D_fake_upper = self.criterion_D(self.net_D(reconstruction_2layer(fake_layers, 1, 2)) * dis_masks,
                                             label_fake_upper)
        loss_D_fake_soft_tissue = self.criterion_D(self.net_D(fake_layers[:, 2, :, :].unsqueeze(1)) * dis_masks,
                                                   label_fake_soft_tissue)

        loss_D_fake = (loss_D_fake_lower + loss_D_fake_upper + loss_D_fake_soft_tissue) / 3

        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        self.optimizer_D.step()

        # Generator training
        self.optimizer_G.zero_grad()

        pre_layers = self.net_G(image, layer_masks)

        loss_G = self.criterion_LS(org_image=image, org_masks=layer_masks, pre_layers=pre_layers,
                                   discriminator=self.net_D, times=5)

        loss_G.backward()
        self.optimizer_G.step()

        return loss_G, loss_D

    def validation_step(self, batch_idx, batch):
        self.net_G.eval()
        self.net_D.eval()

        image, layer_masks, layer_masks_crop = batch

        # Discriminator testing
        # real image
        label_real = torch.cat((layer_masks_crop, layer_masks_crop), dim=1)
        # fake image
        label_fake_lower, label_fake_upper, label_fake_soft_tissue = make_fake_label(layer_masks)

        fake_layers = self.net_G(image, layer_masks)

        dis_mask = torch.clone(layer_masks[:, 2, :, :].unsqueeze(1))
        dis_masks = torch.cat((dis_mask, dis_mask, dis_mask, dis_mask, dis_mask, dis_mask), dim=1)

        loss_D_real = self.criterion_D(self.net_D(image), label_real)

        loss_D_fake_lower = self.criterion_D(self.net_D(reconstruction_2layer(fake_layers, 0, 2)) * dis_masks, label_fake_lower)
        loss_D_fake_upper = self.criterion_D(self.net_D(reconstruction_2layer(fake_layers, 1, 2)) * dis_masks, label_fake_upper)
        loss_D_fake_soft_tissue = self.criterion_D(self.net_D(fake_layers[:, 2, :, :].unsqueeze(1)) * dis_masks, label_fake_soft_tissue)

        loss_D_fake = (loss_D_fake_lower + loss_D_fake_upper + loss_D_fake_soft_tissue) / 3

        loss_D_val = (loss_D_real + loss_D_fake) / 2

        # Generator testing
        self.optimizer_G.zero_grad()

        pre_layers = self.net_G(image, layer_masks)

        # showing
        pre_masks_lower = self.net_D(reconstruction_2layer(pre_layers, 0, 2)) * dis_masks
        pre_masks_upper = self.net_D(reconstruction_2layer(pre_layers, 1, 2)) * dis_masks
        pre_masks_soft_tissue = self.net_D(pre_layers[:, 2, :, :].unsqueeze(1)) * dis_masks

        recon_image = reconstruction_all(pre_layers) * dis_mask

        loss_G_val = self.criterion_LS(org_image=image, org_masks=layer_masks, pre_layers=pre_layers,
                                       discriminator=self.net_D, times=5)

        image_list = [image, recon_image, pre_layers, pre_masks_lower, pre_masks_upper, pre_masks_soft_tissue]

        return loss_D_val, loss_G_val, image_list

    def configure_saveprocess(self):
        self.set_save_parameter(model=self.net_G, loss_name='Loss_G',
                                save_path='{}/parameter/best_model_generator.pth'.format(ROOT_PATH))
        self.set_save_parameter(model=self.net_D, loss_name='Loss_D',
                                save_path='{}/parameter/best_model_discriminator.pth'.format(ROOT_PATH))

    def show_single_log_image(self, image_box):
        result_image_show(image_box)


if __name__ == "__main__":
    # Data Loading
    image_size = 256
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                    ])

    train_dataset = Data_Loader(ROOT_PATH + '/Data/LS_K1_all_train', transform)
    valid_dataset = Data_Loader(ROOT_PATH + '/Data/LS_K1_all_test', transform)

    my_train_module = MyTrainModule()
    trainer = Trainer(train_module=my_train_module, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.configure(batch_size=2, epochs=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      result_number=3, lr_G=0.001, lr_D=0.001)
    trainer.fit()
