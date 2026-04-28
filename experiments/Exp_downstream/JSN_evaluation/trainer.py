from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
import os
import logging
from torchvision import transforms
import torch
import platform
import re
import numpy as np

ROOT_PATH = os.getcwd()
time = datetime.now()
DATE_TIME = time.strftime('%Y%m%d')
INSTALLATION_INFO = platform.machine() + '_' + platform.system()
LOG_ROOT_PATH = os.getcwd() + '/logs/{}_{}.log'.format(DATE_TIME, INSTALLATION_INFO)


def log_view(log_path=LOG_ROOT_PATH):
    with open(log_path, 'r') as log_file:
        data_dict = {}
        for line in log_file:
            if data_dict == {}:
                pattern = r'(\w+): ([\d.]+)'
                matches = re.findall(pattern, line)
                data_dict = {key: [] for key, value in matches}
            pattern = r'(\w+): ([\d.]+)'
            matches = re.findall(pattern, line)
            for key, value in matches:
                if key == 'Epoch':
                    data_dict[key].append(int(value))
                else:
                    data_dict[key].append(float(value))

    epoch = data_dict['Epoch']
    for key, value in data_dict.items():
        if key == 'Epoch':
            continue
        plt.plot(epoch, value, '.-')
        plt.xticks(epoch, ['{:.0f}'.format(val) for val in epoch])
        plt.title('{} vs Epoch'.format(key))
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.show()


class BaseTrainModule():
    def __init__(self):
        '''
        __init__ defines the networks.
        '''
        self.device = None
        # log_value = {'name': '', 'obj': '', 'category': '', 'value_list': [] }
        self.log_lr_dict = []
        self.log_loss_dict = {'train': [], 'valid': [], 'test': []}
        self.log_image_list = []
        self.model_save_list = []
        self.local_best_loss = []

    def set_log_image(self, image_list):
        if self.device == 'cpu':
            image_list = [tensor.detach().numpy()[0] for tensor in image_list]
        else:
            image_list = [tensor.cpu().detach().numpy()[0] for tensor in image_list]
        self.log_image_list.append(image_list)

    def show_log_image(self):
        for image_box in self.log_image_list:
            self.show_single_log_image(image_box)
        self.log_image_list = []

    def set_save_parameter(self, model, loss_name, save_path):
        model_save_dict = {'model': model, 'loss_name': loss_name, 'save_path': save_path}
        self.model_save_list.append(model_save_dict)

    def save_parameter(self, epoch):
        length = 20
        if self.local_best_loss == []:
            self.local_best_loss = [float('inf') for i in range(len(self.model_save_list))]
        print(self.local_best_loss)
        for idx, model_save_dict in enumerate(self.model_save_list):
            for loss_dict in self.log_loss_dict['train']:
                if model_save_dict['loss_name'] == loss_dict['name']:
                    
                    if loss_dict['best_loss'] > loss_dict['recent_loss']:
                        loss_dict['best_loss'] = loss_dict['recent_loss']
                        net = model_save_dict['model']
                        if isinstance(net, torch.nn.DataParallel):
                            torch.save(net.module.state_dict(), model_save_dict['save_path'])
                        else:
                            torch.save(net.state_dict(), model_save_dict['save_path'])

                    if self.local_best_loss[idx] > loss_dict['recent_loss']:
                        self.local_best_loss[idx] = loss_dict['recent_loss']
                        net = model_save_dict['model']
                        if isinstance(net, torch.nn.DataParallel):
                            torch.save(net.module.state_dict(), model_save_dict['save_path'][:-4]+'_checkpoint_{}.pth'.format(str((epoch // length + 1) * length)))
                        else:
                            torch.save(net.state_dict(), model_save_dict['save_path'][:-4]+'_checkpoint_{}.pth'.format(str((epoch // length + 1) * length)))

                    if epoch % length == 0:
                        self.local_best_loss = [float('inf') for i in range(len(self.model_save_list))]
        print(self.local_best_loss)



    def set_log(self, name, obj, category='', mode=''):
        '''
        set lr or loss as class into dicts
        :param name: name of log variant
        :param obj: object of log variant
        :param category: category of log variant, including 'lr', ' loss' and 'acc'
        :param mode: category of log mode, including 'train', ' valid' and 'test'
        :return: None
        '''
        if category == 'lr':
            log_value = {'name': name, 'obj': obj, 'category': category, 'value_list': []}
            self.log_lr_dict.append(log_value)
        if category == 'loss':
            log_value = {'name': name, 'obj': obj, 'category': category, 'recent_loss': float('inf'),
                         'best_loss': float('inf'), 'value_list': [], 'tmp_list': []}
            self.log_loss_dict[mode].append(log_value)

    def _set_lr(self):
        lr_str_dict = {}
        for i in self.log_lr_dict:
            lr_value = i['obj'].state_dict()['param_groups'][0]['lr']
            i['value_list'].append(lr_value)
            lr_str_dict[i['name']] = lr_value

        return lr_str_dict

    def _set_step_loss(self, loss_list=[], mode=''):
        loss_str_dict = {}
        for loss_idx in range(len(loss_list)):
            loss_value = loss_list[loss_idx].item()
            self.log_loss_dict[mode][loss_idx]['tmp_list'].append(loss_value)
            # print out str
            loss_str_dict[self.log_loss_dict[mode][loss_idx]['name']] = loss_value

        return loss_str_dict

    def _set_loss(self, mode=''):
        loss_str_dict = {}
        for loss_dict in self.log_loss_dict[mode]:
            loss_mean = np.mean(loss_dict['tmp_list'])
            loss_dict['value_list'].append(loss_mean)
            loss_dict['tmp_list'] = []
            loss_dict['recent_loss'] = loss_mean
            # if loss_mean < loss_dict['best_loss']:
            #     loss_dict['best_loss'] = loss_mean
            loss_str_dict[loss_dict['name']] = loss_mean

        return loss_str_dict

    def set_device(self, device):
        self.device = device

    def training_step(self, batch_idx, batch):
        '''
        training_step defines the train loop and backward pipline.
        :param batch: batch data in each loop.
        :param batch: batch index in each epoch.
        :return: value of loss function
        '''

    def validation_step(self, batch_idx, batch):
        '''
        training_step defines the train loop and backward pipline.
        :param batch: batch data in each loop.
        :param batch: batch index in each epoch.
        :return: value of loss function
        '''

    def configure_parameters(self):
        '''
        configure_parameters defines load in the pre-train parameters.
        :return: None
        '''

    def configure_optimizers(self):
        '''
        configure_optimizers defines the optimizers.
        :return: optimizer
        '''

    def configure_scheduler(self):
        '''
        configure_scheduler defines the scheduler.
        :return: scheduler
        '''

    def configure_lossfunctions(self):
        '''
        configure_lossfunctions defines the criterion of loss functions.
        :return: loss
        '''

    def configure_logs(self):
        '''
        configure_logs defines the logs for loss, acc and lr carves.
        :return:
        '''

    def configure_saveprocess(self):
        '''
        configure_saveprocess defines the models saving process
        use self.set_save_parameter(model, loss_name, save_path, mode='best')
        :return:
        '''

    def show_single_log_image(self, image_box):
        '''
        show the log image with plt
        :param image_box: image box with one set of result image
        :return: None
        '''


class Trainer():
    def __init__(self, valid_available=True, train_module=BaseTrainModule, train_dataset=None, valid_dataset=None):
        self.train_module = train_module
        self.train_loader = None
        self.valid_loader = None
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.logs = logs

        self.epochs = 1
        self.batch_size = 1
        self.valid_available = valid_available
        self.result_number = 0

        log_path = LOG_ROOT_PATH
        with open(log_path, 'w'):
            pass

        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger('Logger')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def __set_Dataset(self, train_dataset=None, valid_dataset=None, sampler='off'):
        if valid_dataset == None:
            self.valid_available = False

        if sampler == 'off':
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            drop_last=True
                                                            )
            if self.valid_available:
                self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                drop_last=True
                                                                )

    def configure(self, epochs=10, batch_size=1, device='', result_number=2, **kwargs):
        self.optimizers = self.train_module.configure_optimizers(**kwargs)
        self.schedulers = self.train_module.configure_scheduler(self.optimizers)
        self.train_module.configure_lossfunctions()
        self.train_module.configure_logs()
        self.train_module.configure_parameters()
        self.train_module.configure_saveprocess()

        self.train_module.set_device(device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.result_number = result_number
        self.__set_Dataset(self.train_dataset, self.valid_dataset)

    def __log_step_loss(self, epoch, batch_idx, steps, loss_str_dict):
        log_message = f"Epoch: {epoch + 1}, Step: {batch_idx + 1} / {steps}, " \
                      f"Losses: {', '.join([f'{key}: {value:.6f}' for key, value in loss_str_dict.items()])}"
        print(log_message)

    def __log_epoch_summary(self, epoch, epochs, lr_str_dict, loss_str_dict_train, loss_str_dict_valid):
        print('-' * 42 + ' Epoch Summary ' + '-' * 43)
        log_message_title = f"Epoch: {epoch + 1} / {epochs}"
        log_message_train = f"{', '.join([f'{key}: {value:.6f}' for key, value in loss_str_dict_train.items()])}"
        print(log_message_title + '\n' + 'Train Loss: ' + log_message_train)
        log_message_valid = ''
        if self.valid_available:
            log_message_valid = f"{', '.join([f'{key}: {value:.6f}' for key, value in loss_str_dict_valid.items()])}"
            print('Valid Loss: ' + log_message_valid)

        log_message_lr = f"{', '.join([f'{key}: {value:.6f}' for key, value in lr_str_dict.items()])}"
        log_message = f"Epoch: {epoch + 1}" + ', ' + log_message_lr + ', ' + log_message_train + ', ' + log_message_valid

        self.logger.info(log_message)

    def fit(self):
        for epoch in range(self.epochs):
            print('=' * 100)
            print('Epoch: {} / {}'.format(epoch + 1, self.epochs))
            print('-' * 44 + ' Train Mode ' + '-' * 44)

            str_lr_dict = self.train_module._set_lr()
            for batch_idx, batch in enumerate(self.train_loader):
                batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                loss = self.train_module.training_step(batch_idx, batch)
                if not isinstance(loss, tuple):
                    loss = (loss,)
                log_loss_str_dict = self.train_module._set_step_loss(loss_list=loss, mode='train')
                self.__log_step_loss(epoch, batch_idx, len(self.train_loader), log_loss_str_dict)
            str_dict_train = self.train_module._set_loss(mode='train')

            if self.valid_available:
                print('-' * 44 + ' Valid Mode ' + '-' * 44)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.valid_loader):
                        batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                        loss_batch = self.train_module.validation_step(batch_idx, batch)
                        loss = loss_batch[:-1]
                        image_list = loss_batch[-1]

                        if not isinstance(loss, tuple):
                            loss = (loss,)
                        log_loss_str_dict = self.train_module._set_step_loss(loss_list=loss, mode='valid')

                        pacemaker = len(self.valid_loader) // self.result_number
                        if batch_idx % pacemaker == 0:
                            self.train_module.set_log_image(image_list)

                        self.__log_step_loss(epoch, batch_idx, len(self.valid_loader), log_loss_str_dict)
                    str_dict_valid = self.train_module._set_loss(mode='valid')

            self.__log_epoch_summary(epoch, self.epochs, str_lr_dict, str_dict_train, str_dict_valid)
            self.train_module.show_log_image()
            self.train_module.save_parameter(epoch)

            if not isinstance(self.schedulers, tuple):
                self.schedulers = (self.schedulers,)

            for item in self.schedulers:
                item.step()

    def fit_valid(self):
        print('-' * 44 + ' Valid Mode ' + '-' * 44)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                loss_batch = self.train_module.validation_step(batch_idx, batch)
                loss = loss_batch[:-1]
                image_list = loss_batch[-1]

                if not isinstance(loss, tuple):
                    loss = (loss,)
                log_loss_str_dict = self.train_module._set_step_loss(loss_list=loss, mode='valid')

                pacemaker = len(self.valid_loader) // self.result_number
                if batch_idx % pacemaker == 0:
                    self.train_module.set_log_image(image_list)

        self.train_module.show_log_image()

if __name__ == "__main__":
    log_view()
