import os
import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import generate_train_val_test
from data.datasets import DRCNNDataset, StandardScaler
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 

class GMANTrainer(BaseTrainer):
    def __init__(self, model, config):
        self.config = config
        self.device = self.config.device
        self.setup_model(model)

    def setup_model(self, model):
        self.model = model(self.config).to(self.device)

    def compose_dataset(self):
        if os.path.exists("{}/{}_train_{}.npz".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio)) and \
            os.path.exists("{}/{}_test_{}.npz".format(self.config.dataset_dir, self.config.dataset_name, self.config.test_ratio)):
                print(toGreen('Found generated dataset in '+self.config.dataset_dir))
        else:    
            print(toGreen('Generating dataset...'))
            generate_train_val_test(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)

        datasets = {}
        for category, ratio in zip(['train', 'val', 'test'], ['_{}'.format(self.config.train_ratio), '', '_{}'.format(self.config.test_ratio)]):
            cat_data = np.load("{}/{}_{}{}.npz".format(self.config.dataset_dir, self.config.dataset_name, category, ratio))
            datasets[category] = DRCNNDataset(cat_data['x'], cat_data['y'])

        x_train = np.load("{}/{}_train_{}.npz".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio))['x']
        self.scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
        
        num_train_sample = len(datasets['train'])
        num_val_sample = len(datasets['val'])

        # get number of iterations per epoch for progress bar
        self.num_train_iteration_per_epoch = math.ceil(num_train_sample / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(num_val_sample / self.config.test_batch_size)

        return datasets['train'], datasets['val'], datasets['test']

    def compose_loader(self):
        train_dataset, val_dataset, test_dataset = self.compose_dataset()
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}] Val [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset)), toGreen(len(val_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.train_loader):
            label = target[..., :self.config.output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.num_train_iteration_per_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.config.cl_decay_steps)

            output = self.model(data, target, teacher_forcing_ratio)
            output = torch.transpose(output.view(12, self.config.batch_size, self.config.num_nodes, 
                            self.config.output_dim), 0, 1)  # back to (50, 12, 207, 1)

            loss = self.loss(output.cpu(), label)  # loss is self-defined, need cpu input
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())

            print_train(epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, total_metrics)

            # TODO: logging           

    def validate_epoch(self, epoch):
        pass 

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    
    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics