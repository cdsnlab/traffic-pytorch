import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import *
from data.datasets import DCRNNDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger


class DCRNNTrainer(BaseTrainer):
    def __init__(self, cls, config):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.logger = Logger()

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = DCRNNDataset(datasets[category])

        num_train_sample = len(datasets['train'])
        num_val_sample = len(datasets['val'])

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
            label = target[..., :self.config.output_dim].to(self.device)  
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.num_train_iteration_per_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.config.cl_decay_steps)

            output = self.model(data, target, teacher_forcing_ratio)

            output = output * self.std 
            output = output + self.mean

            output = torch.transpose(output.view(self.config.num_pred, self.config.batch_size, self.config.num_nodes, 
                            self.config.output_dim), 0, 1)  # back to (50, 12, 207, 1)

            loss = self.loss(output, label) 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            label = label.detach().cpu()

            this_metrics = self._eval_metrics(output, label)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        if epoch % self.config.valid_every_epoch == 0:
            avg_loss = total_loss / len(self.train_loader)
            avg_metrics = total_metrics / len(self.train_loader)
            self.logger.log_training(avg_loss, avg_metrics, epoch) 
            self.validate_epoch(epoch)


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