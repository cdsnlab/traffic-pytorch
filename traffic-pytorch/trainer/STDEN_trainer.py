import time
import math
import torch 
import numpy as np 
import os
from data.datasets import STDENDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class STDENTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)

    def setup_model(self):
        self.model = self.cls(self.config, self.adj).to(self.device)

    def compose_dataset(self):
        datasets, _ = self.load_dataset()
        self.adj = torch.tensor(np.load(self.config.adj_file)).to(self.config.device)
        self.num_edges = (self.adj > 0.).sum()

        for category in ['train', 'val', 'test']:
            datasets[category] = STDENDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
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
            
            data = self._prepare_data(data)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, fe = self.model(data)

            output = output * self.std 
            output = output + self.mean

            output, target = self._unprepare_data(output), self._unprepare_data(target)

            loss = self.loss(output, target) 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch, is_test):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data = self._prepare_data(data)
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output, fe = self.model(data)

            output = output * self.std 
            output = output + self.mean

            output, target = self._unprepare_data(output), self._unprepare_data(target)

            loss = self.loss(output, target) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics

    def _prepare_data(self, data):
        """
        :param data: shape (batch_size, num_his, num_edges, input_dim)
        :return: data: shape (num_his, batch_size, num_edges * input_dim)
        """
        data = data.permute(1, 0, 2, 3)
        batch_size = data.size(1)
        data = data.view(self.config.num_his, batch_size, self.config.num_edges * self.config.input_dim)
        return data

    def _unprepare_data(self, target):
        """
        :param target: shape (num_pred, batch_size, num_edges * output_dim
        :return: target: shape (batch_size, num_pred, num_edges, output_dim)
        """
        target = target.reshape(target.size(0), target.size(1), self.config.num_edges, self.config.output_dim)
        target = target.transpose(1, 0)
        return target