import time
import math
import torch 
import numpy as np 
from data.utils import *
from data.datasets import WaveNetDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 

class WaveNetTrainer(BaseTrainer):
    def __init__(self, cls, config):
        self.config = config
        self.device = self.config.device
        self.cls = cls

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = WaveNetDataset(datasets[category])

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
            data = data.transpose(1, 3)
            target = target.transpose(1, 3)
            
            data = data.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            output = output.transpose(1, 3)

            output *= self.std 
            output += self.mean

            loss = self.loss(output.cpu(), target[:, 0, :, :])
            loss.backward()

            self.optimizer.step()
            training_time = time.time() - start_time

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().cpu().numpy(), target[:, 0, :, :].numpy())

            print_train(epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, total_metrics)

            # TODO: logging           

    def validate_epoch(self, epoch):
        pass 
    
    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            # break
        return acc_metrics