import time
import math
import torch 
import numpy as np 
from data.datasets import GMANDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 

class GMANTrainer(BaseTrainer):
    def __init__(self, cls, config):
        self.config = config
        self.device = self.config.device
        self.cls = cls

    def setup_model(self):
        self.model = self.cls(self.config, self.SE).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()

        with open(self.config.dataset_dir + '/' + self.config.se_file, mode='r') as f:
            lines = f.readlines()
            temp = lines[0].split(' ')
            num_vertex, dims = int(temp[0]), int(temp[1])
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                temp = line.split(' ')
                index = int(temp[0])
                SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
            self.SE = SE 

        for category in ['train', 'val', 'test']:
            datasets[category] = GMANDataset(datasets[category])

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

        for batch_idx, (data, te, target) in enumerate(self.train_loader):
            data, te, target = data.to(self.device), te.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data, te)

            loss = self.loss(output, target)  # loss is self-defined, need cpu input
            loss.backward()
            self.optimizer.step()
            training_time = time.time() - start_time

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.cpu().detach().numpy(), target.cpu().numpy())

            print_train(epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, total_metrics)

            # TODO: logging           

    def validate_epoch(self, epoch):
        pass 