import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import *
from data.datasets import STGCNDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class MSTGCNTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger()
    
    def load_dataset(self):
        if os.path.exists("{}/{}_train_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
            os.path.exists("{}/{}_test_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
                os.path.exists("{}/{}_val_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)):
            print(toGreen('Found generated dataset in '+self.config.dataset_dir))
        else:    
            print(toGreen('Generating dataset...'))
            generate_train_val_test(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio, self.config.data_format)
            #generate_data_matrix(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio,
            # self.config.test_ratio, self.config.sigma1, self.config.sigma2, self.config.thres1, self.config.thres2, self.config.data_format)
        data = np.load("{}{}_train_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio))
        data = data[:, :, np.newaxis]
        self.config.num_nodes = data.shape[1]
        self.config.in_channels = data.shape[2]
        datasets = {}
        for category in ['train', 'val', 'test']:
            data = np.load("{}{}_{}_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, category, self.config.train_ratio, self.config.test_ratio))
            data = data[:, :, np.newaxis]
            x, y = seq2instance3d(data, self.config.num_his, self.config.num_pred)              
            if category == 'train':
                self.mean, self.std = np.mean(x), np.std(x)
            x = (x - self.mean) / self.std
        
            datasets[category] = {'x': x, 'y': y}
        return datasets
    
    def setup_model(self):
        adj_matrix = pd.read_csv(self.config.adj_mat_path, header=None).values.astype(float)
        L = scaled_laplacian(adj_matrix)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.config.device) for i in cheb_poly(L, self.config.K)]
        self.model = self.cls(self.config, cheb_polynomials).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = STGCNDataset(datasets[category])
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
            # TODO: Check if it is correct. Model may not be training properly
            target = target.squeeze().transpose(1,2)
            label = target[..., :self.config.output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)

            output = output.cpu()
            #print(output.shape, label.shape)
            loss = self.loss(output, label)  # loss is self-defined, need cpu input
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()
            this_metrics = self._eval_metrics(output, label)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics

    def validate_epoch(self, epoch, is_test):
        self.model.eval()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            label = target[..., :self.config.output_dim]
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                output = self.model(data)

            output = output.cpu()
            loss = self.loss(output, label)  # loss is self-defined, need cpu input

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()
            this_metrics = self._eval_metrics(output, label)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics
        
    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            # break
        return acc_metrics