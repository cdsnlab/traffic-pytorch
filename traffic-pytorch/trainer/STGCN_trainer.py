import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import *
from data.datasets import STGCNDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer, StandardScaler
from util.logging import * 
from logger.logger import Logger

class STGCNTrainer(BaseTrainer):
    def __init__(self, cls, config):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.logger = Logger()
    
    def load_dataset(self):
        if os.path.exists("{}/{}_train_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
            os.path.exists("{}/{}_test_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
                os.path.exists("{}/{}_val_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)):
            print(toGreen('Found generated dataset in '+self.config.dataset_dir))
        else:    
            print(toGreen('Generating dataset...'))
            generate_train_val_test_csv(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)
            
        datasets = {}
        for category in ['train', 'val', 'test']:
            data = np.load("{}{}_{}_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, category, self.config.train_ratio, self.config.test_ratio))
            x, y = seq2instance(data, self.config.num_his, self.config.num_pred)  
            
            if category == 'train':
                self.mean, self.std = np.mean(x), np.std(x)
                self.scaler = StandardScaler(self.mean, self.std)
            x = (x - self.mean) / self.std 
            datasets[category] = {'x': x, 'y': y}
        return datasets
    
    def setup_model(self):
        Ko = self.config.n_his - (self.config.Kt - 1) * 2 * self.config.stblock_num
        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        '''
        blocks = []
        blocks.append([1])
        for l in range(self.config.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([1])
        '''
        blocks = self.config.blocks
        Lk = get_matrix(self.config.adj_mat_path, self.config.Ks).to(self.device)
        self.model = self.cls(self.config, blocks, Lk).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = STGCNDataset(datasets[category])

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
            label = target[..., :self.config.output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.num_train_iteration_per_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.config.cl_decay_steps)

            output = self.model(data)
            '''
            output = torch.transpose(output.view(12, self.config.batch_size, self.config.num_nodes, 
                            self.config.output_dim), 0, 1)  # back to (50, 12, 207, 1)
            '''
            output = output.cpu()
            loss = self.loss(output, label)  # loss is self-defined, need cpu input
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())

            print_train(epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, total_metrics)

        # TODO: logging   
        if epoch % self.config.valid_every_epoch == 0:
            avg_loss = total_loss / len(self.train_loader)
            avg_metrics = total_metrics / len(self.train_loader)
            self.logger.log_training(avg_loss, avg_metrics, epoch) 
            self.validate_epoch(epoch)


    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.val_loader):
            label = target[..., :self.config.output_dim]
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.num_train_iteration_per_epoch + batch_idx

            output = self.model(data)
            output = output.cpu()
            loss = self.loss(output, label)  # loss is self-defined, need cpu input
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = total_metrics / len(self.val_loader)
        self.logger.log_validation(avg_loss, avg_metrics, epoch)
        #print_val(epoch, self.config.total_epoch, avg_loss, self.config.metrics, avg_metrics)


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
            # break
        return acc_metrics