import os
import numpy as np
from abc import abstractmethod
from util.logging import * 
import torch
from data.utils import *
import importlib

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class BaseTrainer:
    '''
    Base class for all trainers
    '''

    def load_dataset(self):
        if os.path.exists("{}/{}_train_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
            os.path.exists("{}/{}_test_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)) and \
                os.path.exists("{}/{}_val_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)):
            print(toGreen('Found generated dataset in '+self.config.dataset_dir))
        else:    
            print(toGreen('Generating dataset...'))
            generate_train_val_test(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio, self.config.use_dow, self.config.use_tod)

        datasets = {}
        for category in ['train', 'val', 'test']:
            data = np.load("{}{}_{}_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, category, self.config.train_ratio, self.config.test_ratio))
            if self.config.use_tod:
                tod = np.load("{}{}_{}_tod_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, category, self.config.train_ratio, self.config.test_ratio))
                tod = seq2instance(tod, self.config.num_his, self.config.num_pred)
            else:
                tod = None
            if self.config.use_dow:
                dow = np.load("{}{}_{}_dow_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, category, self.config.train_ratio, self.config.test_ratio))
                dow = seq2instance(dow, self.config.num_his, self.config.num_pred)
            else:
                dow = None
            x, y = seq2instance(data, self.config.num_his, self.config.num_pred)  
            
            if category == 'train':
                self.mean, self.std = np.mean(x), np.std(x)
                self.scaler = StandardScaler(self.mean, self.std)
            x = (x - self.mean) / self.std 
            datasets[category] = {'x': x, 'y': y, 'tod': tod, 'dow': dow}
        
        return datasets

    @abstractmethod 
    def compose_dataset(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod 
    def compose_loader(self, *inputs):
        raise NotImplementedError

    @abstractmethod 
    def train_epoch(self, *inputs):
        raise NotImplementedError

    @abstractmethod 
    def validate_epoch(self, *inputs):
        raise NotImplementedError
    
    def train(self):
        print(toGreen('\nSETUP TRAINING'))
        self.setup_train()
        print(toGreen('\nTRAINING START'))
        for epoch in range(self.config.total_epoch):
            self.train_epoch(epoch)
            if epoch % self.config.valid_every_epoch == self.config.valid_every_epoch-1: 
                self.validate_epoch(epoch)
        print(toGreen('\nTRAINING END'))
    
    def setup_train(self):
        # loss, metrics, optimizer, scheduler
        try:
            loss_class = getattr(importlib.import_module('evaluation.metrics'), self.config.loss)
            self.loss = loss_class(self.scaler, self.config.null_value)
            self.metrics = [getattr(importlib.import_module('evaluation.metrics'), met) for met in self.config.metrics]        
        except:
            print(toRed('No such metric in evaluation/metrics.py'))
            raise 

        try:
            # TODO Allow different types of optimizer like I did in scheduler 
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optim_class = getattr(importlib.import_module('torch.optim'), self.config.optimizer)
            self.optimizer = optim_class(trainable_params)
        except:
            print(toRed('Error loading optimizer: {}'.format(self.config.optimizer)))
            raise 

        try: 
            scheduler_class = getattr(importlib.import_module('torch.optim.lr_scheduler'), self.config.scheduler)
            scheduler_args = self.config.scheduler_args 
            scheduler_args['optimizer'] = self.optimizer
            self.lr_scheduler = scheduler_class(**scheduler_args)
        except:
            print(toRed('Error loading scheduler: {}'.format(self.config.scheduler)))
            raise 

        print_setup(self.config.loss, self.config.metrics, self.config.optimizer, self.config.scheduler)
    
    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics