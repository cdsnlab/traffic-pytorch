import os
import numpy as np
from abc import abstractmethod
from util.logging import * 
import torch
from data.utils import *
import importlib
import shutil
import json
from datetime import datetime

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
        num_nodes = np.load("{}/{}_train_{}_{}.npy".format(self.config.dataset_dir, self.config.dataset_name, self.config.train_ratio, self.config.test_ratio)).shape[1]
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
            x = (x - self.mean) / self.std 
            datasets[category] = {'x': x, 'y': y, 'tod': tod, 'dow': dow}
        
        return datasets, num_nodes

    @abstractmethod 
    def compose_dataset(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod 
    def compose_loader(self, *inputs):
        raise NotImplementedError

    @abstractmethod 
    def train_epoch(self, *inputs):
        raise NotImplementedError

    def validate(self, epoch, is_test=False):
        total_loss, total_metrics = self.validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        self.logger.log_validation(avg_loss, avg_metrics, epoch)
        print_total('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)

    @abstractmethod 
    def validate_epoch(self, epoch, is_test):
        raise NotImplementedError
    
    def train(self):
        print(toGreen('\nSETUP TRAINING'))
        self.setup_train()
        print(toGreen('\nTRAINING START'))
        for epoch in range(self.config.total_epoch):
            total_loss, total_metrics = self.train_epoch(epoch)
            avg_loss = total_loss / len(self.train_loader)
            avg_metrics = total_metrics / len(self.train_loader)
            self.logger.log_training(avg_loss, avg_metrics, epoch) 
            print_total('TRAIN', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)
            if epoch % self.config.valid_every_epoch == 0:
                self.validate(epoch, is_test=False)
                torch.save(self.model.state_dict(), '../results/saved_models/{}/{}.pth'.format(self.save_name, epoch))
        print(toGreen('\nTRAINING END'))
        self.validate(epoch, is_test=True)
        self.logger.close()
    
    def setup_train(self):
        # loss, metrics, optimizer, scheduler
        try:
            loss_class = getattr(importlib.import_module('evaluation.metrics'), self.config.loss)
            self.loss = loss_class()
            self.metrics = [getattr(importlib.import_module('evaluation.metrics'), met)() for met in self.config.metrics]        
        except:
            print(toRed('No such metric in evaluation/metrics.py'))
            raise 

        try:
            # TODO Allow different types of optimizer like I did in scheduler 
            optim_class = getattr(importlib.import_module('torch.optim'), self.config.optimizer)
            self.optimizer = optim_class(self.model.parameters(), lr=1e-4)
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
        acc_metrics = []
        for metric in self.metrics:
            with torch.no_grad():
                acc_metrics.append(metric(output, target))
        return acc_metrics

    def setup_save(self, args):
        self.save_name = '{}_{}'.format(args.model, datetime.now().strftime("%d_%H_%M_%S"))
        save_dir = '../results/saved_models/{}/'.format(self.save_name)
        os.makedirs(save_dir)
        shutil.copy('config/{}.py'.format(args.config), save_dir + 'config.txt')
        with open(save_dir + 'cmd_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)