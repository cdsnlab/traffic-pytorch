from abc import abstractmethod
from util.logging import * 
import importlib

class BaseTrainer:
    '''
    Base class for all trainers
    '''

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
        