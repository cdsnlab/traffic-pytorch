import os 
from config.base_config import BaseConfig

class ASTGCN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        # Model 
        self.batch_size = 32
        self.K = 3
        self.nb_block = 2
        self.n_his = 12
        self.n_pred = 1
        self.in_channels = 1
        self.time_strides = 1
        self.nb_chev_filter = 64
        self.nb_time_filter = 64
        self.output_dim = 1

        # Model-specific hyperparameters
        self.cl_decay_steps = 2000
        self.max_grad_norm = 5

        self.use_tod = False
        self.use_dow = False
