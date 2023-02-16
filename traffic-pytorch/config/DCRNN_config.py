import os 
from config.base_config import BaseConfig

class DCRNN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        # Model 
        self.batch_size = 64 
        self.enc_input_dim = 2
        self.dec_input_dim = 1
        self.max_diffusion_step = 2
        self.num_rnn_layers = 2
        self.rnn_units = 64
        self.output_dim = 1
        self.adj_mat_path = os.path.join(dataset_dir, dataset_name) + ".pkl"
        self.filter_type = "dual_random_walk"

        # Model-specific hyperparameters
        self.cl_decay_steps = 2000
        self.max_grad_norm = 5

        self.use_tod = True 
        self.use_dow = False
        self.total_epoch = 24
