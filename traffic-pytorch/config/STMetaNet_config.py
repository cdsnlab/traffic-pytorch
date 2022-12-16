from config.base_config import BaseConfig

class STMetaNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio, mode):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        if mode == 'traffic':
            self.input_dim = 9
            self.output_dim = 1
            self.n_neighbors = 8
            self.geo_hiddens = [32, 32]
            self.rnn_hiddens = [32, 32]
            self.batch_size = 32
            self.early_stop_epoch = 50

        if mode == 'flow':
            self.input_dim = 3
            self.output_dim = 2
            self.geo_hiddens = [32, 32]
            self.rnn_hiddens = [64, 64]
            self.batch_size = 16
            self.early_stop_epoch = 200

        self.wd = 0
        self.lr = 0.01
        self.lr_decay_step = 7020
        self.lr_decay_factor = 0.1
        self.lr_min = 0.000002
        self.use_sampling: True

        self.max_grad_norm = 5
        self.cl_decay_steps = 2000

        self.early_stop_metric = 'rmse'
