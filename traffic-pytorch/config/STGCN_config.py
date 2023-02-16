import os 
from config.base_config import BaseConfig

class STGCN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        # Model 
        self.batch_size = 32
        self.Kt = 3
        self.Ks = 3 # choises [3, 2]
        # not sure how the blocks should be constructed
        self.blocks = [[1, 32, 64], [64, 32, 128]]
        self.n_his = 12
        self.output_dim = 1
       
        # TODO
        self.enable_bias = True
        self.droprate = 0.3

        # self.dataset_name = dataset_name
        if "PEMS-M" in dataset_name or "PEMSD" in dataset_name:
            self.dataset_name = "V_228"
            self.data_format = 'csv'
            self.adj_mat_path = os.path.join(dataset_dir, "W_228") + ".csv" # [os.path.join(dataset_dir, "W_228") + ".csv", os.path.join(dataset_dir, "adj") + ".npz"
        else:
            self.adj_mat_path = os.path.join(dataset_dir, dataset_name) + ".csv"

        # Model-specific hyperparameters
        self.cl_decay_steps = 2000
        self.max_grad_norm = 5

        self.use_tod = False
        self.use_dow = False
        self.total_epoch = 150
        self.scheduler_args = {
			"milestones": [50, 100],
            "gamma": 0.5
        }
