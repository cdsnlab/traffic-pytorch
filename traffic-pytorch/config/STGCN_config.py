import os 
from config.base_config import BaseConfig

class STGCN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)

        # Model 
        self.batch_size = 32
        self.stblock_num = 2
        self.Kt = 3
        self.Ks = 3 # choises [3, 2]
        self.num_nodes = 228
        self.blocks = [[1, 32, 64], [64, 32, 128]]
        self.act_func = 'glu' # choises ['glu', 'gtu']
        self.n_his = 12
        self.output_dim = 1
       
        # TODO
        self.graph_conv_type = 'graph_conv' # choices=['cheb_graph_conv', 'graph_conv']
        self.gso_type = 'rw_renorm_adj' # choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']
        self.enable_bias = True
        self.droprate = 0.0

        # self.dataset_name = dataset_name
        self.adj_mat_path = os.path.join(dataset_dir, "W_228") + ".csv" # [os.path.join(dataset_dir, "W_228") + ".csv", os.path.join(dataset_dir, "adj") + ".npz"
        self.filter_type = "dual_random_walk"

        # Model-specific hyperparameters
        self.cl_decay_steps = 2000
        self.max_grad_norm = 5

        self.use_tod = False
        self.use_dow = False