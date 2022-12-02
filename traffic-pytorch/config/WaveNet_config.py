from config.base_config import BaseConfig

class WaveNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size = 64

        # Model 
        self.dropout = 0.3
        self.supports = None
        self.gcn_bool = True
        self.addaptadj = True
        self.aptinit = None
        self.in_dim = 2
        self.residual_channels = 32
        self.dilation_channels = 32
        self.skip_channels = 256
        self.end_channels = 512
        self.kernel_size = 2
        self.blocks = 4
        self.layers = 2

        self.use_tod = True 
        self.use_dow = False 