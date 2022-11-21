class BaseConfig: 
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Device
        self.device = device 

        # Data
        self.test_batch_size = 1
        self.dataset_dir = dataset_dir 
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio 
        self.test_ratio = test_ratio

        # Train 
        self.optimizer = 'Adam'
        self.loss = 'masked_mae_loss'
        # self.metrics = ["masked_mae_np", "masked_mape_np", "masked_rmse_np"]
        self.metrics = []
        self.scheduler = 'MultiStepLR'
        self.scheduler_args = {
			"milestones": [20, 30, 40, 50],
            "gamma": 0.1
        }
        self.total_epoch = 60 
        self.valid_every_epoch = 4 # Validate epoch