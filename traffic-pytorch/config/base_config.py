class BaseConfig: 
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Device
        self.device = device 

        # Data
        self.data_format = 'h5'
        self.test_batch_size = 1
        self.num_his = 12 
        self.num_pred = 3
        self.dataset_dir = dataset_dir 
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio 
        self.test_ratio = test_ratio

        # Temporal features
        self.use_tod = True 
        self.use_dow = True 

        # Train 
        self.optimizer = 'Adam'
        self.loss = 'MaskedMAE'
        self.metrics = ['MaskedRMSE', 'MaskedMAE']
        self.scheduler = 'MultiStepLR'
        self.scheduler_args = {
			"milestones": [8, 16],
            "gamma": 0.5
        }
        self.model_checkpoint = None # checkpoint path if continue training
        self.null_value = 0.0
        self.start_epoch = 0
        self.total_epoch = 24 
        self.valid_every_epoch = 4 # Validate epoch