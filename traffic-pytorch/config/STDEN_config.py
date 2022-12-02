from config.base_config import BaseConfig

class STDEN_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name, train_ratio, test_ratio):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name, train_ratio, test_ratio)
        self.batch_size = 32 
        self.max_grad_norm = 1.0
        self.input_dim = 1
        self.output_dim = 1 
        self.use_curriculum_learning = False

        self.use_tod = True 
        self.use_dow = True 
        
        #### Model ####
        # Encoder attributes
        self.gcn_step = 2
        self.filter_type = 'default'
        self.num_rnn_layers = 1
        self.rnn_units = 64
        self.latent_dim = 4
        self.save_latent = False
        self.recg_type = 'gru'

        # ODE solver
        self.n_traj_samples = 1
        self.ode_method = 'dopri5'
        self.odeint_atol = 1e-4
        self.odeint_rtol = 1e-3
        self.gen_layers = 1
        self.gen_dim = 64

        ### Training ### 
        self.base_lr = 1e-4 # TODO
        self.patience = 50