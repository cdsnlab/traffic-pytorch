import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.rnn import GRU
from torchdiffeq import odeint
import scipy.sparse as sp

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def graph_grad(adj_mx):
    """Fetch the graph gradient operator."""
    num_nodes = adj_mx.shape[0]

    num_edges = (adj_mx > 0.).sum()
    grad = torch.zeros(num_nodes, num_edges)
    e = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mx[i, j] == 0:
                continue

            grad[i, e] = 1.
            grad[j, e] = -1.
            e += 1
    return grad

def init_network_weights(net, std = 0.1):
    """
    Just for nn.Linear net.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2
    
    res = data[..., :last_dim], data[..., last_dim:]
    return res

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def create_net(n_inputs, n_outputs, n_layers = 0, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str, device):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.deivce = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=self.device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=self.device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class ODEFunc(nn.Module):
    def __init__(self, config, adj_mx, nonlinearity='tanh'):
        super(ODEFunc, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.device = config.device

        self._num_nodes = config.num_nodes
        self._num_units = config.num_units # hidden dimension
        self._latent_dim = config.latent_dim
        self._gen_layers = config.gen_layers
        self.nfe = 0
        
        self._filter_type = config.filter_type
        if(self._filter_type == "unkP"):
            ode_func_net = create_net(config.latent_dim, config.latent_dim, n_units=config.num_units)
            init_network_weights(ode_func_net)
            self.gradient_net = ode_func_net
        else:
            self._gcn_step = config.gcn_step
            self._gconv_params = LayerParams(self, 'gconv', self.device)
            self._supports = []
            supports = []
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
            
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))
    
    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self.device)
        return L
    
    def forward(self, t_local, y, backwards = False):
        """
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point, shape (B, num_nodes * latent_dim)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * latent_dim)`.
        """
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad
    
    def get_ode_gradient_nn(self, t_local, inputs):
        if(self._filter_type == "unkP"):
            grad = self._fc(inputs)
        elif (self._filter_type == "IncP"):
            grad = - self.ode_func_net(inputs)
        else: # default is diffusion process
            # theta shape: (B, num_nodes * latent_dim)
            theta = torch.sigmoid(self._gconv(inputs, self._latent_dim, bias_start=1.0)) 
            grad = - theta * self.ode_func_net(inputs)
        return grad

    def ode_func_net(self, inputs):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv(c, self._num_units)
            c = self._activation(c)
        c = self._gconv(c, self._latent_dim)
        c = self._activation(c)
        return c
    
    def _fc(self, inputs):
        batch_size = inputs.size()[0]
        grad = self.gradient_net(inputs.view(batch_size * self._num_nodes, self._latent_dim))
        return grad.reshape(batch_size, self._num_nodes * self._latent_dim) # (batch_size, num_nodes, latent_dim)
    
    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        # state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

class DiffeqSolver(nn.Module):
    def __init__(self, config, odefunc):
        nn.Module.__init__(self)
        self.device = config.device

        self.ode_method = config.ode_method
        self.odefunc = odefunc
        self.latent_dim = config.latent_dim

        self.rtol = config.rtol
        self.atol = config.atol

    def forward(self, first_point, time_steps_to_pred):
        """
        Decoder the trajectory through the ODE Solver.

        :param time_steps_to_pred: num_pred
        :param first_point: (n_traj_samples, batch_size, num_nodes * latent_dim)
        :return: pred_y: # shape (num_pred, n_traj_samples, batch_size, self.num_nodes * self.output_dim)
        """
        n_traj_samples, batch_size = first_point.size()[0], first_point.size()[1] 
        first_point = first_point.reshape(n_traj_samples * batch_size, -1) # reduce the complexity by merging dimension
        
        # pred_y shape: (num_pred, n_traj_samples * batch_size, num_nodes * latent_dim)
        start_time = time.time()
        self.odefunc.nfe = 0
        pred_y = odeint(self.odefunc, 
                            first_point, 
                            time_steps_to_pred, 
                            rtol=self.rtol, 
                            atol=self.atol,
                            method=self.ode_method)
        time_fe = time.time() - start_time
        
        # pred_y shape: (num_pred, n_traj_samples, batch_size, num_nodes * latent_dim)
        pred_y = pred_y.reshape(pred_y.size()[0], n_traj_samples, batch_size, -1)
        # assert(pred_y.size()[1] == n_traj_samples)
        # assert(pred_y.size()[2] == batch_size)
        
        return pred_y, (self.odefunc.nfe, time_fe)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Encoder_z0_RNN(nn.Module):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)

        device = config.device
        self.device = device
        
        self.adj_mx = adj_mx
        self.num_nodes = adj_mx.shape[0]
        self.num_edges = (adj_mx > 0.).sum()
        self.gcn_step = config.gcn_step
        self.filter_type = config.filter_type
        self.num_rnn_layers = config.num_rnn_layers
        self.rnn_units = config.rnn_units
        self.latent_dim = config.latent_dim

        self.recg_type = config.recg_type # gru
        
        if(self.recg_type == 'gru'):
            # gru settings
            self.input_dim = config.input_dim
            self.gru_rnn = GRU(self.input_dim, self.rnn_units).to(device)
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        self.inv_grad = graph_grad(adj_mx).transpose(-2, -1)
        self.inv_grad[self.inv_grad != 0.] = 0.5
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),)

        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (num_his, batch_size, num_edges * input_dim)
        :return: mean, std: # shape (n_samples=1, batch_size, self.latent_dim)
        """
        if(self.recg_type == 'gru'):
            # shape of outputs: (num_his, batch, num_senor * rnn_units) 
            num_his, batch_size = inputs.size(0), inputs.size(1)
            inputs = inputs.reshape(num_his, batch_size, self.num_edges, self.input_dim)
            inputs = inputs.reshape(num_his, batch_size * self.num_edges, self.input_dim)
            
            outputs, _ = self.gru_rnn(inputs)
            last_output = outputs[-1]
            # (batch_size, num_edges, rnn_units)
            last_output = torch.reshape(last_output, (batch_size, self.num_edges, -1))
            last_output = torch.transpose(last_output, (-2, -1)) 
            # (batch_size, num_nodes, rnn_units)
            last_output = torch.matmul(last_output, self.inv_grad).transpose(-2, -1)
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")
        
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        mean = mean.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)
        std = std.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0) # for n_sample traj

class Decoder(nn.Module):
    def __init__(self, config, adj_mx):
        super(Decoder, self).__init__()
        self.num_nodes = config.num_nodes
        self.num_edges = config.num_edges
        self.grap_grad = graph_grad(adj_mx)

        self.output_dim = config.output_dim
    
    def forward(self, inputs):
        """
        :param inputs: (num_pred, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (num_pred, batch_size, num_edges * output_dim), average result of n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        num_pred, n_traj_samples, batch_size = inputs.size()[:3]
        
        inputs = inputs.reshape(num_pred, n_traj_samples, batch_size, self.num_nodes, -1).transpose(-2, -1)
        latent_dim = inputs.size(-2)
        # transform z with shape `(..., num_nodes)` to f with shape `(..., num_edges)`.
        outputs = torch.matmul(inputs, self.grap_grad)

        outputs = outputs.reshape(num_pred, n_traj_samples, batch_size, latent_dim, self.num_edges, self.output_dim)
        outputs = torch.mean(
            torch.mean(outputs, axis=3),
            axis=1
        )
        outputs = outputs.reshape(num_pred, batch_size, -1)
        return outputs


class STDENModel(nn.Module):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)

        # device 
        device = config.device
        self.device = device

        self.adj_mx = adj_mx
        self.num_nodes = adj_mx.shape[0]
        self.num_edges = (adj_mx > 0.).sum()
        self.gcn_step = config.gcn_step
        self.filter_type = config.filter_type
        self.num_rnn_layers = config.num_rnn_layers
        self.rnn_units = config.rnn_units
        self.latent_dim = config.latent_dim

        # recognition net
        self.encoder_z0 = Encoder_z0_RNN(config, adj_mx)

        # ode solver
        self.n_traj_samples = config.n_traj_samples
        self.ode_method = config.ode_method
        self.atol = config.odeint_atol
        self.rtol = config.odeint_rtol
        self.num_gen_layer = config.gen_layers
        self.ode_gen_dim = config.gen_dim

        odefunc = ODEFunc(config, adj_mx).to(device)
        self.diffeq_solver = DiffeqSolver(config, odefunc)

        self.save_latent = config.save_latent 
        self.latent_feat = None # used to extract the latent feature

        # decoder
        self.num_pred = config.num_pred
        self.out_feat = config.output_dim
        self.decoder = Decoder(config, adj_mx).to(device)

    def forward(self, inputs):
        first_point_mu, first_point_std = self.encoder_z0(inputs)

        means_z0 = first_point_mu.repeat(self.n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(self.n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

        time_steps_to_predict = torch.arange(start=0, end=self.num_pred, step=1).float().to(self.device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)

        # Shape of sol_ys (num_pred, n_traj_samples, batch_size, self.num_nodes * self.latent_dim)
        sol_ys, fe = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        if(self.save_latent):
            # Shape of latent_feat (num_pred, batch_size, self.num_nodes * self.latent_dim)
            self.latent_feat = torch.mean(sol_ys.detach(), axis=1)
        
        outputs = self.decoder(sol_ys)
        return outputs, fe