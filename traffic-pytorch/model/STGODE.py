import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



class ODEFunc(nn.Module):
    '''
    Define the ODE function.
    :param t: A tensor with shape [], meaning the current time.
    :param x: A tensor with shape [#batches, dims], meaning the value of x at t.
    :output dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
    '''
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z



class ODEG(nn.Module):
    '''
    ODEGCN model.
    '''
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs : channel's number of input data's feature
        :param num_channels : numbers of data feature tranform channels, the last is the output channel
        :param kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        '''
        like ResNet
        :param X : input data of shape (B, N, T, F) 
        '''
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels,):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()
    
    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat):
        """
        :param in_channels: Number of input features at each node in each time step.
        :param out_channels: a list of feature channels in timeblock, the last is output feature channel
        :param num_nodes: Number of nodes in the graph
        :param A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 12, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
                                   
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        :output Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))

        return self.batch_norm(t)


class STGODEModel(nn.Module):
    """ the overall network framework """
    def __init__(self, config, A_sp_hat, A_se_hat):
        """ 
        :param num_nodes : number of nodes in the graph
        :param num_features : number of features at each node in each time step
        :param num_timesteps_input : number of past time steps fed into the network
        :param num_timesteps_output : desired number of future time steps output by the network
        :param A_sp_hat : nomarlized adjacency spatial matrix
        :param A_se_hat : nomarlized adjacency semantic matrix
        """        
        super(STGODEModel, self).__init__()
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=config.num_features, out_channels=[64, 32, 64],
                num_nodes=config.num_nodes, A_hat=A_sp_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=config.num_nodes, A_hat=A_sp_hat)) for _ in range(3)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
                STGCNBlock(in_channels=config.num_features, out_channels=[64, 32, 64],
                num_nodes=config.num_nodes, A_hat=A_se_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=config.num_nodes, A_hat=A_se_hat)) for _ in range(3)
            ]) 

        self.pred = nn.Sequential(
            nn.Linear(config.num_his * 64, config.num_pred * 32), 
            nn.ReLU(),
            nn.Linear(config.num_pred * 32, config.num_pred)
        )

    def forward(self, x):
        """
        :param x : input data of shape (batch_size, num_nodes, num_his, num_his) == (B, N, T, F)
        prediction for future of shape (batch_size, num_nodes, num_pred)
        """
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))

        return self.pred(x)