# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.device)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        #print("X shape: ", x.shape)
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.device)  # (b, N, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                temp = graph_signal.permute(0, 2, 1)
                temp = temp.matmul(T_k)
                rhs = temp.permute(0, 2, 1)
                #print(theta_k.shape, temp.shape, output.shape)
                temp = rhs.matmul(theta_k)
                output = output + temp
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCN_block(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = cheb_conv(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)
        return x_residual


class MSTGCN_submodule(nn.Module):
    def __init__(self, device, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(MSTGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList([MSTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials)])
        self.BlockList.extend([MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


class MSTGCNModel(nn.Module):
    def __init__(self, config, cheb_polynomials):
        '''
        in config
        :param device:
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        :param len_input
        :return:
        '''
        super(MSTGCNModel, self).__init__()
        config.in_channels = 3
        self.submodule = MSTGCN_submodule(config.device, config.nb_block, config.in_channels, config.K, config.nb_chev_filter,\
         config.nb_time_filter, config.time_strides, cheb_polynomials, config.n_pred, config.n_his)
        for p in self.submodule.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        x = x.squeeze(1)
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        return self.submodule(x)
