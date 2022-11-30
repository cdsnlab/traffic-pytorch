import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class WaveNetModel(nn.Module):
    def __init__(self, config):
        super(WaveNetModel, self).__init__()

        self.batch_size = config.batch_size
        self.dropout = config.dropout
        self.blocks = config.blocks
        self.layers = config.layers
        self.gcn_bool = config.gcn_bool
        self.addaptadj = config.addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=config.in_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1,1))
        self.supports = config.supports

        receptive_field = 1

        self.supports_len = 0
        if config.supports is not None:
            self.supports_len += len(config.supports)

        if config.gcn_bool and config.addaptadj:
            if config.aptinit is None:
                if config.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(config.num_nodes, 10).to(config.device), requires_grad=True).to(config.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, config.num_nodes).to(config.device), requires_grad=True).to(config.device)
                self.supports_len +=1
            else:
                if config.supports is None:
                    self.supports = []
                m, p, n = torch.svd(config.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(config.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(config.device)
                self.supports_len += 1

        for b in range(config.blocks):
            additional_scope = config.kernel_size - 1
            new_dilation = 1
            for i in range(config.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=config.residual_channels,
                                                   out_channels=config.dilation_channels,
                                                   kernel_size=(1, config.kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=config.residual_channels,
                                                 out_channels=config.dilation_channels,
                                                 kernel_size=(1, config.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=config.dilation_channels,
                                                     out_channels=config.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=config.dilation_channels,
                                                 out_channels=config.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(config.residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(config.dilation_channels,config.residual_channels,config.dropout,support_len=self.supports_len))


        self.end_conv_1 = nn.Conv2d(in_channels=config.skip_channels,
                                  out_channels=config.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=config.end_channels,
                                    out_channels=config.num_pred,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)

            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
