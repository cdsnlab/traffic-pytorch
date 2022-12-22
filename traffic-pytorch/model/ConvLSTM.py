import torch 
import torch.nn as nn 

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
        '''
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.device = device

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        x = torch.cat([x, h_cur], dim=1)
        x = self.conv(x)
        i, f, o, g = torch.split(x, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.sigmoid(g)

        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, input_size):
        height, width = input_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device).float(),
                torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device).float())


class ConvLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_cells = config.num_cells
        cells = []
        for i in range(self.num_cells):
            dim = config.input_dim if i == 0 else config.hidden_dim[i-1]
            cells.append(ConvLSTMCell(dim, config.hidden_dim[i], config.kernel_size[i], config.bias, config.device))
        self.cells = nn.ModuleList(cells)

    def _init_hidden(self, batch_size, input_size):
        init_states = []
        for i in range(self.num_cells):
            init_states.append(self.cells[i].init_hidden(batch_size, input_size))
        return init_states

    def forward(self, x):
        '''
        x: tensor 
            [B, num_his, 2, 32, 32]
        returns: tensor 
            [B, num_pred, 2, 32, 32]
        '''
        b, num_his, _, h, w = x.size()
        hidden_state =  self._init_hidden(batch_size=b, input_size=(h, w))
        input = x 

        for idx in range(self.num_cells):
            h, c = hidden_state[idx]
            outs = None
            for t in range(num_his):
                h, c = self.cells[idx](input[:, t, :, :, :], [h,c])
                if outs == None: 
                    outs = h.unsqueeze(1)
                else: 
                    outs = torch.cat([outs, h.unsqueeze(1)], dim=1)
            input = outs 
            if idx == self.num_cells - 1:
                last_out = outs 
                last_state = [h, c]
        
        return last_out