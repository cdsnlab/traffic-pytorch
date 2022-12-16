import torch
import torch.nn as nn 
import dgl
from dgl import DGLGraph
import math
import random
import numpy as np
from config import MODEL

class MLP(nn.Module):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        The initializer.

        Parameters
        ----------
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Weather apply activation function after last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        layers = []
        for i, h in enumerate(hiddens):
            layers.append(nn.Linear(h))
            if i != len(hiddens) -1:
                layers.append(act_type)
        self.layers = nn.ModuleList(layers)

class MetaDense(Block):
    """ The meta-dense layer. """
    def __init__(self, input_hidden_size, output_hidden_size, meta_hiddens, prefix=None):
        """
        The initializer.

        Parameters
        ----------
        input_hidden_size: int
            The hidden size of the input.
        output_hidden_size: int
            The hidden size of the output.
        meta_hiddens: list of int
            The list of hidden units of meta learner (a MLP).
        """
        super(MetaDense, self).__init__(prefix=prefix)
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.act_type = 'sigmoid'
        
        self.w_mlp = MLP(meta_hiddens + [self.input_hidden_size * self.output_hidden_size,], act_type=self.act_type, out_act=False, prefix='w_')
        self.b_mlp = MLP(meta_hiddens + [1,], act_type=self.act_type, out_act=False, prefix='b_')

    def forward(self, feature, data):
        """ Forward process of a MetaDense layer

        Parameters
        ----------
        feature: NDArray with shape [n, d]
        data: NDArray with shape [n, b, input_hidden_size]

        Returns
        -------
        output: NDArray with shape [n, b, output_hidden_size]
        """
        weight = self.w_mlp(feature) # [n, input_hidden_size * output_hidden_size]
        weight = nd.reshape(weight, (-1, self.input_hidden_size, self.output_hidden_size))
        bias = nd.reshape(self.b_mlp(feature), shape=(-1, 1, 1)) # [n, 1, 1]
        return nd.batch_dot(data, weight) + bias

class RNNCell(Block):
    def __init__(self, prefix):
        super(RNNCell, self).__init__(prefix=prefix)

    @staticmethod
    def create(rnn_type, pre_hidden_size, hidden_size, prefix):
        if rnn_type == 'MyGRUCell': return MyGRUCell(hidden_size, prefix)
        elif rnn_type == 'MetaGRUCell': return MetaGRUCell(pre_hidden_size, hidden_size, meta_hiddens=MODEL['meta_hiddens'], prefix=prefix)
        else: raise Exception('Unknown rnn type: %s' % rnn_type)
    
    def forward_single(self, feature, data, begin_state):
        """ Unroll the recurrent cell with one step

        Parameters
        ----------
        data: a NDArray with shape [n, b, d].
        feature: a NDArray with shape [n, d].
        begin_state: a NDArray with shape [n, b, d]

        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

        """
        raise NotImplementedError("To be implemented")

    def forward(self, feature, data, begin_state):
        """ Unroll the temporal sequence sequence.

        Parameters
        ----------
        data: a NDArray with shape [n, b, t, d].
        feature: a NDArray with shape [n, d].
        begin_state: a NDArray with shape [n, b, d]

        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, t, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

        """
        raise NotImplementedError("To be implemented")


class MyGRUCell(RNNCell):
    """ A common GRU Cell. """
    def __init__(self, hidden_size, prefix=None):
        super(MyGRUCell, self).__init__(prefix=prefix)
        self.hidden_size = hidden_size
        with self.name_scope():
            self.cell = rnn.GRUCell(self.hidden_size)

    def forward_single(self, feature, data, begin_state):
        # add a temporal axis
        data = nd.expand_dims(data, axis=2)

        # unroll
        data, state = self(feature, data, begin_state)

        # remove the temporal axis
        data = nd.mean(data, axis=2)

        return data, state

    def forward(self, feature, data, begin_state):
        n, b, length, _ = data.shape

        # reshape the data and states for rnn unroll
        data = nd.reshape(data, shape=(n * b, length, -1)) # [n * b, t, d]
        if begin_state is not None:
            begin_state = [
                nd.reshape(state, shape=(n * b, -1)) for state in begin_state
            ] # [n * b, d]
        
        # unroll the rnn
        data, state = self.cell.unroll(length, data, begin_state, merge_outputs=True)

        # reshape the data & states back
        data = nd.reshape(data, shape=(n, b, length, -1))
        state = [nd.reshape(s, shape=(n, b, -1)) for s in state]

        return data, state

class MetaGRUCell(RNNCell):
    """ Meta GRU Cell. """

    def __init__(self, pre_hidden_size, hidden_size, meta_hiddens, prefix=None):
        super(MetaGRUCell, self).__init__(prefix=prefix)
        self.pre_hidden_size = pre_hidden_size
        self.hidden_size = hidden_size
        with self.name_scope():
            self.dense_z = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_z_')
            self.dense_r = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_r_')

            self.dense_i2h = MetaDense(pre_hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_i2h_')
            self.dense_h2h = MetaDense(hidden_size, hidden_size, meta_hiddens=meta_hiddens, prefix='dense_h2h_')

    def forward_single(self, feature, data, begin_state):
        """ unroll one step

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, d].        
        begin_state: a NDArray with shape [n, b, d].
        
        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        
        """
        if begin_state is None:
            num_nodes, batch_size, _ = data.shape
            begin_state = [nd.zeros((num_nodes, batch_size, self.hidden_size), ctx=feature.context)]

        prev_state = begin_state[0]
        data_and_state = nd.concat(data, prev_state, dim=-1)
        z = nd.sigmoid(self.dense_z(feature, data_and_state))
        r = nd.sigmoid(self.dense_r(feature, data_and_state))

        state = z * prev_state + (1 - z) * nd.tanh(self.dense_i2h(feature, data) + self.dense_h2h(feature, r * prev_state))
        return state, [state]

    def forward(self, feature, data, begin_state):
        num_nodes, batch_size, length, _ = data.shape

        data = nd.split(data, axis=2, num_outputs=length, squeeze_axis=1)

        outputs, state = [], begin_state
        for input in data:
            output, state = self.forward_single(feature, input, state)
            outputs.append(output)

        outputs = nd.stack(*outputs, axis=2)
        return outputs, state

class Graph(Block):
    """ The base class of GAT and MetaGAT. We implement the methods based on DGL library. """

    @staticmethod
    def create(graph_type, dist, edge, hidden_size, prefix):
        """ create a graph. """
        if graph_type == 'None': return None
        elif graph_type == 'GAT': return GAT(dist, edge, hidden_size, prefix=prefix)
        elif graph_type == 'MetaGAT': return MetaGAT(dist, edge, hidden_size, prefix=prefix)
        else: raise Exception('Unknow graph: %s' % graph_type)

    @staticmethod
    def create_graphs(graph_type, graph, hidden_size, prefix):
        """ Create a list of graphs according to graph_type & graph. """
        if graph_type == 'None': return None
        dist, e_in, e_out = graph
        return [
            Graph.create(graph_type, dist.T, e_in, hidden_size, prefix + 'in_'),
            Graph.create(graph_type, dist, e_out, hidden_size, prefix + 'out_')
        ]

    def __init__(self, dist, edge, hidden_size, prefix=None):
        super(Graph, self).__init__(prefix=prefix)
        self.dist = dist
        self.edge = edge
        self.hidden_size = hidden_size

        # create graph
        self.num_nodes = n = self.dist.shape[0]
        src, dst, dist = [], [], []
        for i in range(n):
            for j in edge[i]:
                src.append(j)
                dst.append(i)
                dist.append(self.dist[j, i])

        self.src = src
        self.dst = dst
        self.dist = mx.nd.expand_dims(mx.nd.array(dist), axis=1)
        self.ctx = []
        self.graph_on_ctx = []

        self.init_model()    

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
        g.edata['dist'] = self.dist.as_in_context(ctx)
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)
    
    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def forward(self, state, feature): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx(state.context)
        g.ndata['state'] = state
        g.ndata['feature'] = feature        
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state

    def init_model(self):
        raise NotImplementedError("To be implemented")

    def msg_edge(self, edge):
        """ Messege passing across edge.
        More detail usage please refers to the manual of DGL library.

        Parameters
        ----------
        edge: a dictionary of edge data.
            edge.src['state'] and edge.dst['state']: hidden states of the nodes, which is NDArrays with shape [e, b, t, d] or [e, b, d]
            edge.src['feature'] and  edge.dst['state']: features of the nodes, which is NDArrays with shape [e, d]
            edge.data['dist']: distance matrix of the edges, which is a NDArray with shape [e, d]

        Returns
        -------
            A dictionray of messages
        """
        raise NotImplementedError("To be implemented")

    def msg_reduce(self, node):
        raise NotImplementedError("To be implemented")
        
class GAT(Graph):
    def __init__(self, dist, edge, hidden_size, prefix=None):
        super(GAT, self).__init__(dist, edge, hidden_size, prefix)

    def init_model(self):
        self.weight = self.params.get('weight', shape=(self.hidden_size * 2, self.hidden_size))
    
    def msg_edge(self, edge):
        state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
        ctx = state.context

        alpha = nd.LeakyReLU(nd.dot(state, self.weight.data(ctx)))

        dist = edge.data['dist']
        while len(dist.shape) < len(alpha.shape):
            dist = nd.expand_dims(dist, axis=-1)

        alpha = alpha * dist 
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1))
        return { 'new_state': new_state }

class MetaGAT(Graph):
    """ Meta Graph Attention. """
    def __init__(self, dist, edge, hidden_size, prefix=None):
        super(MetaGAT, self).__init__(dist, edge, hidden_size, prefix)

    def init_model(self):
        from model.basic_structure import MLP
        with self.name_scope():
            self.w_mlp = MLP(MODEL['meta_hiddens'] + [self.hidden_size * self.hidden_size * 2,], 'sigmoid', False)
            self.weight = self.params.get('weight', shape=(1,1))
    
    def msg_edge(self, edge):
        state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
        feature = nd.concat(edge.src['feature'], edge.dst['feature'], edge.data['dist'], dim=-1)

        # generate weight by meta-learner
        weight = self.w_mlp(feature)
        weight = nd.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size))

        # reshape state to [n, b * t, d] for batch_dot (currently mxnet only support batch_dot for 3D tensor)
        shape = state.shape
        state = nd.reshape(state, shape=(shape[0], -1, shape[-1]))

        alpha = nd.LeakyReLU(nd.batch_dot(state, weight))

        # reshape alpha to [n, b, t, d]
        alpha = nd.reshape(alpha, shape=shape[:-1] + (self.hidden_size,))
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1)) * nd.sigmoid(self.weight.data(state.context))
        return { 'new_state': new_state }

class Encoder(Block):
    """ Seq2Seq encoder. """
    def __init__(self, cells, graphs, prefix=None):
        super(Encoder, self).__init__(prefix=prefix)

        self.cells = cells
        for cell in cells:
            self.register_child(cell)

        self.graphs = graphs
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)

    def forward(self, feature, data):
        """ Encode the temporal sequence sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, t, d].        

        Returns
        -------
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        """

        _, batch, seq_len, _ = data.shape
        states = []
        for depth, cell in enumerate(self.cells):
            # rnn unroll
            data, state = cell(feature, data, None)
            states.append(state)

            # graph attention
            if self.graphs[depth] != None:
                _data = 0
                for g in self.graphs[depth]:
                    _data = _data + g(data, feature)
                data = _data

        return states

class Decoder(Block):
    """ Seq2Seq decoder. """
    def __init__(self, cells, graphs, input_dim, output_dim, use_sampling, cl_decay_steps, prefix=None):
        super(Decoder, self).__init__(prefix=prefix)
        self.cells = cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_sampling = use_sampling
        self.global_steps = 0.0
        self.cl_decay_steps = float(cl_decay_steps)

        for cell in cells:
            self.register_child(cell)

        self.graphs = graphs
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)
        
        # initialize projection layer for the output
        with self.name_scope():
            self.proj = nn.Dense(output_dim, prefix='proj_')

    def sampling(self):
        """ Schedule sampling: sampling the ground truth. """
        threshold = self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_steps / self.cl_decay_steps))
        return float(random.random() < threshold)

    def forward(self, feature, label, begin_states, is_training):
        ''' Decode the hidden states to a temporal sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        label: a NDArray with shape [n, b, t, d].
        begin_states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        is_training: bool
        
        Returns
        -------
            outputs: the prediction, which is a NDArray with shape [n, b, t, d]
        '''
        ctx = label.context

        num_nodes, batch_size, seq_len, _ = label.shape 
        aux = label[:,:,:, self.output_dim:] # [n,b,t,d]
        label = label[:,:,:, :self.output_dim] # [n,b,t,d]
        
        go = nd.zeros(shape=(num_nodes, batch_size, self.input_dim), ctx=ctx)
        output, states = [], begin_states

        for i in range(seq_len):
            # get next input
            if i == 0: data = go
            else:
                prev = nd.concat(output[i - 1], aux[:,:,i - 1], dim=-1)
                truth = nd.concat(label[:,:,i - 1], aux[:,:,i - 1], dim=-1)
                if is_training and self.use_sampling: value = self.sampling()
                else: value = 0
                data = value * truth + (1 - value) * prev

            # unroll 1 step
            for depth, cell in enumerate(self.cells):
                data, states[depth] = cell.forward_single(feature, data, states[depth])
                if self.graphs[depth] is not None:
                    _data = 0
                    for g in self.graphs[depth]:
                        _data = _data + g(data, feature)
                    data = _data / len(self.graphs[depth])

            # append feature to output
            _feature = nd.expand_dims(feature, axis=1) # [n, 1, d]
            _feature = nd.broadcast_to(_feature, shape=(0, batch_size, 0)) # [n, b, d]
            data = nd.concat(data, _feature, dim=-1) # [n, b, t, d]

            # proj output to prediction
            data = nd.reshape(data, shape=(num_nodes * batch_size, -1))
            data = self.proj(data)
            data = nd.reshape(data, shape=(num_nodes, batch_size, -1))
            
            output.append(data)

        output = nd.stack(*output, axis=2)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()

        # initialize encoder
        with self.name_scope():
            encoder_cells = []
            encoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, prefix='encoder_c%d_' % i)
                g = Graph.create_graphs('None' if i == len(rnn_hiddens) - 1 else graph_type[i], graph, hidden_size, prefix='encoder_g%d_' % i)
                encoder_cells.append(c)
                encoder_graphs.append(g)
        self.encoder = Encoder(encoder_cells, encoder_graphs)

        # initialize decoder
        with self.name_scope():
            decoder_cells = []
            decoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, prefix='decoder_c%d_' % i)
                g = Graph.create_graphs(graph_type[i], graph, hidden_size, prefix='decoder_g%d_' % i)
                decoder_cells.append(c)
                decoder_graphs.append(g)
        self.decoder = Decoder(decoder_cells, decoder_graphs, input_dim, output_dim, use_sampling, cl_decay_steps)

        # initalize geo encoder network (node meta knowledge learner)
        self.geo_encoder = MLP(geo_hiddens, act_type='relu', out_act=True, prefix='geo_encoder_')
    
    def meta_knowledge(self, feature):
        return self.geo_encoder(nd.mean(feature, axis=0))
        
    def forward(self, feature, data, label, mask, is_training):
        """ Forward the seq2seq network.

        Parameters
        ----------
        feature: NDArray with shape [b, n, d].
            The features of each node. 
        data: NDArray with shape [b, t, n, d].
            The flow readings.
        label: NDArray with shape [b, t, n, d].
            The flow labels.
        is_training: bool.


        Returns
        -------
        loss: loss for gradient descent.
        (pred, label): each of them is a NDArray with shape [n, b, t, d].

        """
        data = nd.transpose(data, axes=(2, 0, 1, 3)) # [n, b, t, d]
        label = nd.transpose(label, axes=(2, 0, 1, 3)) # [n, b, t, d]
        mask = nd.transpose(mask, axes=(2, 0, 1, 3)) # [n, b, t, d]

        # geo-feature embedding (NMK Learner)
        feature = self.geo_encoder(nd.mean(feature, axis=0)) # shape=[n, d]

        # seq2seq encoding process
        states = self.encoder(feature, data)

        # seq2seq decoding process
        output = self.decoder(feature, label, states, is_training) # [n, b, t, d]
             
        # loss calculation
        label = label[:,:,:,:self.decoder.output_dim]
        output = output * mask
        label = label * mask

        loss = nd.mean(nd.abs(output - label), axis=1, exclude=True)
        return loss, [output, label, mask]