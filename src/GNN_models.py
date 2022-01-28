import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

torch.autograd.set_detect_anomaly(True)

from DGCNConv import DGCNConv
from DiGCN_Inception_Block import DiGCN_InceptionBlock as InceptionBlock

class DGCN(torch.nn.Module):
    r"""An implementation of the DGCN node classification model from `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
    Args:
        input_dim (int): Dimention of input features.
        filter_num (int): Hidden dimention.
        out_dim (int): Output dimension.
        dropout (float, optional): Dropout value. Default: None.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
    """
    def __init__(self, input_dim: int, filter_num: int, out_dim: int, dropout: Optional[float]=None, \
        improved: bool = False, cached: bool = False):
        super(DGCN, self).__init__()
        self.dropout = dropout
        self.dgconv = DGCNConv(improved=improved, cached=cached)
        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim,    filter_num,   bias=False)
        self.lin2 = torch.nn.Linear(filter_num*3, filter_num, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        self.bias2 = nn.Parameter(torch.Tensor(1, filter_num))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def forward(self, x: torch.FloatTensor, \
        edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor], \
        edge_weight_tuple: Tuple[Union[torch.FloatTensor, None], Union[torch.FloatTensor, None]]) -> torch.FloatTensor:
        """
        Making a forward pass of the DGCN node classification model from `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_in, edge_out (PyTorch LongTensor) - Edge indices for input and output directions, respectively.
            * in_w, out_w (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, 
                with shape (num_nodes, num_classes).
        """
        edge_index, edge_in, edge_out = edge_index_tuple
        in_w, out_w = edge_weight_tuple
        x = self.lin1(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)
        
        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)


@torch.jit._overload
def conv_norm_rw(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def conv_norm_rw(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def conv_norm_rw(edge_index, fill_value=0.5, edge_weight=None, num_nodes=None,
             add_self_loops=True, dtype=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row = edge_index[0]
    row_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = row_deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    return edge_index, deg_inv[row] * edge_weight


class DIMPA_Base(MessagePassing):
    r"""The base class for directed mixed-path aggregation model.
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        fill_value (float, optional): The layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + fill_value*\mathbf{I}`.
            (default: :obj:`0.5`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1} \mathbf{\hat{A}}` 
            on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 fill_value: float = 0.5, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super(DIMPA_Base, self).__init__(**kwargs)

        self.fill_value = fill_value
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = conv_norm_rw(  # yapf: disable
                    edge_index, self.fill_value, edge_weight, x.size(self.node_dim),
                    self.add_self_loops)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class DIMPA(torch.nn.Module):
    r"""The directed mixed-path aggregation model.
    Args:
        hop (int): Number of hops to consider.
    """

    def __init__(self, hop: int,
                fill_value: float = 0.5):
        super(DIMPA, self).__init__()
        self._hop = hop
        self._w_s = Parameter(torch.FloatTensor(hop + 1, 1))
        self._w_t = Parameter(torch.FloatTensor(hop + 1, 1))
        self.conv_layer = DIMPA_Base(fill_value)


        self._reset_parameters()

    def _reset_parameters(self):
        self._w_s.data.fill_(1.0)
        self._w_t.data.fill_(1.0)

    def forward(self, x_s: torch.FloatTensor, x_t: torch.FloatTensor,
                edge_index: torch.FloatTensor, 
                edge_weight: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of DIMPA.
        Arg types:
            * **x_s** (PyTorch FloatTensor) - Souce hidden representations.
            * **x_t** (PyTorch FloatTensor) - Target hidden representations.
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim).
        """
        feat_s = self._w_s[0]*x_s
        feat_t = self._w_t[0]*x_t
        curr_s = x_s.clone()
        curr_t = x_t.clone()
        edge_index_t = edge_index[[1,0]]
        for h in range(1, 1+self._hop):
            curr_s = self.conv_layer(curr_s, edge_index, edge_weight)
            curr_t = self.conv_layer(curr_t, edge_index_t, edge_weight)
            feat_s += self._w_s[h]*curr_s
            feat_t += self._w_t[h]*curr_t

        feat = torch.cat([feat_s, feat_t], dim=1)  # concatenate results

        return feat

class DIGRAC(torch.nn.Module):
    r"""The directed graph clustering model.
    Args:
        nfeat (int): Number of features.
        hidden (int): Hidden dimensions of the initial MLP.
        nclass (int): Number of clusters.
        dropout (float): Dropout probability.
        hop (int): Number of hops to consider.
        fill_value (float): Value for added self-loops.
    """

    def __init__(self, nfeat: int, hidden: int, nclass: int, fill_value: float, dropout: float, hop: int):
        super(DIGRAC, self).__init__()
        nh1 = hidden
        nh2 = hidden
        self._num_clusters = int(nclass)
        self._w_s0 = Parameter(torch.FloatTensor(nfeat, nh1))
        self._w_s1 = Parameter(torch.FloatTensor(nh1, nh2))
        self._w_t0 = Parameter(torch.FloatTensor(nfeat, nh1))
        self._w_t1 = Parameter(torch.FloatTensor(nh1, nh2))

        self._dimpa = DIMPA(hop, fill_value)
        self._relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

        self._prob_linear = torch.nn.Linear(2*nh2, nclass)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_s0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_s1, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t1, gain=1.414)

        self._prob_linear.reset_parameters()

    def forward(self, features: torch.FloatTensor, edge_index: torch.FloatTensor,
                edge_weight: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of the DIGRAC.
        Arg types:
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
        Return types:
            * **output** (PyTorch FloatTensor) - Log of the probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        # MLP
        x_s = torch.mm(features, self._w_s0)
        x_s = self._relu(x_s)
        x_s = self.dropout(x_s)
        x_s = torch.mm(x_s, self._w_s1)

        x_t = torch.mm(features, self._w_t0)
        x_t = self._relu(x_t)
        x_t = self.dropout(x_t)
        x_t = torch.mm(x_t, self._w_t1)

        z = self._dimpa(x_s, x_t, edge_index, edge_weight)

        output = self._prob_linear(z)
        
        output = F.log_softmax(output, dim=1)

        return output

class DiGCN_Inception_Block(torch.nn.Module):
    r"""An implementation of the DiGCN model with inception blocks for ranking from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **embedding_dim** (int) - Embedding dimension.
        * **prob_dim** (int) - Dimension of the probability matrix.
    """
    def __init__(self, num_features: int, dropout: float, 
                embedding_dim: int, prob_dim: int):
        super(DiGCN_Inception_Block, self).__init__()
        self.ib1 = InceptionBlock(num_features, embedding_dim)
        self.ib2 = InceptionBlock(embedding_dim, embedding_dim)
        self.ib3 = InceptionBlock(embedding_dim, prob_dim)
        self._dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, features: torch.FloatTensor, \
        edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor], \
        edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN model with inception blocks modified from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
            * features (PyTorch FloatTensor) - Node features.
            
        Return types:
            * x (PyTorch FloatTensor) - Log class probabilities for all nodes, 
                with shape (num_nodes, prob_dim).
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2

        return F.log_softmax(x, dim=1)