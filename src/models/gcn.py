from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor
from torch_geometric.utils import softmax, degree, negative_sampling
from torch_geometric.nn.inits import glorot, zeros

from src.utils import create_activation


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False):
        """
        in_dim: Dimension of input features.
        num_hidden: Hidden layer dimension.
        out_dim: Dimension of output features.
        num_layers: Number of graph convolution layers.
        dropout: Dropout probability.
        activation: Activation function (or string to be converted by create_activation).
        residual: Whether to use residual connections.
        norm: A normalization layer constructor (e.g., nn.BatchNorm1d).
        encoding: Flag used to change the final layer settings.
        """
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        self.link_head = None

        # Two learnable parameters as in the original code.
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

        # Adjust settings for the last layer if encoding is enabled.
        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(in_dim, out_dim, norm=last_norm, activation=last_activation,
                                               residual=last_residual))
        else:
            # Input projection (no residual for the very first layer if desired).
            self.gcn_layers.append(GraphConv(in_dim, num_hidden, norm=norm, activation=create_activation(activation),
                                               residual=residual))
            # Hidden layers.
            for l in range(1, num_layers - 1):
                self.gcn_layers.append(GraphConv(num_hidden, num_hidden, norm=norm,
                                                   activation=create_activation(activation), residual=residual))
            # Output projection.
            self.gcn_layers.append(GraphConv(num_hidden, out_dim, norm=last_norm, activation=last_activation,
                                               residual=last_residual))

        # If extra normalization is desired outside of each layer, you might add it here.
        self.norms = None  # Not used here since each GraphConv handles normalization.
        self.head = nn.Identity()

    def forward(self, graph, x, return_hidden=False):
        edge_index = graph.edge_index
        h = x
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](graph, h, edge_index)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes, concat=False, datas_dim=16):
        dtype = next(self.parameters()).dtype  # Get the current dtype.
        if concat:
            self.head = nn.Sequential(
                nn.Linear(self.out_dim + datas_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ).to(dtype)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.out_dim, 1)
            ).to(dtype)
        self.link_head = nn.Sequential(
            nn.Linear(self.out_dim, 128)
        ).to(dtype)


class GraphConv(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 norm=None, 
                 activation=None, 
                 residual=True):
        # We use 'add' aggregation to mimic the sum aggregation in DGL (in og GraphMAE implementation).
        super(GraphConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        # This layer will be applied after aggregation.
        self.fc = nn.Linear(in_channels, out_channels)

        # Residual connection: if in_channels != out_channels, use a linear mapping.
        # self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.res_fc = nn.Linear(in_channels, out_channels, bias=False)
                print("! Linear Residual !")
            else:
                self.res_fc = nn.Identity()
                print("! Identity Residual !")
        else:
            self.register_buffer('res_fc', None)
        # Optionally include a normalization layer (e.g. nn.BatchNorm1d or nn.LayerNorm)
        self.norm_layer = norm(out_channels) if norm is not None else None

        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        # if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
        #     self.res_fc.reset_parameters()
        # if self.norm_layer is not None and hasattr(self.norm_layer, 'reset_parameters'):
        #     self.norm_layer.reset_parameters()

    def forward(self, graph, x, edge_index):
        """
        x: Node feature tensor of shape [N, in_channels]
        edge_index: Graph connectivity in COO format with shape [2, E]
        """
        
        #x = graph.x

        # Ensure the input features have the same dtype as the model parameters
        x = x.to(next(self.parameters()).dtype)

        #edge_index = graph.edge_index
        # Compute degrees for pre- and post-normalization.
        row, col = edge_index
        # Out-degree normalization for source nodes.
        deg_out = degree(row, x.size(0), dtype=x.dtype).clamp(min=1)
        norm_out = deg_out.pow(-0.5)
        # In-degree normalization for destination nodes.
        deg_in = degree(col, x.size(0), dtype=x.dtype).clamp(min=1)
        norm_in = deg_in.pow(-0.5)

        # Pre-normalize: multiply source node features by norm_out.
        x_norm = x * norm_out.view(-1, 1)

        # Message passing: here our message() just passes the (pre-normalized) neighbor features.
        out = self.propagate(edge_index, x=x_norm, num_nodes=x.size(0))

        # After aggregation, apply the linear transformation.
        out = self.fc(out)
        # Post-normalize: multiply by norm_in for the destination nodes.
        out = out * norm_in.view(-1, 1)

        # Add residual connection.
        if self.res_fc is not None:
            out = out + self.res_fc(x)

        # Apply normalization layer (if any) and activation.
        if self.norm_layer is not None:
            out = self.norm_layer(out)
        if self._activation is not None:
            out = self._activation(out)
        return out

    def message(self, x_j):
        # x_j has shape [E, in_channels]; no extra weighting is done here.
        return x_j
