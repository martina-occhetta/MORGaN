from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.conv import MessagePassing
from src.utils import create_activation

class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge_types, num_layers, dropout, activation, residual=True, norm=None, encoding=False):
        """
        A multi-layer RGCN model for homogeneous graphs with relation types encoded as integers.
        
        Args:
            in_channels (int): Input node feature dimension.
            hidden_channels (int): Hidden layer dimension.
            out_channels (int): Output node feature dimension.
            num_edge_types (int): Number of distinct relation types.
            num_layers (int): Number of RGCN layers.
            dropout (float): Dropout probability.
            activation (callable): Activation function (e.g., torch.relu).
            residual (bool): Whether to use residual connections.
            norm (callable or None): Normalization layer constructor.
            encoding (bool): If True, apply activation, residual, and norm in the last layer.
        """
        super(RGCN, self).__init__()

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rgcn_layers = nn.ModuleList()
        self.num_edge_types = num_edge_types
        
        self.dropout = dropout

        self.activation = create_activation(activation)
        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.rgcn_layers.append(RGCNConv(
                in_channels, out_channels, 
                num_edge_types, dropout=dropout, 
                residual=last_residual, norm=last_norm))
        else:
            # input projection
            self.rgcn_layers.append(RGCNConv(in_channels, 
                    hidden_channels, num_edge_types, dropout=dropout,
                    activation=create_activation(activation),
                    residual=residual, norm=norm))          
            
            #hidden layers
            for l in range(1, num_layers - 1):
                self.rgcn_layers.append(RGCNConv(hidden_channels, 
                    hidden_channels, num_edge_types, dropout=dropout,
                    activation=create_activation(activation),
                    residual=residual, norm=norm))
            # output projection
            self.rgcn_layers.append(RGCNConv(hidden_channels, 
                    out_channels, num_edge_types, 
                    dropout=dropout,
                    activation=last_activation,
                    residual=last_residual, norm=last_norm))       
        
        self.head = nn.Identity()
    
    def forward(self, x, edge_index, edge_type, return_hidden=False):
        """
        Args:
            x (Tensor): Node feature tensor of shape [N, in_channels].
            edge_index (LongTensor): Edge indices of shape [2, E].
            edge_type (LongTensor): Edge type tensor of shape [E].
            return_hidden (bool): If True, also return hidden representations from each layer.
            
        Returns:
            Tensor: Final node representations.
            (optional) List[Tensor]: List of hidden representations per layer.
        """
        h = x
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.rgcn_layers[l](h, edge_index, edge_type)
            hidden_list.append(h)
        out = self.head(h)
        if return_hidden:
            return out, hidden_list
        else:
            return out
    
    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_channels, num_classes)

class RGCNConv(MessagePassing):
    def __init__(self, 
                 in_channels: Union[int, Tuple[int,int]], 
                 out_channels: int, 
                 num_edge_types: Optional[int] = 1, 
                 dropout: float = 0.0,
                 bias: bool = True,
                 activation = None,
                 residual=False, 
                 norm=None,
                 **kwargs
                ):
        """
        A relational graph convolution layer for homogeneous graphs with integer-encoded edge types.
        
        Args:
            in_channels (int): Dimension of input node features.
            out_channels (int): Dimension of output node features.
            num_edge_types (int): Number of distinct relation types.
            residual (bool): Whether to use residual connection.
            norm (callable or None): Normalization layer constructor.
            bias (bool): Whether to include a bias term.
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        #super(RGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.dropout = dropout

        self.activation = activation
        
        # Each relation type has its own weight matrix.
        self.weight = nn.Parameter(torch.Tensor(num_edge_types, in_channels, out_channels))
        # Self-loop (or root) transformation.
        self.self_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # TODO test
        self.fc = nn.Linear(in_channels, out_channels)

        # Set up residual connection.
        self.residual = residual
        if residual:
            if self.in_channels != out_channels:
                self.res_fc = nn.Linear(self.in_channels, out_channels, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None
        
        # Set up normalization layer.
        self.norm = norm(out_channels) if norm is not None else None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        xavier_uniform_(self.weight) # TODO check difference between xavier uniform and xavier normal
        xavier_uniform_(self.self_weight)

        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            xavier_uniform_(self.res_fc.weight)

        if self.bias is not None:
            zeros_(self.bias)
    
    def forward(self, x: Union[Tensor, OptPairTensor], 
                edge_index: Adj, 
                edge_type):
        """
        Args:
            x (Tensor): Node features of shape [N, in_channels].
            edge_index (LongTensor): Edge indices with shape [2, E].
            edge_type (LongTensor): Edge type tensor with shape [E] and integer values in [0, num_edge_types-1].
            
        Returns:
            Tensor: Updated node features of shape [N, out_channels].
        """
        # Self-loop transformation.
        out = torch.matmul(x, self.self_weight)
        
        # Propagate messages from neighbors using relation-specific weight matrices.
        out = out + self.propagate(edge_index, x=x, edge_type=edge_type)
        
        #out = self.fc(out)

        # Add residual connection if enabled
        if self.res_fc is not None:
            out = out + self.res_fc(x)
        
        # Apply normalization if provided
        if self.norm is not None:
            out = self.norm(out)
        
        if self.activation is not None:
            out = self.activation(out)
        
        # Add bias if provided
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_j, edge_type):
        # For each edge, select its corresponding weight matrix.
        # x_j: shape [E, in_channels]
        # edge_type: shape [E]
        weight = self.weight[edge_type]  # shape [E, in_channels, out_channels]
        # Multiply each source node feature by the corresponding weight.
        # Unsqueeze x_j to shape [E, 1, in_channels] then batch multiply.
        msg = torch.bmm(x_j.unsqueeze(1), weight)  # shape [E, 1, out_channels]
        return msg.squeeze(1)
